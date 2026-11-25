#!/usr/bin/env python3
"""Score a submitted boundary for the compact vacuum stellarator task.

    python eval.py boundary.json --workdir DIR [--reward-dir /logs/verifier]

The submission is JSON:

    {"nfp": 2,
     "modes": [{"m": 0, "n": 0, "rbc": 1.0, "zbs": 0.0}, ...]}

with R = sum rbc cos(m theta - n nfp phi), Z = sum zbs sin(m theta - n nfp phi),
0 <= m <= 12, |n| <= 12, n >= 0 when m = 0, zbs(0,0) = 0.  The scale is free:
the evaluator rescales the boundary to a = 1.70 m.

Every scored number comes from a maintained program: VMEC2000, booz_xform,
NEO, SIMPLE and SIMSOPT.  This file only prepares their inputs, runs them, reads
their outputs and applies the published score.  A stage that cannot finish
makes the submission invalid (score 0) with the reason recorded; nothing is
estimated in its place.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MINOR_RADIUS = 1.70  # m
AXIS_FIELD = 5.70  # T, Boozer B00 on axis
M_MAX, N_MAX, NFP_MAX = 12, 12, 6

# Constraint limits of the problem statement.
F_ALPHA_MAX = 0.05
EPS_EFF_MAX = 0.01
IOTA_MIN = 0.30
WELL_MIN = 0.0
KAPPA_MAX = 8.0
LGRADB_MIN = 2.4  # m at a = 1.70 m: coils about 1.2 m from the plasma (Kappel et al. 2023 fit)
WELL_SCALE = 0.01  # progress scale of the well term only
A_MID = 4.0  # aspect ratio at which a feasible boundary scores 0.75

# Geometry grid of the problem statement: per field period.
GEOM_NPHI, GEOM_NTHETA = 256, 512


class Invalid(Exception):
    """The submission cannot be scored; the message is the reason."""


# ---------------------------------------------------------------- submission


@dataclass(frozen=True)
class Boundary:
    nfp: int
    rbc: dict  # (m, n) -> float
    zbs: dict

    def _present(self) -> list:
        return [k for k in self.rbc if self.rbc[k] != 0 or self.zbs[k] != 0]

    @property
    def m_max(self) -> int:
        return max(m for m, n in self._present())

    @property
    def n_max(self) -> int:
        return max(abs(n) for m, n in self._present())

    def scaled(self, factor: float) -> "Boundary":
        return Boundary(
            self.nfp,
            {k: v * factor for k, v in self.rbc.items()},
            {k: v * factor for k, v in self.zbs.items()},
        )


def read_boundary(path: Path) -> Boundary:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        raise Invalid(f"submission file {path} does not exist")
    except (OSError, ValueError) as exc:
        raise Invalid(f"submission file {path} is not readable JSON: {exc}")
    if not isinstance(data, dict) or set(data) != {"nfp", "modes"}:
        raise Invalid('submission must be an object with exactly the keys "nfp" and "modes"')
    nfp = data["nfp"]
    if not isinstance(nfp, int) or isinstance(nfp, bool) or not 1 <= nfp <= NFP_MAX:
        raise Invalid(f"nfp must be an integer in 1..{NFP_MAX}, got {nfp!r}")
    modes = data["modes"]
    if not isinstance(modes, list) or not modes:
        raise Invalid('"modes" must be a non-empty list')
    rbc, zbs = {}, {}
    for i, mode in enumerate(modes):
        if not isinstance(mode, dict) or set(mode) != {"m", "n", "rbc", "zbs"}:
            raise Invalid(f'modes[{i}] must have exactly the keys "m", "n", "rbc", "zbs"')
        m, n = mode["m"], mode["n"]
        for name, value in (("m", m), ("n", n)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise Invalid(f"modes[{i}].{name} must be an integer")
        if not (0 <= m <= M_MAX and abs(n) <= N_MAX):
            raise Invalid(f"modes[{i}] has (m, n) = ({m}, {n}) outside 0<=m<={M_MAX}, |n|<={N_MAX}")
        if m == 0 and n < 0:
            raise Invalid(f"modes[{i}] has m = 0 with n < 0; use n >= 0 for m = 0")
        if (m, n) in rbc:
            raise Invalid(f"mode (m, n) = ({m}, {n}) appears twice")
        values = []
        for name in ("rbc", "zbs"):
            value = mode[name]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise Invalid(f"modes[{i}].{name} must be a finite number")
            try:
                value = float(value)
            except OverflowError:
                raise Invalid(f"modes[{i}].{name} must be a finite number")
            if not math.isfinite(value):
                raise Invalid(f"modes[{i}].{name} must be a finite number")
            values.append(value)
        rbc[(m, n)], zbs[(m, n)] = values
    if zbs.get((0, 0), 0.0) != 0.0:
        raise Invalid("zbs(0, 0) must be 0")
    if rbc.get((0, 0), 0.0) <= 0.0:
        raise Invalid("rbc(0, 0) must be positive")
    return Boundary(nfp, rbc, zbs)


# ------------------------------------------------------------------ geometry


def _surface(b: Boundary, theta: np.ndarray, phi: np.ndarray):
    """R, Z and dZ/dtheta on the grid theta x phi (indexing [phi, theta])."""
    th, ph = np.meshgrid(theta, phi)
    R = np.zeros_like(th)
    Z = np.zeros_like(th)
    Zt = np.zeros_like(th)
    for (m, n), rc in b.rbc.items():
        zs = b.zbs[(m, n)]
        angle = m * th - n * b.nfp * ph
        c, s = np.cos(angle), np.sin(angle)
        R += rc * c
        Z += zs * s
        Zt += m * zs * c
    return R, Z, Zt


def _segments_cross(P: np.ndarray) -> bool:
    """True if the closed polygon P (N x 2) has two non-adjacent crossing edges."""
    A = P
    B = np.roll(P, -1, axis=0)
    N = len(P)

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (
            r[..., 0] - p[..., 0]
        )

    i, j = np.triu_indices(N, k=2)
    keep = ~((i == 0) & (j == N - 1))
    i, j = i[keep], j[keep]
    for lo in range(0, len(i), 2_000_000):
        a, b = A[i[lo:lo + 2_000_000]], B[i[lo:lo + 2_000_000]]
        c, d = A[j[lo:lo + 2_000_000]], B[j[lo:lo + 2_000_000]]
        d1, d2 = orient(a, b, c), orient(a, b, d)
        d3, d4 = orient(c, d, a), orient(c, d, b)
        if np.any((d1 * d2 <= 0) & (d3 * d4 <= 0)):
            return True
    return False


def geometry(b: Boundary) -> dict:
    """Aspect ratio, minor radius and elongation by the problem's definitions.

    With theta and phi uniform and periodic, the trapezoidal rule integrates
    the trigonometric polynomials below exactly (their degree is at most
    4 * 12 = 48, far below the 512 x 256 grid), so the values are exact to
    rounding.  Section moments use Green's theorem on the section curve
    theta -> (R, Z) at fixed cylindrical angle phi.
    """
    theta = 2 * np.pi * np.arange(GEOM_NTHETA) / GEOM_NTHETA
    phi = (2 * np.pi / b.nfp) * np.arange(GEOM_NPHI) / GEOM_NPHI
    R, Z, Zt = _surface(b, theta, phi)
    if R.min() <= 0:
        raise Invalid(f"boundary reaches R <= 0 (min R = {R.min():.6g})")

    def ring(f):  # integral over theta of f dtheta, per section
        return 2 * np.pi * f.mean(axis=1)

    area = ring(R * Zt)
    orientation = np.sign(area)
    if not (np.all(orientation == orientation[0]) and orientation[0] != 0):
        raise Invalid("cross sections change orientation: the boundary folds over itself")
    area = np.abs(area)
    sgn = orientation[0]
    I_R = sgn * ring(R**2 / 2 * Zt)
    I_Z = sgn * ring(R * Z * Zt)
    I_RR = sgn * ring(R**3 / 3 * Zt)
    I_ZZ = sgn * ring(R * Z**2 * Zt)
    I_RZ = sgn * ring(R**2 / 2 * Z * Zt)

    for k in range(GEOM_NPHI):
        if _segments_cross(np.stack([R[k], Z[k]], axis=1)):
            raise Invalid(f"cross section at phi = {phi[k]:.6f} intersects itself")

    mean_area = area.mean()
    volume = 2 * np.pi * I_R.mean()
    minor = math.sqrt(mean_area / math.pi)
    major = volume / (2 * math.pi * mean_area)

    Rc, Zc = I_R / area, I_Z / area
    c_rr = I_RR / area - Rc**2
    c_zz = I_ZZ / area - Zc**2
    c_rz = I_RZ / area - Rc * Zc
    half_trace = (c_rr + c_zz) / 2
    root = np.sqrt(((c_rr - c_zz) / 2) ** 2 + c_rz**2)
    kappa = np.sqrt((half_trace + root) / (half_trace - root))
    return {
        "aspect_ratio": major / minor,
        "minor_radius": minor,
        "major_radius": major,
        "volume": volume,
        "kappa_max": float(kappa.max()),
        "kappa_max_phi": float(phi[int(kappa.argmax())]),
        "min_R": float(R.min()),
    }


# ---------------------------------------------------------------------- vmec

VMEC_NTHETA = VMEC_NZETA = 96
VMEC_MPI_RANKS = 8  # fixed so that the floating-point reduction order is fixed


@dataclass(frozen=True)
class VmecAttempt:
    ns: tuple
    niter: tuple
    ftol: tuple
    delt: float
    precon: str = "none"


def vmec_attempts(ns_final: int) -> list[VmecAttempt]:
    """Fixed retry sequence; the final radial grid always has ns_final surfaces."""
    steps = [(n, it) for n, it in ((17, 1000), (33, 2000), (65, 3000), (129, 4000)) if n < ns_final]
    base = VmecAttempt(tuple(n for n, _ in steps) + (ns_final,), tuple(it for _, it in steps) + (60000,),
                       (1e-12,) * len(steps) + (1e-16,), 0.9)
    slow = VmecAttempt(base.ns, tuple(2 * n for n in base.niter), base.ftol, 0.7)
    fine_ns = tuple(n for n in (17, 33, 65, 97, 129, 193) if n < ns_final) + (ns_final,)
    fine = VmecAttempt(fine_ns, tuple(4 * 1000 * (i + 1) for i in range(len(fine_ns) - 1)) + (240000,),
                       (1e-12,) * (len(fine_ns) - 1) + (1e-16,), 0.5)
    gmres = VmecAttempt(fine.ns, fine.niter, fine.ftol, 0.5, "gmres")
    return [base, slow, fine, gmres]


def vmec_resolution(b: Boundary) -> tuple[int, int]:
    """MPOL - 1 = 2 m_max + 4 and NTOR = 2 n_max + 4, at least (13, 12)."""
    return max(13, 2 * b.m_max + 5), max(12, 2 * b.n_max + 4)


def vmec_input(b: Boundary, attempt: VmecAttempt) -> str:
    mpol, ntor = vmec_resolution(b)
    fmt = lambda xs: " ".join(repr(x) for x in xs)
    lines = [
        "&INDATA",
        "  LFREEB = F, LASYM = F, LRFP = F",
        f"  NFP = {b.nfp}, MPOL = {mpol}, NTOR = {ntor}",
        f"  NTHETA = {VMEC_NTHETA}, NZETA = {VMEC_NZETA}",
        f"  NS_ARRAY = {fmt(attempt.ns)}",
        f"  NITER_ARRAY = {fmt(attempt.niter)}",
        f"  FTOL_ARRAY = {fmt(attempt.ftol)}",
        f"  NSTEP = 250, DELT = {attempt.delt!r}, TCON0 = 2.0",
        f"  PRECON_TYPE = '{attempt.precon}', PREC2D_THRESHOLD = 1e-12",
        # Vacuum: no pressure, zero enclosed toroidal current on every surface.
        "  GAMMA = 0.0, SPRES_PED = 1.0, PRES_SCALE = 0.0",
        "  PMASS_TYPE = 'power_series', AM = 0.0",
        "  NCURR = 1, CURTOR = 0.0, PCURR_TYPE = 'power_series', AC = 0.0",
        f"  PHIEDGE = {AXIS_FIELD * math.pi * MINOR_RADIUS**2!r}",
        # R = 0 makes VMEC build its own axis guess (guess_axis), as SIMSOPT does,
        # so the result does not depend on any earlier run.
        "  RAXIS_CC = 0.0, ZAXIS_CS = 0.0",
    ]
    for (m, n), rc in sorted(b.rbc.items()):
        lines.append(f"  RBC({n},{m}) = {rc!r}, ZBS({n},{m}) = {b.zbs[(m, n)]!r}")
    lines.append("/")
    return "\n".join(lines) + "\n"


def _wout_status(path: Path) -> tuple[int, int, tuple]:
    """ier_flag, ns and the three final force residuals of a wout."""
    from netCDF4 import Dataset

    with Dataset(path) as ds:
        v = ds.variables
        return (int(v["ier_flag"][:]), int(v["ns"][:]),
                (float(v["fsqr"][:]), float(v["fsqz"][:]), float(v["fsql"][:])))


def _wout(path: Path) -> dict:
    from netCDF4 import Dataset

    names = ("ns", "ier_flag", "fsqr", "fsqz", "fsql", "signgs", "iotas", "vp", "gmnc", "bmnc",
             "xm_nyq", "xn_nyq", "xm", "xn", "rmnc", "zmns", "lmns", "aspect", "volume_p",
             "phi", "bsubumnc", "bsubvmnc", "nfp", "mpol", "ntor")
    with Dataset(path) as ds:
        return {k: np.array(ds.variables[k][:]) for k in names}


def _cos_series(coef: np.ndarray, xm: np.ndarray, xn: np.ndarray, theta: np.ndarray,
                zeta: np.ndarray) -> np.ndarray:
    """sum_mn coef[j, mn] cos(m theta - n zeta) for every j on the grid theta x zeta.

    xn already carries the field-period factor, as in a VMEC wout.
    """
    cm, sm = np.cos(np.outer(theta, xm)), np.sin(np.outer(theta, xm))  # (theta, mn)
    cn, sn = np.cos(np.outer(xn, zeta)), np.sin(np.outer(xn, zeta))  # (mn, zeta)
    # cos(m t - n z) = cos(m t) cos(n z) + sin(m t) sin(n z)
    return (np.einsum("tk,jk,kz->jtz", cm, coef, cn, optimize=True)
            + np.einsum("tk,jk,kz->jtz", sm, coef, sn, optimize=True))


def _period_grid(n: int, nfp: int, offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    theta = 2 * np.pi * (np.arange(n) + offset) / n
    zeta = (2 * np.pi / nfp) * (np.arange(n) + offset) / n
    return theta, zeta


VMEC_NS_FINAL = (257, 201, 101)  # largest radial grid that solves is used; 101 is the floor
VMEC_BUDGET = 4 * 3600.0  # s, all attempts together (--vmec-budget-hours overrides)
EVAL_BUDGET = 12 * 3600.0  # s, the whole evaluation; the verifier timeout leaves half an hour above it


def run_vmec(b: Boundary, workdir: Path, program: str, budget: float = VMEC_BUDGET) -> dict:
    """Solve the vacuum equilibrium on the largest radial grid that converges.

    For each final grid (257, then 201, then 101 surfaces) the fixed retry
    sequence is tried in order; the first converged attempt is used.
    Converged means VMEC's own success flag, all three final force residuals
    below the final FTOL, the requested final radial grid, and one sign of the
    Jacobian on every half-grid surface.  All attempts share one time budget.
    """
    import subprocess

    failures = []
    deadline = time.time() + budget
    number = 0
    for ns_final in VMEC_NS_FINAL:
        for attempt in vmec_attempts(ns_final):
            number += 1
            remaining = deadline - time.time()
            if remaining <= 0:
                failures.append(f"time budget of {budget:.0f} s used up")
                raise Invalid("vacuum equilibrium not solved: " + "; ".join(failures))
            run = workdir / f"vmec_attempt{number}"
            run.mkdir(parents=True, exist_ok=True)
            (run / "input.eq").write_text(vmec_input(b, attempt))
            started = time.time()
            with open(run / "log.txt", "w") as log:
                try:
                    # No core binding: each mpirun would otherwise pin its ranks to cores 0..15,
                    # and concurrent runs would share them.
                    subprocess.run(["mpirun", "--allow-run-as-root", "--bind-to", "none", "-np", str(VMEC_MPI_RANKS),
                                    program, "input.eq"],
                                   cwd=run, stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                except subprocess.TimeoutExpired:
                    failures.append(f"attempt {number} (ns={ns_final}): stopped by the time budget")
                    continue
            seconds = time.time() - started
            wout = run / "wout_eq.nc"
            if not wout.exists():
                failures.append(f"attempt {number} (ns={ns_final}): no wout written ({seconds:.0f} s)")
                continue
            # A failed run still writes a wout, holding only the scalars: check them first.
            ier_flag, ns, residuals = _wout_status(wout)
            if ier_flag != 0 or ns != ns_final or max(residuals) > attempt.ftol[-1]:
                failures.append(f"attempt {number} (ns={ns_final}): ier_flag={ier_flag}, "
                                f"ns={ns}, residuals={residuals} ({seconds:.0f} s)")
                continue
            w = _wout(wout)
            jac = _cos_series(w["gmnc"][1:], w["xm_nyq"], w["xn_nyq"], *_period_grid(2 * VMEC_NTHETA, b.nfp))
            if not np.all(np.sign(jac) == np.sign(w["signgs"])):
                failures.append(f"attempt {number} (ns={ns_final}): Jacobian changes sign ({seconds:.0f} s)")
                continue
            return {"wout": wout, "wout_data": w, "attempt": number, "ns": ns_final, "seconds": seconds,
                    "residuals": residuals, "resolution": vmec_resolution(b), "failed_attempts": failures}
    raise Invalid("vacuum equilibrium not solved: " + "; ".join(failures))


def profile_metrics(w: dict) -> dict:
    """iota_min and the local vacuum well W, from the wout alone.

    V' = dV/ds = 4 pi^2 |sqrt(g)_00| on the half grid.  Its slope between neighbouring
    half-grid surfaces sits on the full-grid surface between them; W is the smallest
    -(dV'/ds)/V'(0) over the full-grid surfaces with 0.05 <= s <= 0.95, so W >= 0 means
    a well on every surface there.  V'(0) is extrapolated from the two innermost
    half-grid values.
    """
    ns = int(w["ns"])
    s_half = (np.arange(1, ns) - 0.5) / (ns - 1)
    iota = np.abs(w["iotas"][1:])
    band = (s_half >= 0.05) & (s_half <= 0.95)
    g00 = w["gmnc"][1:, 0]
    if w["xm_nyq"][0] != 0 or w["xn_nyq"][0] != 0:
        raise Invalid("wout gmnc does not start with the (0,0) mode")
    dvds = 4 * np.pi**2 * np.abs(g00)
    vp = 4 * np.pi**2 * np.asarray(w["vp"][1:])
    if not np.allclose(dvds, np.abs(vp), rtol=1e-12, atol=0):
        raise Invalid("wout vp and gmnc(0,0) disagree")
    v0 = 1.5 * dvds[0] - 0.5 * dvds[1]
    s_full = np.arange(1, ns - 1) / (ns - 1)
    slope = np.diff(dvds) * (ns - 1)
    in_band = (s_full >= 0.05) & (s_full <= 0.95)
    local = -slope[in_band] / v0
    return {
        "iota_min": float(iota[band].min()),
        "iota_min_s": float(s_half[band][iota[band].argmin()]),
        "iota_axis_side": float(iota[0]),
        "iota_edge_side": float(iota[-1]),
        "well": float(local.min()),
        "well_worst_s": float(s_full[in_band][local.argmin()]),
        "vmec_aspect": float(w["aspect"]),
        "volume_from_vp_rel_error": float(abs(dvds.sum() / (ns - 1) / float(w["volume_p"]) - 1)),
    }


# ------------------------------------------------------------ boozer and neo

BOOZ_TOL = 1e-12  # stop doubling once e_B reaches this
BOOZ_VALID = 1e-8  # e_B above this at the cap: no valid eps_eff
BOOZ_CAP = (256, 192)
NEO_TARGETS = (0.2, 0.4, 0.6, 0.8)
NEO_TIMEOUT = 3600.0  # s; NEO takes seconds to minutes, hours only when |iota| per period is below ~5e-4


def neo_surfaces(ns: int) -> list[int]:
    """0-based half-grid indices j (s_j = (j + 0.5)/(ns - 1)) nearest 0.2 ... 0.8, plus the outermost.

    A tie goes to the smaller s.  Exact rational arithmetic, so the choice
    never depends on rounding.
    """
    from fractions import Fraction

    chosen = []
    for target in NEO_TARGETS:
        x = Fraction(str(target)) * (ns - 1) - Fraction(1, 2)
        lo = math.floor(x)
        chosen.append(lo if x - lo <= lo + 1 - x else lo + 1)
    return chosen + [ns - 2]


def boozer_error(b, jb: int, js: int, nfp: int, ntheta: int) -> float:
    """max |B_Boozer - B_VMEC| over the planes phi = 0, pi/nfp, both at the same physical points.

    VMEC points (theta, phi) are mapped to Boozer angles with booz_xform's own
    formulas (surface_solve.cpp): nu = (w - I lambda)/(G + iota I),
    theta_B = theta + lambda + iota nu, zeta_B = phi + nu, where w is built
    from the covariant components bsubu, bsubv.
    """
    xm, xn = np.asarray(b.xm), np.asarray(b.xn)
    xmn, xnn = np.asarray(b.xm_nyq), np.asarray(b.xn_nyq)
    lmns, bmnc = np.asarray(b.lmns)[:, js], np.asarray(b.bmnc)[:, js]
    bu, bv = np.asarray(b.bsubumnc)[:, js], np.asarray(b.bsubvmnc)[:, js]
    i00 = int(np.flatnonzero((xmn == 0) & (xnn == 0))[0])
    I, G, iota = bu[i00], bv[i00], float(np.asarray(b.iota)[js])
    w = np.zeros_like(bu)
    k = xmn != 0
    w[k] = bu[k] / xmn[k]
    k = (xmn == 0) & (xnn != 0)
    w[k] = -bv[k] / xnn[k]
    xmb, xnb, bb = np.asarray(b.xm_b), np.asarray(b.xn_b), np.asarray(b.bmnc_b)[:, jb]
    theta = 2 * np.pi * np.arange(ntheta) / ntheta
    worst = 0.0
    for phi in (0.0, np.pi / nfp):
        ph = np.full_like(theta, phi)
        lam = np.sin(np.outer(theta, xm) - np.outer(ph, xn)) @ lmns
        an = np.outer(theta, xmn) - np.outer(ph, xnn)
        nu = (np.sin(an) @ w - I * lam) / (G + iota * I)
        b_vmec = np.cos(an) @ bmnc
        b_booz = np.cos(np.outer(theta + lam + iota * nu, xmb) - np.outer(ph + nu, xnb)) @ bb
        worst = max(worst, float(np.max(np.abs(b_booz - b_vmec))))
    return worst


def run_boozer(wout: Path, nfp: int, workdir: Path) -> dict:
    """Boozer transform of the surfaces NEO needs and of the two innermost ones.

    Truncation: from mboz = 2 MPOL, nboz = 2 NTOR, doubling until e_B <= 1e-12
    or the cap (256, 192); e_B is normalised by the axis field B00(0).
    B00(0) is the Boozer (0,0) mode extrapolated linearly from the two
    innermost half-grid surfaces.
    """
    import booz_xform as bx

    b = bx.Booz_xform()
    b.verbose = 0
    b.read_wout(str(wout))  # resets compute_surfs, mboz, nboz: set them afterwards
    ns = b.ns_in + 1
    neo = neo_surfaces(ns)
    surfs = sorted(set([0, 1] + neo))  # ascending: LIBSTELL unpacks jlist in order
    b.compute_surfs = surfs
    mboz, nboz = 2 * b.mpol, 2 * b.ntor
    history = []
    while True:
        b.mboz, b.nboz = mboz, nboz
        started = time.time()
        b.run()
        i00 = int(np.flatnonzero((np.asarray(b.xm_b) == 0) & (np.asarray(b.xn_b) == 0))[0])
        b00 = np.asarray(b.bmnc_b)[i00]
        s = (np.asarray(surfs) + 0.5) / (ns - 1)
        b00_axis = float(b00[0] - s[0] * (b00[1] - b00[0]) / (s[1] - s[0]))
        e_b = max(boozer_error(b, jb, js, nfp, 4 * mboz) for jb, js in enumerate(surfs)) / b00_axis
        history.append({"mboz": mboz, "nboz": nboz, "e_B": e_b, "seconds": time.time() - started})
        if e_b <= BOOZ_TOL or (mboz, nboz) == BOOZ_CAP:
            break
        mboz, nboz = min(2 * mboz, BOOZ_CAP[0]), min(2 * nboz, BOOZ_CAP[1])
    if e_b > BOOZ_VALID:
        raise Invalid(f"Boozer transform not resolved: e_B = {e_b:.3g} at the cap {BOOZ_CAP}")
    path = workdir / "boozmn_eq.nc"
    b.write_boozmn(str(path))
    return {"boozmn": path, "B00_axis": b00_axis, "e_B": e_b, "history": history,
            "neo_js": [j + 2 for j in neo], "s_neo": [(j + 0.5) / (ns - 1) for j in neo]}


def neo_control(js: list[int]) -> str:
    """NEO's line-by-line control file (read in order by neo_read_control.f90)."""
    values = [
        "boozmn_eq.nc", "neo_out.eq", str(len(js)), " ".join(map(str, js)),
        "300", "300",  # theta_n, phi_n
        "0", "0",  # max_m_mode, max_n_mode: all modes
        "150", "1", "0.005", "100",  # npart, multra, acc_req, no_bins
        "75", "500", "2000", "0",  # nstep_per, nstep_min, nstep_max, calc_nstep_max
        "1", "0", "0",  # eout_swi (column 2 = EPSTOT), lab_swi, inp_swi (booz_xform netCDF)
        "2",  # ref_swi: B_ref = max B on each surface; R_ref = R00 of the first listed surface
        "0", "0", "0", "0", "0",  # write_progress, write_output_files, spline_test, write_integrate, write_diagnostic
    ]
    tail = ["0", "neo_cur.eq", "200", "2", "0"]  # calc_cur off
    return "\n".join(["'#'"] * 3 + values + ["'#'"] * 3 + tail) + "\n"


def run_neo(boozer: dict, workdir: Path, program: str, timeout: float) -> dict:
    """eps_eff = EPSTOT^(2/3) on the five surfaces, computed in one NEO run in ascending s."""
    import subprocess

    run = workdir / "neo"
    run.mkdir(parents=True, exist_ok=True)
    target = run / "boozmn_eq.nc"
    target.symlink_to(boozer["boozmn"].resolve())
    (run / "neo_in.eq").write_text(neo_control(boozer["neo_js"]))
    started = time.time()
    with open(run / "log.txt", "w") as log:
        try:
            done = subprocess.run([program, "eq"], cwd=run, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise Invalid(f"NEO did not finish within {timeout:.0f} s")
    if done.returncode != 0 or not (run / "neo_out.eq").exists():
        raise Invalid(f"NEO failed with code {done.returncode} (see neo/log.txt)")
    table = np.atleast_2d(np.loadtxt(run / "neo_out.eq"))
    if table.shape[0] != len(boozer["neo_js"]) or list(table[:, 0].astype(int)) != boozer["neo_js"]:
        raise Invalid("NEO output does not list the requested surfaces")
    epstot = table[:, 1]
    if not np.all(np.isfinite(epstot)) or np.any(epstot < 0):
        raise Invalid(f"NEO returned invalid EPSTOT {epstot.tolist()}")
    eps = epstot ** (2.0 / 3.0)
    return {"s": boozer["s_neo"], "eps_eff": eps.tolist(), "eps_eff_max": float(eps.max()),
            "B_ref": table[:, 4].tolist(), "R_ref": table[:, 5].tolist(), "seconds": time.time() - started}


# -------------------------------------------------------------------- alphas

N_ALPHA = 4096
ALPHA_SEEDS = (2026091301, 2026091302)  # two groups of 2048
S_LAUNCH = 0.25
SAMPLE_GRID = 512
ALPHA_TIME = 0.2  # s
ALPHA_MASS = 3.97259969  # m_alpha / m_p (CODATA 2018); SIMPLE's n_d is in proton masses


def _inverse_cdf(weights: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Continuous inverse of a piecewise-constant density on unit cells; returns cell units."""
    cdf = np.concatenate([[0.0], np.cumsum(weights)])
    cdf /= cdf[-1]
    k = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, len(weights) - 1)
    return k + (u - cdf[k]) / (cdf[k + 1] - cdf[k])


def alpha_starts(w: dict, nfp: int) -> np.ndarray:
    """4096 start points, uniform in volume in a thin shell at s = 0.25.

    In VMEC angles the volume element is |sqrt(g)| ds dtheta dzeta, so the
    angular density is |sqrt(g)(0.25, theta, zeta)| (the same distribution as
    1/B^2 in Boozer angles for a current-free field).  sqrt(g) at s = 0.25 is
    interpolated linearly between the two neighbouring half-grid surfaces.
    Angles come from a 512 x 512 cell grid over one field period: zeta from
    its marginal, theta from its conditional, each by a continuous inverse
    CDF; then xi = v_par/v uniform in [-1, 1].  Each group draws a 2048 x 3
    array of uniforms (zeta, theta, xi) from PCG64DXSM with a fixed seed.
    Columns: s, theta_VMEC, zeta (geometric), v/v0, v_par/v, as SIMPLE's
    start.dat expects.
    """
    ns = int(w["ns"])
    s_half = (np.arange(1, ns) - 0.5) / (ns - 1)
    hi = int(np.searchsorted(s_half, S_LAUNCH))
    lo = hi - 1
    t = (S_LAUNCH - s_half[lo]) / (s_half[hi] - s_half[lo])
    g = (1 - t) * w["gmnc"][1 + lo] + t * w["gmnc"][1 + hi]
    theta_c, zeta_c = _period_grid(SAMPLE_GRID, nfp, offset=0.5)  # cell centres
    dth, dze = theta_c[1] - theta_c[0], zeta_c[1] - zeta_c[0]
    jac = np.abs(_cos_series(g[None, :], w["xm_nyq"], w["xn_nyq"], theta_c, zeta_c)[0]).T  # (zeta, theta)
    rows = []
    for seed in ALPHA_SEEDS:
        U = np.random.Generator(np.random.PCG64DXSM(seed)).random((N_ALPHA // len(ALPHA_SEEDS), 3))
        z = _inverse_cdf(jac.sum(axis=1), U[:, 0])
        cell = np.minimum(z.astype(int), SAMPLE_GRID - 1)
        th = np.array([_inverse_cdf(jac[c], np.array([u]))[0] for c, u in zip(cell, U[:, 1])])
        rows.append(np.column_stack([np.full(len(U), S_LAUNCH), th * dth, z * dze, np.ones(len(U)),
                                     2 * U[:, 2] - 1]))
    return np.vstack(rows)


SIMPLE_EXIT = {0: "confined", 1: "lost", 2: "lost"}  # every other code is unresolved
# SIMPLE (libneo) splines R, Z and lambda on ns x n_theta x n_phi points, n_theta = 5 m_V + 1 and
# n_phi = 5 n_V + 1 (multharm = 5; m_V, n_V the largest poloidal and toroidal mode numbers of the
# wout, n_V including N_FP).  Peak memory measured on Landreman-Paul QA+well (257 x 71 x 141):
# 24.95 GiB, 9.7e-6 GiB per point; LP QH+well (257 x 71 x 281) twice that.
SIMPLE_GIB_PER_POINT = 1.0e-5
SIMPLE_MEMORY_LIMIT_GIB = 110.0  # of the 128 GiB of the evaluation


def simple_memory_gib(w: dict) -> float:
    n_theta = 5 * int(np.max(w["xm"])) + 1
    n_phi = 5 * int(np.max(np.abs(w["xn"]))) + 1
    return SIMPLE_GIB_PER_POINT * int(w["ns"]) * n_theta * n_phi


def simple_input(field_scale: float) -> str:
    return f"""&config
netcdffile = 'wout.nc'
isw_field_type = 2
integmode = 1
npoiper2 = 256
relerr = 1d-13
multharm = 5
ns_s = 5
ns_tp = 5
startmode = 2
ntestpart = {N_ALPHA}
sbeg = {S_LAUNCH}d0
trace_time = {ALPHA_TIME}d0
ntimstep = 10001
n_e = 2
n_d = {ALPHA_MASS}d0
facE_al = 1d0
vmec_B_scale = {field_scale!r}
vmec_RZ_scale = 1d0
contr_pp = -1d10
tcut = -1d0
class_plot = .False.
fast_class = .False.
swcoll = .False.
deterministic = .True.
/
"""


def run_simple(wout: Path, starts: np.ndarray, field_scale: float, workdir: Path, program: str,
               threads: int, timeout: float) -> dict:
    """Trace the alphas; f_alpha = (lost + unresolved) / 4096."""
    import os
    import re
    import subprocess

    run = workdir / "simple"
    run.mkdir(parents=True, exist_ok=True)
    target = run / "wout.nc"
    target.symlink_to(wout.resolve())
    (run / "simple.in").write_text(simple_input(field_scale))
    np.savetxt(run / "start.dat", starts, fmt="%.17e")
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), GFORTRAN_UNBUFFERED_PRECONNECTED="y")
    started = time.time()
    with open(run / "log.txt", "w") as log:
        try:
            done = subprocess.run([program, "simple.in"], cwd=run, stdout=log, stderr=subprocess.STDOUT, env=env,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            raise Invalid(f"SIMPLE did not finish within the evaluation time limit ({timeout:.0f} s left for it)")
    seconds = time.time() - started
    if done.returncode < 0:
        raise EvaluatorError(f"SIMPLE was killed by signal {-done.returncode} (see simple/log.txt)")
    if done.returncode != 0:
        raise Invalid(f"SIMPLE exited with code {done.returncode} (see simple/log.txt)")
    table = np.loadtxt(run / "orbit_exit_code.dat", ndmin=2)
    if table.shape[0] != N_ALPHA or not np.array_equal(table[:, 0].astype(int), np.arange(1, N_ALPHA + 1)):
        raise Invalid(f"SIMPLE reported {table.shape[0]} orbits instead of {N_ALPHA}")
    codes = table[:, 1].astype(int)
    lost = np.isin(codes, (1, 2)) & (table[:, 2] <= ALPHA_TIME)
    confined = codes == 0
    unresolved = ~(lost | confined)
    log_text = (run / "log.txt").read_text()
    b00 = re.findall(r"B_00\s*=\s*([-+0-9.EeDd]+)", log_text)
    # SIMPLE recovers some integration steps silently; its last progress line counts them.
    progress = [line for line in log_text.splitlines() if line.lstrip().startswith("[progress]")]
    recoveries = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)(?=\s|$)", progress[-1])} if progress else {}
    half = N_ALPHA // len(ALPHA_SEEDS)
    return {
        "N_L": int(lost.sum()),
        "N_U": int(unresolved.sum()),
        "f_alpha": float((lost.sum() + unresolved.sum()) / N_ALPHA),
        "f_alpha_groups": [float((lost | unresolved)[i * half:(i + 1) * half].mean())
                           for i in range(len(ALPHA_SEEDS))],
        "exit_codes": {str(c): int((codes == c).sum()) for c in np.unique(codes)},
        "simple_axis_field_gauss": float(b00[-1].replace("D", "E")) if b00 else None,
        "simple_recoveries": recoveries,
        "seconds": seconds,
    }


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


# ---------------------------------------------------------------- coil room

LGRADB_GRID = 128  # theta x phi points over one field period; 256 x 256 changes the minimum by < 0.2%


def coil_room(wout: Path, nfp: int) -> dict:
    """Smallest magnetic gradient scale length L_grad(B) = sqrt(2) |B| / ||grad B||_F on the boundary.

    SIMSOPT's vmec_compute_geometry at s = 1, the quantity Kappel, Landreman and
    Malhotra (PPCF 66, 025018, 2024) correlate with the achievable coil distance.
    """
    from simsopt.mhd import Vmec
    from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry

    theta = 2 * np.pi * np.arange(LGRADB_GRID) / LGRADB_GRID
    phi = (2 * np.pi / nfp) * np.arange(LGRADB_GRID) / LGRADB_GRID
    lgradb = np.asarray(vmec_compute_geometry(Vmec(str(wout)), np.array([1.0]), theta, phi).L_grad_B)[0]
    i, j = np.unravel_index(int(np.argmin(lgradb)), lgradb.shape)
    return {"lgradb_min": float(lgradb[i, j]), "theta": float(theta[i]), "phi": float(phi[j])}


# --------------------------------------------------------------------- score


def completion(metrics: dict) -> dict:
    """Constraint completions q of the problem statement."""

    def upper(limit, value):
        return 1.0 if value == 0 else min(1.0, limit / value)

    well = metrics["well"]
    return {
        "f_alpha": upper(F_ALPHA_MAX, metrics["f_alpha"]),
        "eps_eff": upper(EPS_EFF_MAX, metrics["eps_eff_max"]),
        "iota": min(1.0, metrics["iota_min"] / IOTA_MIN),
        "well": 1.0 if well >= WELL_MIN else 1.0 / (1.0 + (-well) / WELL_SCALE),
        "kappa": upper(KAPPA_MAX, metrics["kappa_max"]),
        "coil": min(1.0, metrics["lgradb_min"] / LGRADB_MIN),
    }


def score(metrics: dict) -> tuple[float, bool, dict]:
    q = completion(metrics)
    feasible = all(v == 1.0 for v in q.values())
    if feasible:
        return 0.5 + 0.5 / (1.0 + math.exp(metrics["aspect_ratio"] - A_MID)), True, q
    return 0.5 * math.prod(q.values()), False, q


# ---------------------------------------------------------------------- main

PROGRAMS = ("xvmec2000", "xneo", "simple.x")


class EvaluatorError(Exception):
    """The evaluator itself is broken (missing program, unexpected output); not the submission's fault."""


def evaluate(submission: Path, workdir: Path, threads: int, details: dict, deadline: float,
             vmec_budget: float = VMEC_BUDGET) -> None:
    """Run every stage in order, filling details; the first Invalid stops the evaluation.

    deadline is the wall-clock time by which the whole evaluation must end.
    """
    import shutil

    programs = {name: shutil.which(name) for name in PROGRAMS}
    missing = [name for name, path in programs.items() if path is None]
    if missing:
        raise EvaluatorError(f"programs not on PATH: {', '.join(missing)}")
    stages = details.setdefault("stages", {})

    b = read_boundary(submission)
    submitted = geometry(b)
    b = b.scaled(MINOR_RADIUS / submitted["minor_radius"])
    stages["geometry"] = geometry(b)
    stages["geometry"]["scale_factor"] = MINOR_RADIUS / submitted["minor_radius"]

    vmec = run_vmec(b, workdir, programs["xvmec2000"], min(vmec_budget, deadline - time.time()))
    w = vmec["wout_data"]
    stages["vmec"] = {k: vmec[k] for k in ("attempt", "ns", "seconds", "residuals", "resolution", "failed_attempts")}
    stages["profile"] = profile_metrics(w)
    stages["coil"] = coil_room(vmec["wout"], b.nfp)

    boozer = run_boozer(vmec["wout"], b.nfp, workdir)
    stages["boozer"] = {k: boozer[k] for k in ("B00_axis", "e_B", "history", "s_neo")}
    stages["neo"] = run_neo(boozer, workdir, programs["xneo"], min(NEO_TIMEOUT, deadline - time.time()))

    memory = simple_memory_gib(w)
    stages["simple_memory_gib"] = memory
    if memory > SIMPLE_MEMORY_LIMIT_GIB:
        raise Invalid(f"SIMPLE would need about {memory:.0f} GiB for its field splines, "
                      f"above the {SIMPLE_MEMORY_LIMIT_GIB:.0f} GiB limit")
    field_scale = AXIS_FIELD / boozer["B00_axis"]
    alpha = run_simple(vmec["wout"], alpha_starts(w, b.nfp), field_scale, workdir, programs["simple.x"], threads,
                       deadline - time.time())
    alpha["field_scale"] = field_scale
    alpha["wilson_95"] = [wilson(alpha["N_L"], N_ALPHA)[0], wilson(alpha["N_L"] + alpha["N_U"], N_ALPHA)[1]]
    stages["alpha"] = alpha
    if alpha["N_U"] / N_ALPHA > 0.001:
        raise Invalid(f"{alpha['N_U']} of {N_ALPHA} alpha orbits unresolved (limit 0.1%)")

    metrics = {
        "aspect_ratio": stages["geometry"]["aspect_ratio"],
        "f_alpha": alpha["f_alpha"],
        "eps_eff_max": stages["neo"]["eps_eff_max"],
        "iota_min": stages["profile"]["iota_min"],
        "well": stages["profile"]["well"],
        "kappa_max": stages["geometry"]["kappa_max"],
        "lgradb_min": stages["coil"]["lgradb_min"],
    }
    value, feasible, q = score(metrics)
    details.update(metrics=metrics, completion=q, score=value, feasible=feasible)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("submission", type=Path)
    parser.add_argument("--workdir", type=Path, required=True, help="scratch directory for program runs")
    parser.add_argument("--reward-dir", type=Path, default=Path("/logs/verifier"))
    parser.add_argument("--threads", type=int, default=32, help="OpenMP threads for SIMPLE")
    parser.add_argument("--vmec-budget-hours", type=float, default=VMEC_BUDGET / 3600,
                        help="wall-clock limit shared by all VMEC attempts")
    args = parser.parse_args()
    if args.workdir.exists() and any(args.workdir.iterdir()):
        parser.error(f"--workdir {args.workdir} must be empty: earlier program output would be read as this run's")
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.reward_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    details = {}
    try:
        evaluate(args.submission, args.workdir, args.threads, details, started + EVAL_BUDGET,
                 args.vmec_budget_hours * 3600)
        reward = {"reward": details["score"], "valid": 1, "feasible": int(details["feasible"]), "evaluator_error": 0}
    except Invalid as exc:
        details["invalid"] = str(exc)
        reward = {"reward": 0.0, "valid": 0, "feasible": 0, "evaluator_error": 0}
    except Exception as exc:  # the evaluator failed; recorded apart from invalid submissions
        details["evaluator_error"] = f"{type(exc).__name__}: {exc}"
        details["traceback"] = traceback.format_exc()
        reward = {"reward": 0.0, "valid": 0, "feasible": 0, "evaluator_error": 1}
    details["seconds"] = time.time() - started
    (args.reward_dir / "details.json").write_text(json.dumps(details, indent=1, default=str))
    (args.reward_dir / "reward.json").write_text(json.dumps(reward))
    print(json.dumps(reward))
    return 0


if __name__ == "__main__":
    sys.exit(main())
