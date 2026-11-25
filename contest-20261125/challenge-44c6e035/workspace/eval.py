#!/usr/bin/env python3
"""C0: native SIMSOPT fields; FusionX only owns serialization, diagnostics and score.

--measure computes real physical quantities before release calibration. It never
produces a competition score. Formal mode requires the calibrated matching bank.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class InvalidSubmission(Exception):
    pass


class Unresolved(Exception):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidSubmission(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_submission(path, case):
    p = Path(path)
    if not p.is_file() or p.stat().st_size > 1024 * 1024:
        raise InvalidSubmission("coils.json missing or larger than 1 MiB")
    try:
        data = json.loads(p.read_text(), object_pairs_hook=no_duplicates,
                          parse_constant=lambda x: (_ for _ in ()).throw(InvalidSubmission(x)))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise InvalidSubmission(f"cannot read submission JSON: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"nfp", "order", "coils"}:
        raise InvalidSubmission("top-level keys must be nfp, order, coils")
    if type(data["nfp"]) is not int or data["nfp"] != 2:
        raise InvalidSubmission("nfp must be integer 2")
    if type(data["order"]) is not int or data["order"] != case["order"]:
        raise InvalidSubmission("order must equal the specified Fourier order")
    if not isinstance(data["coils"], list) or len(data["coils"]) != 4:
        raise InvalidSubmission("exactly four base coils are required")
    for i, coil in enumerate(data["coils"]):
        if not isinstance(coil, dict) or set(coil) != {"x", "y", "z", "current", "rotation"}:
            raise InvalidSubmission(f"coil {i}: keys must be x, y, z, current, rotation")
        rotation = coil["rotation"]
        if not isinstance(rotation, list) or len(rotation) != 5 or any(
                type(x) not in (int, float) or abs(x) > math.pi or not math.isfinite(x)
                for x in rotation):
            raise InvalidSubmission(f"coil {i}: rotation requires five finite radians, abs<=pi")
        for axis in "xyz":
            values = coil[axis]
            if not isinstance(values, list) or len(values) != 2 * case["order"] + 1:
                raise InvalidSubmission(f"coil {i}: {axis} has incorrect length")
            if any(type(x) not in (int, float) or abs(x) > 100 or not math.isfinite(x)
                   for x in values):
                raise InvalidSubmission(f"coil {i}: coefficients must be finite metres, abs<=100")
        value = coil["current"]
        if type(value) not in (int, float) or abs(value) > 1e9 or not math.isfinite(value):
            raise InvalidSubmission(f"coil {i}: current must be finite ampere-turns, abs<=1e9")
    return data


def reconstruct(data, count):
    from simsopt.geo import CurveXYZFourier
    from simsopt.field import Coil, Current, coils_via_symmetries
    base = []
    currents = []
    for entry in data["coils"]:
        curve = CurveXYZFourier(count, data["order"])
        curve.local_full_x = np.asarray([*entry["x"], *entry["y"], *entry["z"]])
        base.append(curve)
        currents.append(Current(entry["current"]))
    # Expand each pack independently to keep its material-frame variables attached
    # to the same physical coil under symmetry and canonical sorting.
    coils = []
    for curve, current, entry in zip(base, currents, data["coils"]):
        for coil in coils_via_symmetries([curve], [current], 2, True):
            coil.pack_rotation = np.asarray(entry["rotation"], dtype=float)
            coils.append(coil)
    # Error slots identify physical coils, never their submitted array position.
    # Use one fixed native quadrature for this identity at every field resolution.
    def physical_key(coil):
        exact = materialize(coil, data["order"], 16384)
        center = np.asarray(exact.centroid(), dtype=float)
        if not np.isfinite(center).all():
            raise InvalidSubmission("coil has no finite native arc-length centroid")
        angle = float(np.arctan2(center[1], center[0]) % (2 * np.pi))
        return (angle, *center, *exact.local_full_x, float(coil.current.get_value()), *coil.pack_rotation)
    coils.sort(key=physical_key)
    # Rebuild in canonical order so the native dependency graph / accumulation
    # order cannot retain the submitted base-coil ordering either.
    explicit = []
    for c in coils:
        item = Coil(materialize(c, data["order"], count), Current(c.current.get_value()))
        item.pack_rotation = c.pack_rotation
        explicit.append(item)
    coils = explicit
    return base, currents, coils


def pack_filaments(curve, rotation, case, count):
    """Native filaments at Gauss nodes of a FIXED rectangular cross-section."""
    from numpy.polynomial.legendre import leggauss
    from simsopt.geo import FrameRotation, FramedCurveCentroid, CurveFilament
    pack = case["winding_pack"]
    nodes, weights = leggauss(count)
    angle = FrameRotation(curve.quadpoints, 2, scale=1.)
    angle.local_full_x = np.asarray(rotation)
    frame = FramedCurveCentroid(curve, angle)
    filaments = []
    for i, x in enumerate(nodes):
        for j, y in enumerate(nodes):
            filament = CurveFilament(frame, pack["width_m"]*x/2, pack["height_m"]*y/2)
            filament.quadrature_weight = float(weights[i]*weights[j]/4)
            filaments.append(filament)
    return filaments


def pack_geometry(data, case, values, details):
    """Use native material-frame twist; enclose the entire pack for clearance."""
    pack = case["winding_pack"]
    radius = math.hypot(pack["width_m"], pack["height_m"]) / 2
    values.update(
        pack_coil_distance_lower_m=values["coil_distance_lower_m"] - 2 * radius,
        pack_plasma_distance_lower_m=values["plasma_distance_lower_m"] - radius,
        pack_self_distance_lower_m=values["self_distance_lower_m"] - 2 * radius,
        pack_radius_min_lower_m=values["radius_min_lower_m"] - radius,
        pack_radius_max_upper_m=values["radius_max_upper_m"] + radius,
        pack_z_max_upper_m=values["z_max_upper_m"] + radius,
        pack_current_density_MA_per_m2=values["current_max_Aturn"] / (1e6 * pack["width_m"] * pack["height_m"]),
        pack_local_curvature_product=radius * values["kappa_upper_per_m"])
    twists = []
    frame_margins = []
    for count in pack["frame_grids"]:
        base, _, _ = reconstruct(data, count)
        maxima = []
        for curve, entry in zip(base, data["coils"]):
            vmax, amax, _ = curve_derivative_bounds(curve, data["order"])
            derivative = curve.gammadash()
            speed = np.linalg.norm(derivative, axis=1)
            vmin = float(speed.min()) - amax / (2 * count)
            if vmin <= 1e-5:
                raise InvalidSubmission("material frame has an unresolved tangent")
            tangent = derivative / speed[:, None]
            radial = curve.gamma() - curve.gamma().mean(axis=0)
            transverse = radial - np.sum(radial * tangent, axis=1)[:, None] * tangent
            rmax = np.linalg.norm(radial, axis=1).max() + vmax / (2 * count)
            # Fourier sampling mean equals c0 here. Bound the projected radial
            # vector's derivative to exclude a centroid-frame singularity.
            margin = np.linalg.norm(transverse, axis=1).min() - (2*vmax + 2*rmax*amax/vmin)/(2*count)
            if not math.isfinite(margin) or margin <= 1e-5:
                raise InvalidSubmission("centroid material frame is singular or unresolved")
            frame_margins.append(float(margin))
            framed = pack_filaments(curve, entry["rotation"], case, 1)[0].framedcurve
            twist = np.asarray(framed.frame_torsion())
            if not np.isfinite(twist).all():
                raise InvalidSubmission("native material-frame twist is nonfinite")
            maxima.append(float(np.max(abs(twist))))
        twists.append(np.asarray(maxima))
    delta = abs(twists[-1] - twists[-2])
    if delta.max() > pack["twist_resolution_per_m"]:
        raise Unresolved("material-frame twist has not resolved")
    values["pack_twist_upper_per_m"] = float(np.max(twists[-1] + delta))
    details["winding_pack"] = {"circumscribed_radius_m": radius,
        "frame_nonsingularity_lower_m": min(frame_margins),
        "native_twist_maxima_per_m": [v.tolist() for v in twists],
        "twist_grid_difference_per_m": delta.tolist(),
        "scope": "continuous enclosing-tube clearances; sampled native twist with refinement estimate"}


def materialize(coil, order, count):
    """The native symmetry curve has no local DOFs; retain its exact Fourier map."""
    from simsopt.geo import CurveXYZFourier
    curve = coil.curve
    matrix = np.eye(3)
    while hasattr(curve, "rotmat"):
        matrix = curve.rotmat @ matrix
        curve = curve.curve
    coefficients = np.asarray(curve.local_full_x).reshape(3, 2 * order + 1)
    transformed = matrix.T @ coefficients
    out = CurveXYZFourier(count, order)
    out.local_full_x = transformed.ravel()
    return out


def rigid_transform(curve, translation, rotvec, center, order, count):
    """Exact affine transformation of native Fourier coefficients, no field kernel."""
    from simsopt.geo import CurveXYZFourier
    matrix = Rotation.from_rotvec(rotvec).as_matrix()
    coeff = matrix @ np.asarray(curve.local_full_x).reshape(3, 2 * order + 1)
    coeff[:, 0] += np.asarray(center) + np.asarray(translation) - matrix @ center
    out = CurveXYZFourier(count, order)
    out.local_full_x = coeff.ravel()
    return out


def surface(case, root, nphi, ntheta, shifted=False):
    from simsopt.geo import SurfaceRZFourier
    grid = {"range": "full torus", "nphi": nphi, "ntheta": ntheta}
    if shifted:
        grid = {"quadpoints_phi": (np.arange(nphi) + .5) / nphi,
                "quadpoints_theta": (np.arange(ntheta) + .5) / ntheta}
    result = SurfaceRZFourier.from_vmec_input(root / "reference/target.vmec",
                                             **grid)
    result.local_full_x *= case["length_scale"]
    return result


def curve_derivative_bounds(curve, order):
    coeff = np.asarray(curve.local_full_x).reshape(3, 2 * order + 1)
    amplitudes = [np.linalg.norm(coeff[:, 2 * m - 1]) + np.linalg.norm(coeff[:, 2 * m])
                  for m in range(1, order + 1)]
    return [sum(a * (2 * np.pi * m) ** d for m, a in enumerate(amplitudes, 1))
            for d in (1, 2, 3)]


def surface_cover_radius(s, nphi, ntheta):
    """Conservative Euclidean distance to the nearest tensor-grid point.

    Follows the Fourier derivative triangle bound, in native normalized angles.
    The surface is trusted and fixed, not supplied by the candidate.
    """
    rbound = 0.0
    rphi = 0.0
    zphi = 0.0
    rtheta = 0.0
    ztheta = 0.0
    for m in range(s.mpol + 1):
        for n in range(-s.ntor, s.ntor + 1):
            if m == 0 and n < 0:
                continue
            rc = abs(s.get_rc(m, n))
            zs = abs(s.get_zs(m, n))
            rbound += rc
            rphi += s.nfp * abs(n) * rc
            zphi += s.nfp * abs(n) * zs
            rtheta += m * rc
            ztheta += m * zs
    bphi = 2 * np.pi * (rbound + rphi + zphi)
    btheta = 2 * np.pi * (rtheta + ztheta)
    return bphi / (2 * nphi) + btheta / (2 * ntheta)


def self_distance_lower(curve, order, exclusion, search_radius):
    """Conservative nonlocal separation, with bounded-memory neighbor queries.

    Physical positions and derivatives come from the native curve. Trapezoidal
    arc increments have total error <= A/(4N) for A bounding |gamma''|.
    Moving both exact points to the nearest grid points adds <= V/N in arc.
    We therefore enlarge the eligible arc window by BOTH bounds. A distance
    query is capped at search_radius, which remains a conservative lower bound
    after subtracting the position cover. Only base curves are needed because
    nominal symmetries and assembly errors are rigid isometries.
    """
    g = curve.gamma()
    count = len(g)
    vmax, amax, _ = curve_derivative_bounds(curve, order)
    cover = vmax / (2 * count)
    speed = np.linalg.norm(curve.gammadash(), axis=1)
    increments = .5 * (speed + np.roll(speed, -1)) / count
    arc = np.concatenate(([0.], np.cumsum(increments[:-1])))
    total = float(increments.sum())
    arc_error = amax / (4 * count)
    best = search_radius
    # No pair can have shorter-arc separation >= exclusion in this case.
    if (total + arc_error) / 2 < exclusion:
        return best - 2 * cover
    tree = cKDTree(g)
    eligible = exclusion - 2 * cover - arc_error
    # One query retains at most 64*N indices instead of O(N^2) pairs.
    for start in range(0, count, 64):
        neighborhoods = tree.query_ball_point(g[start:start + 64], best, workers=1)
        for i, neighbors in enumerate(neighborhoods, start):
            indices = np.asarray(neighbors, dtype=int)
            indices = indices[indices > i]
            if not len(indices):
                continue
            delta = abs(arc[indices] - arc[i])
            indices = indices[np.minimum(delta, total - delta) >= eligible]
            if len(indices):
                best = min(best, float(np.linalg.norm(g[indices] - g[i], axis=1).min()))
    return best - 2 * cover


def geometry(data, case, root, details):
    from simsopt.geo import CurveLength, MeanSquaredCurvature, LinkingNumber, CurveXYZFourier
    count = case["geometry"]["curve_points"]
    base, currents, coils = reconstruct(data, count)
    exact = [materialize(c, data["order"], count) for c in coils]
    gamma = [c.gamma() for c in exact]
    lengths = [float(CurveLength(c).J()) for c in base]
    length_upper = [length + curve_derivative_bounds(c, data["order"])[1] / (4 * count)
                    for length, c in zip(lengths, base)]
    errors = []
    curvature_upper = []
    msc_diagnostic = []
    radius_bounds = []
    centers = []
    speed_ratio = []
    for c in exact:
        vmax, amax, jmax = curve_derivative_bounds(c, data["order"])
        speed = np.linalg.norm(c.gammadash(), axis=1)
        vmin = float(speed.min()) - amax / (2 * count)
        if vmin <= 1e-5 or not np.isfinite(c.kappa()).all():
            raise InvalidSubmission("curve has a zero or unresolved tangent")
        errors.append(vmax / (2 * count))
        kappa_lipschitz = jmax / vmin**2 + 3 * amax**2 / vmin**3
        curvature_upper.append(float(max(c.kappa())) + kappa_lipschitz / (2 * count))
        msc = float(MeanSquaredCurvature(c).J())
        msc_diagnostic.append(msc)
        center = np.asarray(c.centroid(), dtype=float)
        centers.append(center)
        radius_bounds.append(float(np.linalg.norm(c.gamma() - center, axis=1).max()) + errors[-1])
        speed_ratio.append(float(speed.max() / speed.min()))
    movement = [np.sqrt(3) * case["error"]["translation_halfwidth_m"]
                + 2 * r * np.sin(np.sqrt(3) * case["error"]["rotation_halfwidth_rad"] / 2)
                for r in radius_bounds]
    cc_lower = float("inf")
    for i in range(len(gamma)):
        tree = cKDTree(gamma[i])
        for j in range(i):
            distance = float(tree.query(gamma[j], workers=1)[0].min())
            cc_lower = min(cc_lower, distance - errors[i] - errors[j] - movement[i] - movement[j])
    gcfg = case["geometry"]
    target = surface(case, root, gcfg["surface_phi"], gcfg["surface_theta"])
    stree = cKDTree(target.gamma().reshape(-1, 3))
    serror = surface_cover_radius(target, gcfg["surface_phi"], gcfg["surface_theta"])
    cs_lower = min(float(stree.query(g, workers=1)[0].min()) - e - m - serror
                   for g, e, m in zip(gamma, errors, movement))
    self_lower = min(self_distance_lower(c, data["order"],
                                        gcfg["self_arclength_exclusion_m"],
                                        gcfg["self_distance_search_m"]) for c in base)
    # Native Gauss linking diagnostics on an independent, affordable quadrature.
    _, _, linkcoils = reconstruct(data, case["geometry"]["link_points"])
    axis = CurveXYZFourier(case["geometry"]["link_points"], 1)
    axis.set("xc(1)", case["link_axis_radius_m"])
    axis.set("ys(1)", case["link_axis_radius_m"])
    link_axis = [float(LinkingNumber([c.curve, axis]).J()) for c in linkcoils]
    interlink = float(LinkingNumber([c.curve for c in linkcoils]).J())
    if not np.isfinite(link_axis).all() or not math.isfinite(interlink) or min(link_axis) < 0 or interlink < 0:
        raise Unresolved("native linking diagnostics are nonfinite or outside their integer domain")
    envelope_min = min(float(np.hypot(g[:, 0], g[:, 1]).min()) - e - m for g, e, m in zip(gamma, errors, movement))
    envelope_max = max(float(np.hypot(g[:, 0], g[:, 1]).max()) + e + m for g, e, m in zip(gamma, errors, movement))
    z_max = max(float(abs(g[:, 2]).max()) + e + m for g, e, m in zip(gamma, errors, movement))
    values = {
        "base_length_sum_upper_m": math.fsum(length_upper), "kappa_upper_per_m": max(curvature_upper),
        "coil_distance_lower_m": cc_lower,
        "plasma_distance_lower_m": cs_lower, "self_distance_lower_m": self_lower,
        "radius_min_lower_m": envelope_min, "radius_max_upper_m": envelope_max,
        "z_max_upper_m": z_max, "link_axis_error": max(abs(x - 1) for x in link_axis),
        "interlink": interlink,
        "first_current_relative_error": abs(data["coils"][0]["current"] / case["first_current_Aturn"] - 1),
        "current_min_Aturn": min(c["current"] for c in data["coils"]),
        "current_max_Aturn": max(c["current"] for c in data["coils"]),
    }
    details["geometry"] = {"metrics": values, "lengths_m": lengths,
                            "cover_curve_m": errors, "cover_surface_m": serror,
                            "movement_bound_m": movement, "link_axis": link_axis,
                            "speed_ratio_diagnostic": speed_ratio,
                            "mean_squared_curvature_diagnostic_per_m2": msc_diagnostic}
    return values, centers


def field_level(data, case, root, errors, centers, level, evidence_dir, shifted=False):
    from simsopt.field import BiotSavart, Coil, Current
    from simsopt.objectives import SquaredFlux
    nphi, ntheta, count, section_count = level
    s = surface(case, root, nphi, ntheta, shifted=shifted)
    _, _, coils = reconstruct(data, count)
    nominal = [materialize(c, data["order"], count) for c in coils]
    p = s.gamma().reshape(-1, 3)
    phi_unit = np.column_stack((-p[:, 1], p[:, 0], np.zeros(len(p)))) / np.hypot(p[:, 0], p[:, 1])[:, None]
    weight = np.linalg.norm(s.normal(), axis=2).ravel()
    active = [materialize(c, data["order"], count) for c in coils]
    current_coils = []
    for curve, original in zip(active, coils):
        filaments = pack_filaments(curve, original.pack_rotation, case, section_count)
        current_coils.extend(Coil(filament, Current(original.current.get_value()*filament.quadrature_weight))
                             for filament in filaments)
    b = BiotSavart(current_coils)
    flux = SquaredFlux(s, b, definition="normalized")
    out = []
    for sample_index, sample in enumerate([None, *errors]):
        for j, original in enumerate(coils):
            curve = nominal[j] if sample is None else rigid_transform(
                nominal[j], sample[j, :3], sample[j, 3:], centers[j], data["order"], count)
            # The same rigid motion carries every material point of one pack.
            # The centroid frame is covariant under these proper rotations.
            active[j].local_full_x = curve.local_full_x
        native = float(flux.J())
        field = np.asarray(b.B())
        norm = np.linalg.norm(field, axis=1)
        if native < 0 or not np.isfinite(field).all() or not math.isfinite(native):
            raise Unresolved("native magnetic field/normalized flux is nonfinite")
        result = {"e": math.sqrt(2 * native),
                  "brms_T": float(np.sqrt(np.dot(weight, norm**2) / weight.sum())),
                  "bmin_T": float(norm.min()), "bmax_T": float(norm.max()),
                  "negative_bphi_min_T": float((-np.sum(field * phi_unit, axis=1)).min())}
        out.append(result)
        # Preserve the actual current-candidate native field for independent analysis.
        np.savez_compressed(evidence_dir / f"field_{nphi}_{ntheta}_{count}_{section_count}_{sample_index:03d}.npz",
                            points=p, B_T=field, normal=s.unitnormal().reshape(-1, 3), area_weight=weight)
    return out


def violation_metrics(metrics, constraints):
    violations = {}
    for name, c in constraints.items():
        value = metrics[name]
        diff = value - c["threshold"] if c["sense"] == "upper" else c["threshold"] - value
        violations[name] = max(0.0, diff / c["scale"])
    return violations


def score(metrics, reference, constraints):
    violations = violation_metrics(metrics, constraints)
    if any(v > 0 for v in violations.values()):
        return min(math.nextafter(0.5, 0.0), 0.5 * math.exp(-sum(v*v for v in violations.values()))), False, violations
    if not math.isfinite(reference) or reference <= 0:
        raise RuntimeError("reference robust objective must be calibrated and strictly positive")
    return 0.5 + 0.5 * reference / (reference + metrics["J"]), True, violations


def physical_run(data, case, root, work, details):
    values, centers = geometry(data, case, root, details)
    pack_geometry(data, case, values, details)
    bank = np.load(work / "scenarios.npz", allow_pickle=False)["errors"]
    if bank.shape != (case["scenario_count"], 16, 6) or not np.isfinite(bank).all():
        raise RuntimeError("trusted scenario bank has incorrect shape/content")
    if np.any(abs(bank[:, :, :3]) > case["error"]["translation_halfwidth_m"]) or np.any(abs(bank[:, :, 3:]) > case["error"]["rotation_halfwidth_rad"]):
        raise RuntimeError("scenario bank exceeds the declared physical distribution")
    levels = []
    for grid in case["field_grids"]:
        levels.append(field_level(data, case, root, bank, centers, grid, work))
        details["completed_field_levels"] = len(levels)
        (work / "stage.json").write_text(json.dumps(details, indent=2, allow_nan=False))
    upper = []
    resolution_errors = []
    for i, result in enumerate(levels[-1]):
        previous = levels[-2][i]
        spatial_previous = levels[-3][i]
        # Last two levels change only cross-section quadrature; the preceding
        # pair changes only surface/line grids. Never let their errors cancel.
        de = abs(result["e"] - previous["e"]) + abs(previous["e"] - spatial_previous["e"])
        if de > max(case["numeric"]["e_absolute_tol"], case["numeric"]["e_relative_tol"] * result["e"]):
            raise Unresolved(f"field scenario {i}: e did not resolve on final grid")
        upper.append(result["e"] + de)
        resolution_errors.append(de)
    # An independent, denser and half-cell-shifted audit checks the nominal
    # state and each bank's worst resolved sample. It is mandatory, and any
    # discrepancy is added to the objective, never used to improve the score.
    selected = [0] + [1 + start + int(np.argmax(upper[1+start:1+start+case["batch_size"]]))
                      for start in range(0, len(bank), case["batch_size"])]
    audit_dir = work / "audit"
    audit_dir.mkdir()
    audit = field_level(data, case, root, bank[np.array(selected[1:])-1], centers,
                        case["audit_grid"], audit_dir, shifted=True)
    for index, row in zip(selected, audit):
        diff = abs(row["e"] - levels[-1][index]["e"])
        if diff > max(case["numeric"]["e_absolute_tol"], case["numeric"]["e_relative_tol"] * row["e"]):
            raise Unresolved(f"field scenario {index}: independent shifted audit did not resolve")
        upper[index] = max(row["e"], levels[-1][index]["e"]) + resolution_errors[index] + diff
    values["nominal_e_upper"] = upper[0]
    values["J"] = max(upper[1:])
    # Extremum convergence is only numerical evidence, not a proof over all points.
    # Publish conservative last-grid bounds and refuse poorly resolved extrema.
    for name in ("brms_T", "bmin_T", "bmax_T", "negative_bphi_min_T"):
        last = np.array([r[name] for r in levels[-1]])
        prev = np.array([r[name] for r in levels[-2]])
        spatial_prev = np.array([r[name] for r in levels[-3]])
        delta = abs(last - prev) + abs(prev - spatial_prev)
        if delta.max() > case["numeric"]["field_T_tol"]:
            raise Unresolved(f"{name}: magnetic diagnostic grid not resolved")
        for index, row in zip(selected, audit):
            diff = abs(row[name] - last[index])
            if diff > case["numeric"]["field_T_tol"]:
                raise Unresolved(f"{name}: independent shifted audit did not resolve")
            delta[index] += diff
        if name == "brms_T":
            values["brms_relative_error_upper"] = float(np.max(abs(last / case["field_rms_T"] - 1) + delta / case["field_rms_T"]))
        elif name == "bmax_T":
            values[name + "_upper"] = float(np.max(last + delta))
        else:
            values[name + "_lower"] = float(np.min(last - delta))
    details["field"] = {"levels": levels, "scenario_e_upper": upper[1:],
                         "resolution_error": resolution_errors,
                         "J_batches": [max(upper[1:][i:i + case["batch_size"]]) for i in range(0, len(bank), case["batch_size"])],
                         "audit": {"scenario_indices_with_nominal_zero": selected, "values": audit,
                                   "grid": case["audit_grid"], "surface_half_cell_shift": True},
                         "scope": "maximum in the frozen finite scenario bank, not a distribution quantile"}
    return values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("submission", type=Path)
    p.add_argument("--assets", type=Path, default=Path(__file__).resolve().parent / "assets")
    p.add_argument("--scenarios", type=Path)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--reward-dir", type=Path, required=True)
    p.add_argument("--measure", action="store_true")
    args = p.parse_args()
    args.reward_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    details = {"valid": False, "feasible": False, "reward": 0.0, "metrics": None, "status": "evaluator_error", "evaluator_error": True}
    code = 1
    try:
        import simsopt
        case_path = args.assets / "case.json"
        case = json.loads(case_path.read_text())
        if sha256(args.assets / "reference/target.vmec") != case["target_sha256"]:
            raise RuntimeError("trusted fixed target identity mismatch")
        if simsopt.__version__ != case["simsopt_version"]:
            raise RuntimeError(f"requires SIMSOPT {case['simsopt_version']}; got {simsopt.__version__}")
        bank_path = args.scenarios or args.assets / "scenarios.npz"
        if not args.measure:
            lock = json.loads((args.assets / "release-lock.json").read_text())
            baseline = json.loads((args.assets / "baseline.json").read_text())
            if lock["state"] != "verified" or baseline["state"] != "calibrated":
                raise RuntimeError("C0 image/baseline is not frozen: use --measure during authoring")
            if baseline["case_sha256"] != sha256(case_path) or baseline["scenarios_sha256"] != sha256(bank_path):
                raise RuntimeError("calibration identity mismatch")
            if baseline["evaluator_sha256"] != sha256(__file__) or lock["evaluator_sha256"] != sha256(__file__):
                raise RuntimeError("calibrated evaluator identity mismatch")
            if "@sha256:" not in lock["image_digest"] or lock["isolation"] != "clean_trusted_verifier":
                raise RuntimeError("immutable image and clean trusted verifier are required")
            if not lock["native_modules"]:
                raise RuntimeError("native physics identity manifest is empty")
            from importlib.util import find_spec
            for module, expected in lock["native_modules"].items():
                actual = find_spec(module).origin
                if sha256(actual) != expected:
                    raise RuntimeError(f"native physics module identity mismatch: {module}")
        if args.workdir.exists() and any(args.workdir.iterdir()):
            raise RuntimeError("workdir must be nonexistent or empty")
        args.workdir.mkdir(parents=True, exist_ok=True)
        data = read_submission(args.submission, case)
        snapshot = args.workdir / "coils.json"
        snapshot.write_text(json.dumps(data, indent=2, allow_nan=False))
        (args.workdir / "scenarios.npz").write_bytes(bank_path.read_bytes())
        details.update(submission_sha256=sha256(snapshot), case_sha256=sha256(case_path), evaluator_sha256=sha256(__file__),
                       scenarios_sha256=sha256(bank_path), simsopt_version=simsopt.__version__)
        metrics = physical_run(data, case, args.assets, args.workdir, details)
        if not all(math.isfinite(x) for x in metrics.values()):
            raise Unresolved("physical metrics contain a nonfinite diagnostic")
        details.update(valid=True, metrics=metrics, evaluator_error=False)
        if args.measure:
            details.update(status="measured_unfrozen", feasible=None, reward=0.0,
                           reason="authoring measurement only; no competition reward")
        else:
            reward, feasible, violations = score(metrics, baseline["J_ref"], case["constraints"])
            details.update(reward=reward, feasible=feasible, violations=violations, status="feasible" if feasible else "infeasible")
        code = 0
    except InvalidSubmission as exc:
        details.update(status="invalid_submission", evaluator_error=False, reason=str(exc))
        code = 0
    except Unresolved as exc:
        details.update(status="candidate_unresolved", evaluator_error=False, reason=str(exc))
        code = 0
    except Exception as exc:
        details.update(status="evaluator_error", evaluator_error=True, valid=False, feasible=False,
                       reward=0.0, reason=str(exc), traceback=traceback.format_exc())
    details["wall_seconds"] = time.monotonic() - start
    (args.reward_dir / "details.json").write_text(json.dumps(details, indent=2, allow_nan=False))
    reward = {"reward": details["reward"], "evaluator_error": int(details["evaluator_error"])}
    (args.reward_dir / "reward.json").write_text(json.dumps(reward, allow_nan=False))
    print(json.dumps({k: details[k] for k in ("status", "valid", "feasible", "reward", "wall_seconds")}, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
