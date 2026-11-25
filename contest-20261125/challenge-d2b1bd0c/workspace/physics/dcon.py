"""Replay a GEQDSK with native DCON and preserve all diagnostics, without grading."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
from netCDF4 import Dataset
from scipy.io import FortranFile


def diagnostics(directory, mode):
    result = {}
    with Dataset(directory / f"dcon_output_n{mode}.nc") as data:
        result["globals"] = {key: float(data.getncattr(key))
                             for key in ("betan", "betat", "bt0", "crnt", "qmin", "q95", "psilim")}
        result["radial_points"] = len(data["psi_n"][:])
        energy = np.asarray(data["W_t_eigenvalue"][:])
        index = int(np.argmin(energy[0]))
        result["minimum_energy"] = energy[:, index].tolist()
        result["mercier_di_max"] = float(np.max(data["di"][:]))
        result["mercier_di_max_outside_1percent"] = float(np.max(
            data["di"][:][data["psi_n"][:] >= 0.01]))
    with FortranFile(directory / "gsec.bin") as data:
        data.read_ints()
        shape = data.read_ints() + 1
        for _ in range(6):
            error = data.read_reals(np.float32)
        result["gs_error_max"] = float(np.max(np.abs(error)))
        result["gs_error_rms"] = float(np.sqrt(np.mean(error**2)))
        result["gs_error_shape"] = shape.tolist()
    log = (directory / "run.log").read_text()
    result["normal_termination"] = "PROGRAM STOP => Normal termination." in log
    result["internal_zero_crossings"] = len(re.findall("Zero crossing at psi", log))
    summary = directory / "ballooning_summary.dat"
    if summary.exists():
        rows = [line.split() for line in summary.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        result["ballooning_integrations"] = len(rows)
        result["ballooning_nodes_max"] = max((int(row[2]) for row in rows), default=0)
        result["ballooning_incomplete"] = sum(row[4] != "T" for row in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("equilibrium", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--modes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--refinement", type=int, choices=[1, 2], default=1)
    parser.add_argument("--normalize-ballooning", action="store_true")
    parser.add_argument("--theta0", type=float)
    parser.add_argument("--ballooning-rtol", type=float)
    parser.add_argument("--poloidal-padding", type=int, default=8)
    parser.add_argument("--vacuum-points", type=int, default=512)
    parser.add_argument("--ballooning-extent", type=float, default=1.0)
    parser.add_argument("--ballooning-cap", type=float, default=100.0)
    parser.add_argument("--edge", type=float)
    parser.add_argument("--jacobian", choices=["hamada", "pest"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"kind": "native_stability_diagnostics", "modes": {}}
    for mode in args.modes:
        directory = args.output / f"n{mode}"
        directory.mkdir()
        for name in ("equil.in", "dcon.in", "vac.in"):
            shutil.copy2(args.template / name, directory / name)
        path = directory / "dcon.in"
        text = re.sub(r"(?m)^(\s*nn\s*=)\s*\d+", rf"\g<1>{mode}", path.read_text())
        extra = []
        if args.normalize_ballooning:
            extra.append("bal_normalize=t")
        if args.theta0 is not None:
            extra.append(f"theta0={args.theta0:.17g}")
        if args.ballooning_rtol is not None:
            extra.append(f"bal_rtol={args.ballooning_rtol:.17g}")
        extra.append(f"bal_thmax_cap={args.ballooning_cap:.17g}")
        for key in ("delta_mlow", "delta_mhigh"):
            text = re.sub(rf"{key}\s*=\s*\d+", f"{key}={args.poloidal_padding}", text)
        text = re.sub(r"thmax0\s*=\s*[\d.eE+-]+", f"thmax0={args.ballooning_extent:.17g}", text)
        text = re.sub(r"mthvac\s*=\s*\d+", f"mthvac={args.vacuum_points}", text)
        text = text.replace("&DCON_CONTROL", "&DCON_CONTROL\n    " + "\n    ".join(extra))
        path.write_text(text)
        path = directory / "equil.in"
        text = re.sub(r'eq_type\s*=\s*"[^"]+"', 'eq_type="efit"', path.read_text())
        if args.edge is not None:
            text = re.sub(r"psihigh\s*=\s*[\d.eE+-]+", f"psihigh={args.edge:.17g}", text)
        if args.jacobian is not None:
            text = re.sub(r'jac_type\s*=\s*"[^"]+"', f'jac_type="{args.jacobian}"', text)
        path.write_text(text)
        if args.refinement == 2:
            path = directory / "equil.in"
            text = path.read_text()
            for name in ("mpsi", "mtheta"):
                text = re.sub(rf"(?m)^(\s*{name}\s*=)\s*(\d+)",
                              lambda m: m[1] + str(2 * int(m[2])), text)
            path.write_text(text)
        shutil.copy2(args.equilibrium, directory / "equilibrium.geqdsk")
        start = time.monotonic()
        with (directory / "run.log").open("w") as log:
            process = subprocess.run([str(args.binary.resolve())], cwd=directory,
                stdout=log, stderr=subprocess.STDOUT, timeout=900,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        entry = {"exit_code": process.returncode, "elapsed_seconds": time.monotonic()-start}
        if process.returncode == 0:
            entry.update(diagnostics(directory, mode))
        report["modes"][str(mode)] = entry
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({str(mode): entry}), flush=True)
        if process.returncode:
            raise RuntimeError(f"DCON n={mode} failed, see {directory / 'run.log'}")


if __name__ == "__main__":
    main()
