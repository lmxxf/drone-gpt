"""Native TokaMaker worker for prescribed-profile target equilibria.

One process owns one native solver. This module defines inputs and gathers
native outputs; it implements no equilibrium or transport equations.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.eqdsk import _read_geqdsk
from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import Shape, boundary


def run(job, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    start = time.monotonic()
    settings = job["resolution"]
    contour, geometry = boundary(Shape(**job["shape"]), volume=job["volume_m3"],
                                 npoints=settings["boundary_points"])
    np.save(output / "boundary.npy", contour)
    env = OFT_env(nthreads=2, unique_tempfiles="local_dir")
    gs = TokaMaker(env)
    mesh = gs_Domain()
    mesh.define_region("plasma", settings["mesh_m"], "plasma")
    mesh.add_polygon(contour, "plasma", angle_tol=0.001, small_thresh=0.0)
    points, cells, _ = mesh.build_mesh()
    gs.setup_mesh(points, cells)
    gs.settings.free_boundary = False
    gs.settings.nl_tol = 1e-10
    gs.settings.maxits = 120
    gs.setup(order=settings["order"], F0=job["R0_m"]*job["B0_T"])
    psi = np.linspace(0, 1, settings["profile_points"])
    gs.set_profiles(
        pp_prof={"type": "linterp", "x": psi, "y": 1-psi},
        ffp_prof={"type": "linterp", "x": psi,
                  "y": (1-psi)*(1+job["ffprime_hollow"]*psi)},
    )
    gs.set_targets(Ip=job["current_A"], pax=job["axis_pressure_Pa"])
    gs.init_psi()
    _, iterations = gs.solve(return_its=True)
    stats = gs.get_stats(lcfs_pad=0.001)
    gs.save_eqdsk(str(output / "equilibrium.geqdsk"), nr=settings["box_points"],
                 nz=settings["box_points"], lcfs_pad=0.001, truncate_eq=False,
                 rcentr=job["R0_m"], cocos=2)
    exported = _read_geqdsk(str(output / "equilibrium.geqdsk"))
    report = {"status": "equilibrium_solved", "iterations": iterations,
              "stats": stats, "geometry": geometry,
              "average_pressure_Pa": stats["W_MHD"]/(1.5*stats["vol"]),
              "exported_current_A": exported["CURRENT"],
              "exported_axis_pressure_Pa": exported["PRES"][0],
              "elapsed_seconds": time.monotonic()-start}
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    (output / "report.json").write_text(json.dumps(report, indent=2, default=encode,
                                                 allow_nan=False))
    print(json.dumps(report, default=encode, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(json.loads(args.job.read_text()), args.output)
