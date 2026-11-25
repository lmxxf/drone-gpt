"""Prescribed-profile tokamak shape evaluator.

Runs maintained TokaMaker/DCON in fresh subprocesses. Candidate input contains
only seven numeric design values. No submitted code or claimed score executes.
"""
import argparse
import json
import math
import os
import signal
import time
import resource
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import Shape, boundary


class InvalidDesign(ValueError):
    pass


class NumericalFailure(RuntimeError):
    pass


def finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def read_design(path, protocol):
    if not path.is_file():
        raise InvalidDesign("Design JSON is missing")
    if path.stat().st_size > 16384:
        raise InvalidDesign("Design JSON exceeds 16 KiB")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidDesign(f"Duplicate design key: {key}")
            result[key] = value
        return result
    try:
        design = json.loads(path.read_text(), object_pairs_hook=unique)
    except (ValueError, UnicodeError) as exc:
        raise InvalidDesign(f"Invalid design JSON: {exc}") from exc
    bounds = dict(protocol["shape_bounds"])
    bounds["axis_pressure_Pa"] = protocol["axis_pressure_Pa_bounds"]
    if not isinstance(design, dict) or set(design) != set(bounds):
        raise InvalidDesign("Design must contain exactly the six shape coefficients and axis_pressure_Pa")
    for key, (low, high) in bounds.items():
        value = design[key]
        if not finite_number(value) or not low <= value <= high:
            raise InvalidDesign(f"{key} must be finite and within [{low}, {high}]")
    try:
        _, geom = boundary(Shape(**{k: design[k] for k in protocol["shape_bounds"]}),
                           volume=protocol["volume_m3"], npoints=4096)
    except ValueError as exc:
        raise InvalidDesign(f"Invalid plasma section: {exc}") from exc
    actual = geom["bounds_m"]
    limit = protocol["envelope_m"]
    if actual[0] < limit[0] or actual[1] < limit[1] or actual[2] > limit[2] or actual[3] > limit[3]:
        raise InvalidDesign("Normalized plasma section exceeds the fixed spatial envelope")
    return design


def execute(command, log, timeout):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1"}
    with log.open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise NumericalFailure(f"Native calculation timed out: {log.name}") from exc
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if returncode:
        raise NumericalFailure(f"Native calculation failed ({returncode}): {log.name}")


def relative_error(value, target):
    if not finite_number(value):
        return math.inf
    return abs(value-target)/abs(target)


def check_equilibrium(result, protocol, pressure):
    a = protocol["acceptance"]
    stats = result["stats"]
    checks = [(stats["vol"], protocol["volume_m3"], a["volume_relative"], "volume"),
              (stats["Ip"], protocol["current_A"], a["current_relative"], "current"),
              (result["exported_current_A"], protocol["current_A"], a["current_relative"], "exported current"),
              (result["exported_axis_pressure_Pa"], pressure, a["pressure_target_relative"], "axis pressure")]
    for value, target, tolerance, name in checks:
        if relative_error(value, target) > tolerance:
            raise NumericalFailure(f"Equilibrium does not reproduce the prescribed {name}")
    if not finite_number(result["average_pressure_Pa"]) or result["average_pressure_Pa"] <= 0:
        raise NumericalFailure("Invalid native average pressure")


def check_stability(result, protocol, modes=None):
    a = protocol["acceptance"]
    violations = []
    expected = protocol["modes"] if modes is None else modes
    if set(result["modes"]) != set(map(str, expected)):
        raise NumericalFailure("DCON diagnostics do not contain every requested mode")
    for mode, item in result["modes"].items():
        if item["exit_code"] or not item.get("normal_termination"):
            raise NumericalFailure(f"Incomplete DCON calculation for n={mode}")
        scalars = [item["mercier_di_max"], item["gs_error_rms"], item["gs_error_max"], *item["minimum_energy"],
                   item["globals"]["qmin"], item["globals"]["crnt"]]
        if not all(finite_number(v) for v in scalars):
            raise NumericalFailure("Non-finite DCON diagnostics")
        if item["gs_error_rms"] > a["gs_residual_rms"] or item["gs_error_max"] > a["gs_residual_max"]:
            raise NumericalFailure("DCON reconstructed equilibrium residual exceeds the calibrated bound")
        if relative_error(item["globals"]["crnt"]*1e6, protocol["current_A"]) > a["dcon_current_relative"]:
            raise NumericalFailure("DCON current disagrees with the equilibrium target")
        energy, imaginary = item["minimum_energy"]
        if abs(imaginary) > max(1e-6, abs(energy)*a["energy_imaginary_relative"]):
            raise NumericalFailure("DCON minimum energy has a significant imaginary component")
        if item["globals"]["qmin"] < a["qmin"]:
            violations.append(f"n={mode}: q minimum below the specified operating bound")
        if energy < a["energy_floor"]:
            violations.append(f"n={mode}: insufficient ideal-MHD energy margin")
        if item["internal_zero_crossings"]:
            violations.append(f"n={mode}: internal ideal mode detected")
        if item["mercier_di_max"] > a["mercier_ceiling"]:
            violations.append("Mercier margin fails")
        if item.get("ballooning_incomplete", 1):
            raise NumericalFailure("Ballooning integration did not complete")
        if item.get("ballooning_integrations") != item["radial_points"]:
            if item["mercier_di_max"] > 0:
                violations.append("Mercier-unstable surfaces were not integrated for ballooning")
            else:
                raise NumericalFailure("Ballooning diagnostics do not cover every specified surface")
        if item.get("ballooning_nodes_max", 0):
            violations.append("Ideal ballooning mode detected")
    return sorted(set(violations))


def evaluate(design, protocol, work, binary, mode):
    here = Path(__file__).resolve().parent
    shape = {key: design[key] for key in protocol["shape_bounds"]}
    all_cases = {}
    violations = []
    pressures = []
    levels = ["coarse"] if mode == "search" else ["coarse", "fine"]
    for scenario in protocol["scenarios"]:
        results = {}
        for level in levels:
            root = work / f"{scenario['name']}-{level}"
            root.mkdir()
            job = {key: protocol[key] for key in ("R0_m", "B0_T", "volume_m3", "current_A")}
            job.update(shape=shape, axis_pressure_Pa=design["axis_pressure_Pa"],
                       resolution=protocol[level], ffprime_hollow=scenario["ffprime_hollow"])
            (root / "job.json").write_text(json.dumps(job))
            execute([sys.executable, "-I", str(here / "equilibrium.py"), str(root / "job.json"),
                     str(root / "equilibrium")], root / "equilibrium.log", 900)
            eq = json.loads((root / "equilibrium/report.json").read_text())
            check_equilibrium(eq, protocol, design["axis_pressure_Pa"])
            refinement = 1 if level == "coarse" else 2
            base = [sys.executable, "-I", str(here / "dcon.py"),
                    str(root / "equilibrium/equilibrium.geqdsk")]
            options = ["--binary", str(binary), "--template", str(here / "assets/templates"),
                       "--normalize-ballooning", "--refinement", str(refinement),
                       "--ballooning-rtol", str(protocol[level]["ballooning_rtol"]),
                       "--poloidal-padding", str(protocol[level]["poloidal_padding"]),
                       "--vacuum-points", str(protocol[level]["vacuum_points"]),
                       "--ballooning-extent", str(protocol[level]["ballooning_extent"]),
                       "--ballooning-cap", str(protocol[level]["ballooning_cap"]),
                       "--jacobian", "pest"]
            execute(base + [str(root / "modes")] + options + ["--modes", *map(str, protocol["modes"])],
                    root / "modes.log", 3600)
            stability = json.loads((root / "modes/report.json").read_text())
            failures = check_stability(stability, protocol)
            if failures:
                results[level] = {"equilibrium": eq, "stability": stability,
                                  "phases": {}, "violations": failures}
                all_cases[scenario["name"]] = results
                return {"status": "infeasible", "mode": mode, "reward": 0.0,
                        "worst_average_pressure_Pa": None,
                        "violations": [f"{scenario['name']}/{level}: {x}" for x in failures],
                        "cases": all_cases, "all_cases_evaluated": False,
                        "protocol_state": protocol["state"]}
            phases = {}
            for index, phase in enumerate(protocol["phases_"+level]):
                if phase == 0:
                    continue
                destination = root / f"phase-{index}"
                execute(base + [str(destination)] + options + ["--modes", "1", "--theta0", str(phase)],
                        root / f"phase-{index}.log", 900)
                phase_result = json.loads((destination / "report.json").read_text())
                failures += check_stability(phase_result, protocol, modes=[1])
                phases[str(phase)] = phase_result
            results[level] = {"equilibrium": eq, "stability": stability, "phases": phases,
                              "violations": sorted(set(failures))}
            violations += [f"{scenario['name']}/{level}: {message}" for message in set(failures)]
            all_cases[scenario["name"]] = results
            (work / "progress.json").write_text(json.dumps(all_cases, indent=2, allow_nan=False))
        if mode == "formal":
            a = protocol["acceptance"]
            coarse, fine = results["coarse"], results["fine"]
            if relative_error(fine["equilibrium"]["average_pressure_Pa"],
                              coarse["equilibrium"]["average_pressure_Pa"]) > a["pressure_refinement_relative"]:
                raise NumericalFailure("Average pressure did not converge under refinement")
            for n in map(str, protocol["modes"]):
                old, new = coarse["stability"]["modes"][n], fine["stability"]["modes"][n]
                if abs(old["globals"]["qmin"]-new["globals"]["qmin"]) > a["q_refinement_absolute"]:
                    raise NumericalFailure("q minimum did not converge under refinement")
                if abs(old["minimum_energy"][0]-new["minimum_energy"][0]) > (
                    a["energy_refinement_absolute"]+a["energy_refinement_relative"]*abs(new["minimum_energy"][0])):
                    raise NumericalFailure(f"n={n} minimum energy did not converge under refinement")
        pressures.append(min(x["equilibrium"]["average_pressure_Pa"] for x in results.values()))
    pressure = min(pressures)
    score = pressure/(pressure+protocol["score_pressure_scale_Pa"]) if not violations else 0.0
    return {"status": "infeasible" if violations else "feasible", "mode": mode,
            "reward": score, "worst_average_pressure_Pa": pressure,
            "violations": sorted(set(violations)), "cases": all_cases,
            "all_cases_evaluated": True,
            "protocol_state": protocol["state"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--mode", choices=["search", "formal"], default="formal")
    args = parser.parse_args()
    work = args.workdir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((Path(__file__).resolve().parent / "assets/protocol.json").read_text())
    (work / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False))
    start = time.monotonic()
    def interrupted(signum, frame):
        raise NumericalFailure(f"Evaluation interrupted by signal {signum}")
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(signum, interrupted)
    signal.alarm(protocol["evaluation_timeout_seconds"])
    try:
        design = read_design(args.candidate, protocol)
        result = evaluate(design, protocol, work, args.binary.resolve(), args.mode)
    except InvalidDesign as exc:
        result = {"status": "invalid_submission", "reward": 0.0, "reason": str(exc)}
    except NumericalFailure as exc:
        result = {"status": "numerical_failure", "reward": 0.0, "reason": str(exc)}
    finally:
        signal.alarm(0)
    result["elapsed_seconds"] = time.monotonic()-start
    result["max_child_process_rss_KiB"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    (work / "details.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (work / "reward.json").write_text(json.dumps({"reward": result["reward"]}))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, allow_nan=False))


if __name__ == "__main__":
    main()
