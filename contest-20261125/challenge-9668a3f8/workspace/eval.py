#!/usr/bin/env python3
"""Trusted T0 evaluator: validate causal controller, call native physics, score."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


class Unresolved(RuntimeError):
    pass


def module(name):
    path = Path(__file__).resolve().parent / (name + '.py')
    spec = importlib.util.spec_from_file_location('t0_' + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def score(energy_J, reference_J, violation):
    if not all(math.isfinite(x) for x in (energy_J, reference_J)) or reference_J <= 0 or energy_J < 0:
        raise ArithmeticError('Energy comparison requires finite, nonnegative native results')
    if violation:
        return min(float(np.nextafter(0.5, 0)), 0.5 * math.exp(-sum(x*x for x in violation.values())))
    # Strictly monotone. Reference replay is .75; no invented "expert best".
    return 0.5 + 0.5 * reference_J / (reference_J + energy_J)


def release_gate(directory, *, development):
    case = read_json(directory / 'case.json')
    runtime = read_json(directory / 'runtime-lock.json')
    if case['state'] != 'frozen' or runtime['state'] != 'verified':
        raise RuntimeError('T0 is an authoring draft; native baseline/image/precision gate is not frozen')
    if not runtime.get('image_digest') or '@sha256:' not in runtime['image_digest']:
        raise RuntimeError('Missing verified immutable T0 image digest')
    if runtime['isolation'] != 'clean_trusted_verifier':
        raise RuntimeError('Scoring requires a clean trusted verifier with immutable native files')
    for package, version in case['versions'].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Native dependency version mismatch: {package}')
    if not runtime['verified_files'] or not runtime.get('release_evidence'):
        raise RuntimeError('Missing native executable/model hashes or release evidence')
    for record in runtime['verified_files']:
        if sha(record['path']) != record['sha256']:
            raise RuntimeError('Native runtime/model hash mismatch: ' + record['path'])
    evidence = read_json(directory / runtime['release_evidence'])
    for check in ('known_zero_drive', 'causality', 'native_baseline', 'time_resolution',
                  'radial_resolution', 'transport_domain', 'energy_balance',
                  'complete_oracle', 'runtime_isolation'):
        row = evidence.get(check)
        if not row or row['state'] != 'passed' or not row['artifacts']:
            raise RuntimeError('Unverified release check: ' + check)
        for artifact in row['artifacts']:
            if sha(directory / artifact['path']) != artifact['sha256']:
                raise RuntimeError('Release evidence hash mismatch')
    baseline = read_json(directory / 'reference/baseline.json')
    if baseline['state'] != 'calibrated' or baseline['case_sha256'] != sha(directory / 'case.json'):
        raise RuntimeError('Reference baseline is not calibrated for this exact problem')
    key = 'development' if development else 'scoring'
    if baseline[key]['J_reference_J'] <= 0:
        raise RuntimeError('Invalid measured reference energy')
    return case, baseline[key]


def scenario(seed, case):
    rng = np.random.Generator(np.random.PCG64(seed))
    d = case['scenario_distribution']
    return {
        'event_start_s': float(rng.uniform(*d['event_start_s_range'])),
        'Z_eff_after_event': float(rng.uniform(*d['Z_eff_after_event_range'])),
        'initial_temperature_multiplier': float(rng.uniform(*d['initial_temperature_multiplier_range'])),
        # Separate from physical event draws; reproducible even if the control graph changes.
        'sensor_seed': int(rng.integers(0, 2**63)),
    }


def features(history, index, target_MW, applied, command, memory, noise, case):
    observed = history[max(0, index - 1)]
    channels = case['observation']['channels']
    scales = np.array([1e6, 1e6, 1, 1, 1e20, 1, 1, 1, 1e20], dtype=float)
    values = np.array([observed[name] for name in channels], dtype=float) / scales
    values *= 1.0 + noise[index]
    return np.concatenate(([history[index]['time_s'], case['control_dt_s']], values,
                           [target_MW], applied, command, memory))


def violations(case, trace, reference, accepted_trace=None):
    c = case['constraints']
    times = np.array([row['time_s'] for row in trace], dtype=float)
    p = np.array([row['P_fusion_W'] for row in trace], dtype=float)
    p_target = np.interp(times, reference['reference_time_s'], reference['reference_fusion_W'])
    if (p_target <= 0).any() or not np.isfinite(p_target).all():
        raise RuntimeError('Calibrated positive fusion reference is missing')
    relative = (p - p_target) / p_target
    rms = float(np.sqrt(np.trapezoid(relative**2, times) / case['horizon_s']))
    peak = float(np.max(np.abs(relative)))
    accepted = trace if accepted_trace is None else accepted_trace
    fusion_energy = float(np.trapezoid([row['P_fusion_W'] for row in accepted],
                                       [row['time_s'] for row in accepted]))
    result = {
        'fusion_energy_J': fusion_energy,
        'fusion_energy_deficit_fraction': max(0., 1. - fusion_energy / reference['required_fusion_energy_J']),
        'fusion_rms_relative': rms,
        'fusion_peak_relative': peak,
        'q_min': min(row['q_min'] for row in accepted),
        'q95_min': min(row['q95'] for row in accepted),
        'beta_N_max': max(row['beta_N'] for row in accepted),
        'temperature_max_keV': max(row['temperature_max_keV'] for row in accepted),
        'fgw_max': max(row['fgw_n_e_line_avg'] for row in accepted),
        'terminal_energy_deficit_fraction': max(0., 1. - trace[-1]['W_thermal_total_J'] / reference['reference_terminal_W_thermal_J']),
    }
    bounds = {
        'fusion_energy_deficit_fraction': ('upper', 0., .02),
        'fusion_rms_relative': ('upper', c['fusion_rms_relative_upper'], .1),
        'fusion_peak_relative': ('upper', c['fusion_peak_relative_upper'], .25),
        'beta_N_max': ('upper', c['beta_N_upper'], 2.8),
        'temperature_max_keV': ('upper', c['temperature_keV_upper'], 35.),
        'terminal_energy_deficit_fraction': ('upper', c['terminal_energy_deficit_fraction_upper'], .02),
    }
    v = {}
    for name, (sense, limit, scale) in bounds.items():
        if not math.isfinite(result[name]):
            raise Unresolved('Nonfinite native diagnostic: ' + name)
        amount = result[name] - limit if sense == 'upper' else limit - result[name]
        if amount > 0:
            v[name] = amount / scale
    return result, v


def episode(controller_module, plant, candidate, directory, output, seed, level, reference):
    case = read_json(directory / 'case.json')
    output.mkdir(parents=True)
    env = plant.CoreHeatEnv(directory, scenario(seed, case), level)
    try:
        return _episode_in_env(controller_module, plant, candidate, output, seed, reference, case, env)
    finally:
        env.close()


def _episode_in_env(controller_module, plant, candidate, output, seed, reference, case, env):
    env.reset()
    controller = controller_module.Controller(candidate)
    try:
        return _controlled_trajectory(controller, controller_module, plant, output, seed, reference, case, env)
    finally:
        controller.close()


def _controlled_trajectory(controller, controller_module, plant, output, seed, reference, case, env):
    history = [plant.diagnostics(env)]
    background = case['background_admission']
    if (history[0]['q_min'] < background['q_min_lower']
            or history[0]['q95'] < background['q95_lower']
            or history[0]['fgw_n_e_line_avg'] > background['density_greenwald_upper']):
        raise RuntimeError('Prescribed background fails task admission; this is not a controller violation')
    accepted = history.copy()
    (output / 'trace.jsonl').write_text(json.dumps(history[0], allow_nan=False)+'\n')
    memory = np.zeros(case['memory_dim'], dtype=float)
    applied = np.array(case['actuator']['initial_MW'], dtype=float)
    command = applied.copy()
    queued = applied.copy()
    count = int(round(case['horizon_s'] / case['control_dt_s']))
    rng = np.random.Generator(np.random.PCG64(scenario(seed, case)['sensor_seed']))
    sigma = np.array(case['observation']['noise_relative_sigma'], dtype=float)
    z = np.clip(rng.standard_normal((count, len(sigma))), -3., 3.)
    noise = z * sigma
    low = np.array(case['actuator']['lower_MW'], dtype=float)
    high = np.array(case['actuator']['upper_MW'], dtype=float)
    max_delta = np.array(case['actuator']['ramp_MW_per_s']) * case['control_dt_s']
    actions = []
    domains = [{'time_s': history[0]['time_s'], 'inputs': plant.transport_domain(env)}]
    if any(row['outside_faces'] for row in domains[0]['inputs'].values()):
        (output / 'transport-domain.json').write_text(json.dumps(domains, indent=2, allow_nan=False)+'\n')
        raise Unresolved('Initial state lies outside active-region QLKNN training bounds')
    for index in range(count):
        target = float(np.interp(history[index]['time_s'], reference['reference_time_s'], reference['reference_fusion_W'])) / 1e6
        feature = features(history, index, target, applied, command, memory, noise, case)
        action, memory = controller.step(feature)
        if np.any(action < low - 1e-6) or np.any(action > high + 1e-6):
            raise controller_module.InvalidController('Heating command outside physical range')
        if np.any(np.abs(action - command) > max_delta + 1e-6):
            raise controller_module.InvalidController('Heating command violates slew limit')
        # Pure one-window command latency. Gym-TORAX applies the queued endpoint
        # through its maintained piecewise-linear source ramp.
        try:
            env.advance(queued)
        finally:
            accepted.extend(env.last_transition)
            domains.extend(env.last_domains)
            # Retain the last accepted state even if the next native step fails.
            (output / 'accepted-trace.json').write_text(json.dumps(accepted, indent=2, allow_nan=False)+'\n')
            (output / 'transport-domain.json').write_text(json.dumps(domains, indent=2, allow_nan=False)+'\n')
        applied = queued.copy()
        queued = action.copy()
        command = action.copy()
        history.append(plant.diagnostics(env))
        if not all(math.isfinite(x) for x in history[-1].values()):
            raise Unresolved('Nonfinite physical trajectory')
        with (output / 'trace.jsonl').open('a') as stream:
            stream.write(json.dumps(history[-1], allow_nan=False)+'\n')
        actions.append({'time_s': feature[0], 'command_MW': action.tolist(), 'applied_end_MW': applied.tolist()})
    if env.store_history:
        env.save_file(str(output / 'native.nc'))
    else:
        env.torax_app.get_state_data().to_netcdf(output / 'native-final.nc', engine='h5netcdf', mode='w')
    (output / 'trace.json').write_text(json.dumps(history, indent=2, allow_nan=False)+'\n')
    (output / 'actions.json').write_text(json.dumps(actions, indent=2, allow_nan=False)+'\n')
    (output / 'transport-domain.json').write_text(json.dumps(domains, indent=2, allow_nan=False)+'\n')
    final = history[-1]['E_external_total_J'] - history[0]['E_external_total_J']
    if final < 0 or min(row['P_ohmic_W'] for row in history) < -1e-6:
        raise Unresolved('Negative external energy/Ohmic heat requires investigation')
    # P_external_total is upstream injected power plus Ohmic power. For this
    # frozen fully absorbed plant it must equal auxiliary absorption plus Ohmic.
    # Do not reinterpret it as absorbed energy if this contract ever changes.
    if any(abs(row['P_external_W'] - row['P_auxiliary_W'] - row['P_ohmic_W'])
           > 1e-8 * max(1., row['P_external_W']) for row in accepted):
        raise RuntimeError('Native injected/absorbed external-energy contract changed')
    external_integral = float(np.trapezoid([r['P_external_W'] for r in accepted],
                                          [r['time_s'] for r in accepted]))
    if abs(final - external_integral) > 1e-8 * max(1., final):
        raise Unresolved('Native cumulative energy disagrees with its accepted-step power integral')
    metrics, v = violations(case, history, reference, accepted)
    metrics.update(supplied_heat_metrics(accepted, final))
    return metrics, v, history


def supplied_heat_metrics(accepted, final):
    """Account native external heat and positive implicit pedestal supply."""
    dt = np.diff([r['time_s'] for r in accepted])
    pedestal = np.array([r['P_pedestal_W'] for r in accepted])
    positive_pedestal = float(np.sum(np.maximum(pedestal[1:], 0.) * dt))
    negative_pedestal = float(np.sum(np.minimum(pedestal[1:], 0.) * dt))
    external_right = float(np.sum(np.array([r['P_external_W'] for r in accepted[1:]]) * dt))
    return dict(E_external_total_J=final,
                   E_external_right_endpoint_J=external_right,
                   E_external_quadrature_difference_J=external_right-final,
                   E_pedestal_positive_J=positive_pedestal,
                   E_pedestal_negative_J=negative_pedestal,
                   E_pedestal_positive_trapezoidal_J=float(np.trapezoid(np.maximum(pedestal, 0.),
                                                                       [r['time_s'] for r in accepted])),
                   J_supplied_total_J=final+positive_pedestal)


def compare_resolutions(case, baseline, runs):
    """Validate a native coarse/fine pair and return its conservative objective."""
    energies = [r[0]['J_supplied_total_J'] for r in runs]
    discrepancy = abs(energies[0] - energies[1])
    if discrepancy > case['numerics']['relative_objective_tolerance'] * baseline['J_reference_J']:
        raise Unresolved('Supplied-heat objective is not numerically resolved')
    powers = [[x['P_fusion_W'] for x in r[2]] for r in runs]
    if np.max(np.abs(np.array(powers[0]) - np.array(powers[1]))) > case['numerics']['relative_power_tolerance'] * max(baseline['reference_fusion_W']):
        raise Unresolved('Fusion trajectory is not numerically resolved')
    for name in ('q_min', 'q95', 'beta_N'):
        metric = {'q_min': 'q_min', 'q95': 'q95_min', 'beta_N': 'beta_N_max'}[name]
        if abs(runs[0][0][metric] - runs[1][0][metric]) > case['numerics']['safety_margins'][name]:
            raise Unresolved('Safety diagnostic is not numerically resolved: ' + name)
    return max(energies) + discrepancy


def evaluate_seed(arguments):
    """Each process owns one independent native environment at a time."""
    candidate, directory, workdir, seed, baseline = arguments
    controller_module = None
    runs = []
    try:
        controller_module, plant = module('controller'), module('plant')
        for level in ('coarse', 'fine'):
            runs.append(episode(controller_module, plant, candidate, directory,
                                workdir / str(seed) / level, seed, level, baseline))
        return seed, runs, None
    except Exception as error:
        # Dynamically loaded controller exception classes are not pickleable.
        # Preserve failure identity explicitly across the process boundary.
        if controller_module is not None and isinstance(error, controller_module.InvalidController):
            state = 'invalid_submission'
        elif isinstance(error, (ArithmeticError, Unresolved)):
            state = 'candidate_unresolved'
        else:
            state = 'evaluator_error'
        return seed, runs, {'state': state, 'reason': str(error),
                            'traceback': traceback.format_exc()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('submission', type=Path)
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--workdir', type=Path, required=True)
    p.add_argument('--reward-dir', type=Path, required=True)
    p.add_argument('--development', action='store_true')
    p.add_argument('--workers', type=int, choices=range(1, 17), default=4,
                   help='Independent native scenario processes; does not change seeds or physics')
    a = p.parse_args()
    a.reward_dir.mkdir(parents=True, exist_ok=True)
    details = {'version': 'fusionx-t0/v0.1', 'started': time.time(), 'episodes': []}
    reward = {'reward': 0.0, 'valid': 0, 'feasible': 0, 'evaluator_error': 0}
    controller_module = None
    try:
        controller_module = module('controller')
        if a.workdir.exists() and any(a.workdir.iterdir()):
            raise RuntimeError('Workdir must be absent or empty')
        a.workdir.mkdir(parents=True, exist_ok=True)
        candidate = a.workdir / 'controller.onnx'
        if not a.submission.is_file() or a.submission.is_symlink():
            raise controller_module.InvalidController('Submission must be a regular controller.onnx file')
        shutil.copyfile(a.submission, candidate)
        with controller_module.Controller(candidate):
            pass
        case, baseline = release_gate(a.resources, development=a.development)
        details.update(submission_sha256=sha(candidate), case_sha256=sha(a.resources / 'case.json'))
        if a.development:
            seeds = case['scenario_distribution']['development_seeds']
            if baseline['development_seeds'] != seeds:
                raise RuntimeError('Reference energy is not calibrated for this development ensemble')
        else:
            manifest_path = Path(__file__).resolve().parent / 'scenarios.json'
            manifest = read_json(manifest_path)
            if manifest['state'] != 'frozen' or len(manifest['seeds']) != case['scenario_distribution']['scoring_seeds_count']:
                raise RuntimeError('Hidden scenario ensemble is not frozen')
            if baseline['seed_manifest_sha256'] != sha(manifest_path):
                raise RuntimeError('Reference energy is not calibrated for this fixed scoring ensemble')
            seeds = manifest['seeds']
        worst_energy = 0.
        all_v = {}
        jobs = [(candidate, a.resources, a.workdir, seed, baseline) for seed in seeds]
        # spawn avoids inheriting live JAX state. Every scenario has its own
        # controller subprocess, native state, random stream and output directory.
        with ProcessPoolExecutor(max_workers=a.workers,
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            for seed, runs, failure in pool.map(evaluate_seed, jobs):
                if failure is not None:
                    details['native_process_failure'] = {'seed': seed, **failure}
                    if failure['state'] == 'invalid_submission':
                        raise controller_module.InvalidController(failure['reason'])
                    if failure['state'] == 'candidate_unresolved':
                        raise Unresolved(failure['reason'])
                    raise RuntimeError(failure['reason'])
                for level, result in zip(('coarse', 'fine'), runs):
                    details['episodes'].append({'seed': seed, 'level': level,
                                                'metrics': result[0], 'violation': result[1]})
                pair_energy = compare_resolutions(case, baseline, runs)
                for run in runs:
                    for name, value in run[1].items():
                        all_v[name] = max(all_v.get(name, 0.), value)
                worst_energy = max(worst_energy, pair_energy)
        reward.update(reward=score(worst_energy, baseline['J_reference_J'], all_v),
                      valid=1, feasible=int(not all_v))
        details.update(state='evaluated', J_supplied_upper_J=worst_energy, violation=all_v,
                       J_reference_J=baseline['J_reference_J'])
    except ValueError as error:
        if controller_module is not None and isinstance(error, controller_module.InvalidController):
            details.update(state='invalid_submission', reason=str(error))
        else:
            reward['evaluator_error'] = 1
            details.update(state='evaluator_error', reason=str(error), traceback=traceback.format_exc())
    except Unresolved as error:
        details.update(state='candidate_unresolved', reason=str(error))
    except Exception as error:
        reward['evaluator_error'] = 1
        details.update(state='evaluator_error', reason=str(error), traceback=traceback.format_exc())
    details['wall_time_s'] = time.time() - details['started']
    (a.reward_dir / 'details.json').write_text(json.dumps(details, indent=2, allow_nan=False)+'\n')
    (a.reward_dir / 'reward.json').write_text(json.dumps(reward, indent=2, allow_nan=False)+'\n')
    print(json.dumps(reward, allow_nan=False))


if __name__ == '__main__':
    main()
