#!/usr/bin/env python3
"""Native interface measurement; never generates competition scores.

Calls the same maintained A magnetic and GX interfaces as the verifier. A probe
result is measured evidence, not a calibrated baseline or a passed release gate.
The generated public background includes this entry point for physics inspection.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def run(args):
    root = args.trusted_dir.resolve()
    spec = importlib.util.spec_from_file_location('d0_eval', root / 'eval.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    case = ev.read_json(root / 'case.json')
    work = args.workdir.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError('probe workdir must be empty')
    work.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record = {'schema': 'fusionx-d0-native-probe/v1', 'mode': args.mode,
              'status': 'running', 'official_verifier_passed': False,
              'case_sha256': ev.digest(root / 'case.json'),
              'probe_source_sha256': ev.digest(Path(__file__)),
              'evaluator_source_sha256': ev.digest(root / 'eval.py')}
    try:
        if args.mode in ('equilibrium', 'magnetic'):
            if args.submission is None:
                raise ValueError('--submission is required for magnetic measurements')
            ev.submission_snapshot(args.submission, work/'submission.json', case)
            mod = ev.load_magnetics(root, case)
            if args.mode == 'magnetic':
                physics = work/'magnetic'
                physics.mkdir()
                metrics, details, wout, nfp = ev.magnetic_stage(
                    work/'submission.json', physics, case, mod, args.threads,
                    started + case['magnetic']['magnetic_budget_sec'])
                record.update(status='measured', metrics=metrics, magnetic=details,
                              violations=ev.violations(metrics, case['constraints']),
                              wout=str(wout), wout_sha256=ev.digest(wout), nfp=nfp)
                return
            b = mod.read_boundary(work / 'submission.json')
            initial = mod.geometry(b)
            b = b.scaled(case['fixed']['minor_radius_m'] / initial['minor_radius'])
            geometry = mod.geometry(b)
            program = shutil.which('xvmec2000')
            if program is None:
                raise ev.EvaluatorError('native VMEC2000 is missing')
            result = mod.run_vmec(b, work, program, case['magnetic']['vmec_budget_sec'])
            boozer = mod.run_boozer(result['wout'], b.nfp, work)
            record.update(status='measured', wout=str(result['wout']),
                          wout_sha256=ev.digest(result['wout']), geometry=geometry,
                          profile=mod.profile_metrics(result['wout_data']),
                          vmec={k:result[k] for k in ('attempt','ns','seconds','residuals','resolution','failed_attempts')},
                          boozer={k:boozer[k] for k in ('B00_axis','e_B','history','s_neo')},
                          field_scale=case['fixed']['axis_B00_T']/boozer['B00_axis'])
        else:
            if args.equilibrium_result is None:
                raise ValueError('--equilibrium-result is required for gx')
            equilibrium = ev.read_json(args.equilibrium_result)
            wout = Path(equilibrium['wout'])
            if equilibrium['case_sha256'] != record['case_sha256'] or ev.digest(wout) != equilibrium['wout_sha256']:
                raise ValueError('equilibrium identity does not match this case or output')
            point = next(p for p in case['points'] if p['id'] == args.point).copy()
            level = next(p for p in case['gx']['levels'] if p['id'] == args.level).copy()
            if args.perpendicular_resolution is not None:
                level.update(id=f'calibration-{args.perpendicular_resolution}',
                             nx=args.perpendicular_resolution, ny=args.perpendicular_resolution)
            budget = (args.native_budget_sec if args.native_budget_sec is not None
                      else case['gx']['per_run_budget_sec'])
            record.update(native_binary_sha256=ev.digest(Path(case['gx']['executable'])),
                          native_execution_timeout_sec=budget,
                          calibration_perpendicular_resolution=args.perpendicular_resolution)
            geometry = ev.geometry_stage(wout, work/'eik.nc', point, case)
            field_scale = equilibrium['field_scale']
            if args.zero_drive:
                point['a_over_L_Ti'] = point['a_over_L_n'] = 0.0
                case['gx'].update(init_amp=0.0, tau_end=20.0, diagnostic_padding_tau=1.0)
            seed = case['gx']['seeds'][args.seed_index]
            (work/'input.in').write_text(ev.gx_input(point, level, seed, geometry, field_scale, case))
            with (work/'gx.log').open('wb') as log:
                job = subprocess.run([case['gx']['executable'],'input.in'],cwd=work,
                                     stdout=log,stderr=subprocess.STDOUT,
                                     timeout=budget,check=False)
            if job.returncode != 0:
                raise ev.Unresolved(f'native GX exit {job.returncode}; see gx.log')
            times, flux, normalization = ev.read_flux(work/'input.out.nc',geometry,field_scale,case)
            np.savetxt(work/'heat-flux-common.csv',np.column_stack((times,flux)),
                       delimiter=',',header='tau,heat_flux_common',comments='')
            record.update(status='measured', point=point, level=level, seed=seed,
                          zero_drive=args.zero_drive, geometry=geometry,
                          normalization=normalization, input_sha256=ev.digest(work/'input.in'),
                          samples=len(times), final_tau=float(times[-1]),
                          raw_flux_min=float(flux.min()), raw_flux_max=float(flux.max()))
            if args.zero_drive:
                record['zero_flux_exact'] = bool(np.all(flux == 0.0))
            else:
                try:
                    blocks, corr = ev.block_means(times,flux,case)
                    record.update(block_means=blocks.tolist(), correlation_time=corr,
                                  fixed_window_mean=float(blocks.mean()))
                except ev.Unresolved as exc:
                    record['statistics_status'] = 'unresolved'
                    record['statistics_reason'] = str(exc)
    except Exception as exc:
        record.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['seconds'] = time.time()-started
        (work/'probe.json').write_text(json.dumps(record,indent=2,allow_nan=False,default=str)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('equilibrium','magnetic','gx'))
    parser.add_argument('--trusted-dir',type=Path,required=True)
    parser.add_argument('--workdir',type=Path,required=True)
    parser.add_argument('--submission',type=Path)
    parser.add_argument('--threads',type=int,default=64)
    parser.add_argument('--equilibrium-result',type=Path)
    parser.add_argument('--point',default='T00-g3')
    parser.add_argument('--level',default='fine')
    parser.add_argument('--seed-index',type=int,choices=(0,1),default=0)
    parser.add_argument('--zero-drive',action='store_true')
    parser.add_argument('--perpendicular-resolution',type=int,choices=(32,64,96,128),
                        help='author-only convergence probe; does not change the case or formal verifier')
    parser.add_argument('--native-budget-sec',type=int,
                        help='author-only measured runtime limit; never generates a ranked result')
    args = parser.parse_args()
    if args.mode != 'gx' and (args.perpendicular_resolution is not None or args.native_budget_sec is not None):
        parser.error('GX convergence options require mode gx')
    if args.native_budget_sec is not None and args.native_budget_sec <= 0:
        parser.error('--native-budget-sec must be positive')
    run(args)
