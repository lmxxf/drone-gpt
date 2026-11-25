"""Actual ONNX feedback, exact noisy/delayed features, native steps batched per control window.
Offline acceleration only. Final artifacts must also pass original evaluator trajectories.
"""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,argparse,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'workspace'))
import eval as ev

def episode(candidate,seed,override,level):
 from fast_native import NativeScan,scan
 from torax._src.orchestration.step_function import SimulationStepFn
 from torax._src.torax_pydantic.interpolated_param_1d import TimeVaryingScalarUpdate
 import jax.numpy as jnp,plant
 case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];physical=ev.scenario(seed,case)
 if override:physical.update(override)
 s=NativeScan(physical,level);initial=plant.diagnostics(s.env);history=[initial];state=s.env.torax_app.current_sim_state;post=s.env.torax_app.current_sim_output;base=s.env.torax_app.step_fn;cm=ev.module('controller');controller=cm.Controller(Path(candidate));memory=np.zeros(32);applied=np.array([28.,15.]);command=applied.copy();queued=applied.copy();noise=np.clip(np.random.Generator(np.random.PCG64(ev.scenario(seed,case)['sensor_seed'])).standard_normal((60,9)),-3,3)*np.array(case['observation']['noise_relative_sigma']);chunks=[];actions=[];t0=time.time()
 try:
  for k in range(60):
   target=float(np.interp(history[-1]['time_s'],ref['reference_time_s'],ref['reference_fusion_W']))/1e6;feature=ev.features(history,k,target,applied,command,memory,noise,case);action,memory=controller.step(feature);assert (action>=[0,0]).all() and (action<=[33,20]).all() and np.max(abs(action-command))<=2+1e-6
   t=float(history[-1]['time_s']);times=jnp.asarray([t,t+.5]);values=jnp.asarray(np.vstack((applied,queued)))*1e6;rp=base.runtime_params_provider.update_provider_from_mapping({'sources.generic_heat.P_total':TimeVaryingScalarUpdate(time=times,value=values[:,0]),'sources.ecrh.P_total':TimeVaryingScalarUpdate(time=times,value=values[:,1]),'numerics.t_final':30.})
   sf=SimulationStepFn(solver=base.solver,time_step_calculator=base.time_step_calculator,runtime_params_provider=rp,geometry_provider=base.geometry_provider)
   (state,post),diag=scan(sf,state,post,jnp.asarray(s.dt),s.indices,s.lo,s.hi,steps=int(round(.5/s.dt)),native=s.native,ped=s.env._native_pedestal_integral);d=np.asarray(diag);assert np.isfinite(d).all() and np.all(d[:,12:14]==0) and np.max(d[:,14])<1e-8;chunks.append(d);last=d[-1];row=initial.copy();row.update(time_s=float(last[0]),P_fusion_W=float(last[1]),W_thermal_total_J=float(last[2]),T_e_axis_keV=float(last[3]),T_i_axis_keV=float(last[4]),beta_N=float(last[5]),q_min=float(last[6]),q95=float(last[7]),P_external_W=float(last[8]),E_external_total_J=float(last[9]),P_pedestal_W=float(last[10]),temperature_max_keV=float(last[11]));history.append(row);applied=queued.copy();queued=action.copy();command=action.copy();actions.append({'time_s':float(feature[0]),'command_MW':action.tolist(),'applied_end_MW':applied.tolist()})
  all_d=np.concatenate(chunks);assert abs(all_d[-1,0]-30)<1e-8
  from library_search import replay_metrics
  m,v,trace=replay_metrics(all_d,initial,s.dt)
  return {'seed':seed,'physical':physical,'level':level,'metrics':m,'violation':v,'trace':trace,'actions':actions,'wall_time_s':time.time()-t0},all_d
 finally:controller.close();s.close()
def job(args):
 name,seed,override,level,candidate,out=args;identity=multiprocessing.current_process()._identity;idx=identity[0]-1 if identity else 0;cpus=[int(x) for x in os.environ.get('DRONE_SCAN_CPUS','5,6,7,8,9,16,17,18,19,10').split(',')];os.sched_setaffinity(0,{cpus[idx%len(cpus)]});r,d=episode(candidate,seed,override,level);r.update(name=name,candidate_sha256=ev.sha(candidate),candidate_path=str(candidate),scope='Actual ONNX block-native offline validation; original replay required');np.save(Path(out)/(name+'-'+level+'.npy'),d);(Path(out)/(name+'-'+level+'.json')).write_text(json.dumps(r,indent=2));return {k:v for k,v in r.items() if k not in ['trace','actions']}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('out');p.add_argument('--parity',action='store_true');p.add_argument('--stress-count',type=int,default=0);p.add_argument('--stress-seed',type=int,default=966820261125);p.add_argument('--include-seed',type=int,action='append',default=[]);p.add_argument('--workers',type=int);p.add_argument('--names',nargs='+');a=p.parse_args();out=Path(a.out).resolve();out.mkdir(exist_ok=False)
 if a.parity:worlds=[('dev3',2026100803,None)];levels=['fine'];workers=1
 elif a.stress_count:
  seeds=np.random.Generator(np.random.PCG64(a.stress_seed)).integers(1,2**31,size=a.stress_count);assert len(set(map(int,seeds)))==len(seeds);worlds=[(f'stress{i+1}',int(seed),None) for i,seed in enumerate(seeds)];levels=['fine','coarse'];workers=a.workers or 7
 else:worlds=[(f'dev{i+1}',s,None) for i,s in enumerate([2026100801,2026100802,2026100803,2026100804])]+[('cold8',17,{'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98}),('cold12',17,{'event_start_s':12.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98}),('cold16',991,{'event_start_s':16.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98})];levels=['fine','coarse'];workers=7
 if a.include_seed:worlds += [(f'extra{i+1}',seed,None) for i,seed in enumerate(a.include_seed)]
 assert len({w[1] for w in worlds})==len(worlds) or not a.stress_count
 if a.names:
  worlds=[w for w in worlds if w[0] in a.names];assert len(worlds)==len(a.names)
 if a.workers:workers=a.workers
 with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(job,(n,s,o,l,str(Path(a.candidate).resolve()),str(out))) for l in levels for n,s,o in worlds]):print(json.dumps(f.result()),flush=True)
 if a.parity:
  r=json.load(open(out/'dev3-fine.json'));old=ev.read_json(ROOT/'runs/robust-local/2026100803/fine/trace.json');oldacts=ev.read_json(ROOT/'runs/robust-local/2026100803/fine/actions.json');err=max(abs(x['P_fusion_W']-y['P_fusion_W'])/y['P_fusion_W'] for x,y in zip(r['trace'],old));ae=float(np.max(abs(np.array([x['command_MW'] for x in r['actions']])-np.array([x['command_MW'] for x in oldacts]))));result={'maximum_relative_fusion_difference':err,'maximum_command_difference_MW':ae,'passed':err<1e-7 and ae<1e-5};(out/'parity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);assert result['passed']
 else:
  case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];rs=[]
  for name,_,_ in worlds:
   rows=[json.load(open(out/(name+'-'+l+'.json'))) for l in ['coarse','fine']];r={'name':name,'violation':[x['violation'] for x in rows]}
   try:r.update(J=ev.compare_resolutions(case,ref,[(x['metrics'],x['violation'],x['trace']) for x in rows]),feasible=not any(r['violation']))
   except ev.Unresolved as e:r.update(feasible=False,reason=str(e))
   rs.append(r)
  (out/'summary.json').write_text(json.dumps({'scope':'Actual ONNX block-native screen, original full replay required','feasible':all(x['feasible'] for x in rs),'stress_seed_generator':a.stress_seed,'stress_count':a.stress_count,'extra_seeds':a.include_seed,'pairs':rs},indent=2));print(json.dumps(rs),flush=True)
