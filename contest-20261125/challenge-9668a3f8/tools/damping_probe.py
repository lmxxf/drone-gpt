"""Causal AC-feedback prototypes; native physics and domain checks unchanged."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev

def job(gain):
    idx=(multiprocessing.current_process()._identity[0]-1)%4;os.sched_setaffinity(0,{[5,6,7,8][idx]})
    import plant,jax
    jax.config.update('jax_compilation_cache_dir',str(ROOT/'runs/jax-cache'));jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
    cm=ev.module('controller');case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];seed=8516;results=[]
    for level in ['coarse','fine']:
        env=plant.CoreHeatEnv(ROOT/'workspace',ev.scenario(seed,case),level,horizon=16.)
        env.reset();history=[plant.diagnostics(env)];accepted=history.copy();mem=np.zeros(32);command=np.array([28.,15.]);queued=command.copy();applied=command.copy();actions=[]
        noise=np.clip(np.random.Generator(np.random.PCG64(ev.scenario(seed,case)['sensor_seed'])).standard_normal((32,9)),-3,3)*np.array(case['observation']['noise_relative_sigma']);slow=fast=balance=0.
        with cm.Controller(ROOT/'runs/pulse.onnx',worker=True) as ctl:
            for k in range(32):
                target=np.interp(history[k]['time_s'],ref['reference_time_s'],ref['reference_fusion_W'])/1e6;x=ev.features(history,k,target,applied,command,mem,noise,case);action,mem=ctl.step(x)
                if k<=4:slow=fast=x[2]
                else:slow=.9*slow+.1*x[2];fast=.4*fast+.6*x[2]
                base_pe=float(np.clip(mem[8]*10-33,0,20)) if k*.5>=8 else action[1]
                ac=gain*(fast-slow)-balance/3 if k*.5>=8 else 0.
                action[1]=np.clip(base_pe+ac,max(0,command[1]-2),min(20,command[1]+2))
                if k*.5>=8:balance+=.5*(action[1]-base_pe)
                env.advance(queued);accepted.extend(env.last_transition);history.append(plant.diagnostics(env));applied=queued.copy();queued=action.copy();command=action.copy();actions.append({'time':k*.5,'PG':action[0],'PE':action[1],'ac':ac,'AC_budget_MJ':balance})
                (ROOT/'runs'/f'damping-{gain}-{level}.json').write_text(json.dumps({'trace':history,'actions':actions},indent=2))
        times=np.array([r['time_s'] for r in accepted]);pf=np.array([r['P_fusion_W'] for r in accepted]);mask=times>=12
        results.append({'trace':history,'energy_J':history[-1]['E_external_total_J'],'native_PF_MW':float(np.trapezoid(pf[mask],times[mask])/(times[mask][-1]-times[mask][0])/1e6),'max_T':max(r['temperature_max_keV'] for r in accepted)})
        env.close()
    c,f=results;gap=max(abs(a['P_fusion_W']-b['P_fusion_W']) for a,b in zip(c['trace'],f['trace']))
    return {'gain':gain,'max_grid_gap_MW':gap/1e6,'heat_difference_MJ':abs(c['energy_J']-f['energy_J'])/1e6,'coarse_native_PF_MW':c['native_PF_MW'],'fine_native_PF_MW':f['native_PF_MW']}
if __name__=='__main__':
    out=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,g) for g in [-2.,-1.,1.,2.]]):
            r=f.result();out.append(r);print(json.dumps(r),flush=True);(ROOT/'runs/damping-summary.json').write_text(json.dumps(out,indent=2))
