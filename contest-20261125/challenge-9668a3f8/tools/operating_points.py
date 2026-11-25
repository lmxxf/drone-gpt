"""Global native operating-point search after the same high-power startup."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev

def job(target):
    cpus=[int(x) for x in os.environ.get('DRONE_POINT_CPUS','5,6,7,8,9,15,16,17,18,19').split(',')];idx=(multiprocessing.current_process()._identity[0]-1)%len(cpus);os.sched_setaffinity(0,{cpus[idx]})
    import plant,jax
    jax.config.update('jax_compilation_cache_dir',str(ROOT/'runs/jax-cache'));jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
    pg,pe=target;trace=[];accepted=[];level=os.environ.get('DRONE_POINT_LEVEL','coarse');out=ROOT/'runs'/(f'frontier-{level}-G{pg}-E{pe}.json' if pe in [3,3.5,3.8,4.2,4.5] else f'operating-{level}-G{pg}-E{pe}.json')
    env=plant.CoreHeatEnv(ROOT/'workspace',{'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98},level,horizon=18.)
    try:
        env.reset();trace=[plant.diagnostics(env)];accepted=trace.copy();command=np.array([28.,15.]);queued=command.copy()
        for k in range(36):
            desired=np.array([33.,20.]) if k*.5<6 else np.array(target,float)
            action=command+np.clip(desired-command,-1.999,1.999);env.advance(queued);accepted.extend(env.last_transition);trace.append(plant.diagnostics(env));command=action;queued=action.copy()
            out.write_text(json.dumps({'target_MW':target,'trace':trace},indent=2))
        t=np.array([r['time_s'] for r in accepted]);pf=np.array([r['P_fusion_W'] for r in accepted]);w=np.array([r['W_thermal_total_J'] for r in accepted]);mask=t>=16
        result={'target_MW':target,'level':level,'state':'native_complete','native_PF_last2_MW':float(np.trapezoid(pf[mask],t[mask])/(t[mask][-1]-t[mask][0])/1e6),'final_W_MJ':w[-1]/1e6,'native_Q_GJ':float(np.trapezoid(pf,t)/1e9)}
        result.update(ev.supplied_heat_metrics(accepted,trace[-1]['E_external_total_J']))
        out.write_text(json.dumps({'result':result,'trace':trace},indent=2));return result
    except Exception as error:
        result={'target_MW':target,'level':level,'state':'unresolved','reason':str(error),'last_time_s':trace[-1]['time_s'] if trace else None};out.write_text(json.dumps({'result':result,'trace':trace},indent=2));return result
    finally:env.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--points');p.add_argument('--workers',type=int,default=10);a=p.parse_args()
    points=[(g,20) for g in [0,10,20,28,33]]+[(g,0) for g in [15,20,25,30,33]]+[(g,10) for g in [10,20,25,28,33]]
    if a.points:points=[tuple(x) for x in json.load(open(a.points))]
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,p) for p in points]):
            r=f.result();rows.append(r);print(json.dumps(r),flush=True);(ROOT/'runs'/('operating-'+os.environ.get('DRONE_POINT_LEVEL','coarse')+'-'+(Path(a.points).stem if a.points else 'global')+'-summary.json')).write_text(json.dumps(rows,indent=2))
