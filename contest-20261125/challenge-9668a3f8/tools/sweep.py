"""Native steady response experiments, not scored controller evaluations."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'))
def job(args):
    pe,level,pg=args
    cores=sorted(os.sched_getaffinity(0));idx=(multiprocessing.current_process()._identity[0]-1)%6;chunk=max(1,len(cores)//6);os.sched_setaffinity(0,cores[idx*chunk:(idx+1)*chunk])
    import plant,jax
    jax.config.update('jax_compilation_cache_dir',str(ROOT/'runs/jax-cache'));jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
    env=plant.CoreHeatEnv(ROOT/'workspace',{'event_start_s':0.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98},level,horizon=8.)
    env.reset();power=np.array([28.,15.]);trace=[plant.diagnostics(env)]
    for k in range(16):
        power+=np.clip(np.array([pg,pe])-power,-2,2);env.advance(power);d=plant.diagnostics(env);d['PG'],d['PE']=power.tolist()
        piece=[trace[-1]]+env.last_transition;times=[x['time_s'] for x in piece]
        d['P_fusion_window_mean_W']=float(np.trapezoid([x['P_fusion_W'] for x in piece],times)/(times[-1]-times[0]))
        trace.append(d)
        (ROOT/'runs'/f'response-{level}-G{pg}-E{pe}.json').write_text(json.dumps(trace,indent=2))
    env.close();last=[d for d in trace if d['time_s']>=5]
    return {'PG':pg,'PE':pe,'level':level,'mean_PF_MW':np.mean([d['P_fusion_W']/1e6 for d in last]),'native_mean_PF_MW':np.mean([d['P_fusion_window_mean_W']/1e6 for d in trace if d['time_s']>4]),'mean_W_MJ':np.mean([d['W_thermal_total_J']/1e6 for d in last])}
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mix',action='store_true');p.add_argument('--level',default='coarse');p.add_argument('--pg',type=float,default=33);p.add_argument('--pe',nargs='+',type=float,default=[4,7,10,13,16,20]);a=p.parse_args()
    jobs=[(40-g,a.level,g) for g in [25,28,30,31,32,33]] if a.mix else [(e,a.level,a.pg) for e in a.pe]
    results=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,x) for x in jobs]):
            r=f.result();results.append(r);print(json.dumps(r),flush=True)
            (ROOT/'runs'/f"response-{a.level}-{'mix' if a.mix else str(a.pg)}-summary.json").write_text(json.dumps(results,indent=2))
