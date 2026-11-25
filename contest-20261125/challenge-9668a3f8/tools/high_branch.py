"""Native calibration of an electron-heating branch with both models at equal total heat."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));sys.path.insert(0,str(ROOT/'tools'))
from library_search import replay_metrics
OUT=ROOT/'runs/high-branch';OUT.mkdir(exist_ok=True)
def commands(level):
    prev=np.array([28.,15.],np.float32);out=[]
    for k in range(60):
        t=k*.5;H=max(45.,53.-3.8*max(t-8.,0));target=np.array([33.,H-33.] if level=='coarse' else [H-20.,20.],np.float32);prev=(prev+np.clip(target-prev,np.float32(-1.999),np.float32(1.999))).astype(np.float32);out.append(prev.copy())
    return np.array(out)
def job(level):
    os.sched_setaffinity(0,{17 if level=='coarse' else 18})
    from fast_native import NativeScan
    import plant
    s=NativeScan({'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98},level);initial=plant.diagnostics(s.env);c=commands(level);d,seconds=s.run(np.vstack(([28.,15.],[28.,15.],c[:59])));m,v,trace=replay_metrics(d,initial,s.dt);r={'level':level,'state':'native_complete','commands_MW':c.tolist(),'metrics':m,'violation':v,'trace':trace,'wall_time_s':seconds};np.save(OUT/(level+'.npy'),d);(OUT/(level+'.json')).write_text(json.dumps(r,indent=2));s.close();return {k:v for k,v in r.items() if k not in ['trace','commands_MW']}
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,l) for l in ['coarse','fine']]):print(json.dumps(f.result()),flush=True)
