"""Native verification of frequency-shaped cooling ramps, not a substitute plant."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));sys.path.insert(0,str(ROOT/'tools'))
from library_search import replay_metrics
OUT=ROOT/'runs/shaping';OUT.mkdir(exist_ok=True)
def cmds(elow,hold,order):
    weights=[.5,.5] if order==1 else [.25,.5,.25];prev=np.array([28.,15.],np.float32);rows=[]
    for k in range(60):
        t=k*.5;e=sum(weight*max(elow,20-3.8*max(t-hold-i,0)) for i,weight in enumerate(weights));e+=3.9*max(t-27,0)
        desired=np.array([33.,min(20,e)],np.float32);prev=(prev+np.clip(desired-prev,np.float32(-1.999),np.float32(1.999))).astype(np.float32);rows.append(prev.copy())
    return np.asarray(rows)
def job(arg):
    elow,hold,order=arg;idx=(multiprocessing.current_process()._identity[0]-1)%6;os.sched_setaffinity(0,{[16,17,18,19,5,6][idx]})
    from fast_native import NativeScan
    import plant
    key=f'fine-e{elow}-h{hold}-order{order}';c=cmds(elow,hold,order);r={'key':key,'level':'fine','elow':elow,'hold':hold,'order':order,'commands_MW':c.tolist()};s=None
    try:
        s=NativeScan({'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98},'fine');initial=plant.diagnostics(s.env);d,seconds=s.run(np.vstack(([28.,15.],[28.,15.],c[:59])));m,v,trace=replay_metrics(d,initial,s.dt);r.update(state='native_complete',metrics=m,violation=v,trace=trace,wall_time_s=seconds);np.save(OUT/(key+'.npy'),d)
    except Exception as e:r.update(state='unresolved',reason=str(e))
    finally:
        if s is not None:s.close()
    (OUT/(key+'.json')).write_text(json.dumps(r,indent=2));return {k:v for k,v in r.items() if k not in ['trace','commands_MW']}
if __name__=='__main__':
    args=[(3.5,6.,1),(3.5,6.,2),(3.8,6.,1),(3.8,6.,2),(3.5,5.5,2),(3.,6.,2)]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,a) for a in args]):print(json.dumps(f.result()),flush=True)
