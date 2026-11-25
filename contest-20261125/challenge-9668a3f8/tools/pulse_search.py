"""Native coarse-model transient alignment using a bounded generic-heating dip."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));sys.path.insert(0,str(ROOT/'tools'));from library_search import replay_metrics
OUT=ROOT/'runs/pulses';OUT.mkdir(exist_ok=True)
def cmds(hold,depth):
    prev=np.array([28.,15.],np.float32);rows=[]
    for k in range(60):
        t=k*.5;drop=max(0.,depth-3.8*max(abs(t-10.)-.25,0.));g=33.-drop;e=max(5.,20.-3.8*max(t-hold,0.))+3.9*max(t-27.,0.);target=np.array([g,min(20.,e)],np.float32);prev=(prev+np.clip(target-prev,np.float32(-1.999),np.float32(1.999))).astype(np.float32);rows.append(prev.copy())
    return np.array(rows)
def job(arg):
    hold,depth=arg;idx=(multiprocessing.current_process()._identity[0]-1)%9;os.sched_setaffinity(0,{[0,1,2,3,4,10,11,12,13][idx]})
    from fast_native import NativeScan
    import plant
    key=f'coarse-h{hold}-dip{depth}';c=cmds(hold,depth);r={'key':key,'level':'coarse','hold':hold,'generic_dip_MW':depth,'commands_MW':c.tolist()};s=None
    try:
        s=NativeScan({'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98},'coarse');initial=plant.diagnostics(s.env);d,seconds=s.run(np.vstack(([28.,15.],[28.,15.],c[:59])));m,v,tr=replay_metrics(d,initial,s.dt);r.update(state='native_complete',metrics=m,violation=v,trace=tr,wall_time_s=seconds);np.save(OUT/(key+'.npy'),d)
    except Exception as e:r.update(state='unresolved',reason=str(e))
    finally:
        if s is not None:s.close()
    (OUT/(key+'.json')).write_text(json.dumps(r,indent=2));return {k:v for k,v in r.items() if k not in ['trace','commands_MW']}
if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=9,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,(h,d)) for h in [5.5,6.,6.5] for d in [3.,4.,5.]]):print(json.dumps(f.result()),flush=True)
