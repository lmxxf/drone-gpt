"""Coarse native hot-ion operating-point tests with a strictly positive ECRH floor."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));sys.path.insert(0,str(ROOT/'tools'));from library_search import replay_metrics
OUT=ROOT/'runs/low-branch';OUT.mkdir(exist_ok=True)
def job(arg):
    pg,boot=arg;idx=(multiprocessing.current_process()._identity[0]-1)%9;os.sched_setaffinity(0,{[5,6,7,8,9,15,16,17,18][idx]})
    from fast_native import NativeScan
    import plant
    key=f'G{pg}-boot{boot}';prev=np.array([28.,15.],np.float32);cmd=[]
    for k in range(60):
        target=np.array([33.,20.] if k*.5<boot else [pg,.1],np.float32);prev=(prev+np.clip(target-prev,np.float32(-1.999),np.float32(1.999))).astype(np.float32);cmd.append(prev.copy())
    r={'key':key,'target_MW':[pg,.1],'bootstrap_s':boot,'commands_MW':np.array(cmd).tolist()};s=None
    try:
        s=NativeScan({'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98},'coarse');initial=plant.diagnostics(s.env);d,seconds=s.run(np.vstack(([28.,15.],[28.,15.],np.array(cmd)[:59])));m,v,tr=replay_metrics(d,initial,s.dt);mask=d[:,0]>25;pf=float(np.trapezoid(d[mask,1],d[mask,0])/(d[mask][-1,0]-d[mask][0,0])/1e6);r.update(state='native_complete',metrics=m,violation=v,trace=tr,PF_last5_MW=pf,W_final_MJ=float(d[-1,2]/1e6),wall_time_s=seconds);np.save(OUT/(key+'.npy'),d)
    except Exception as e:r.update(state='unresolved',reason=str(e))
    finally:
        if s is not None:s.close()
    (OUT/(key+'.json')).write_text(json.dumps(r,indent=2));return {k:v for k,v in r.items() if k not in ['trace','commands_MW']}
if __name__=='__main__':
    args=[(g,0.) for g in [10.,15.,20.,25.,30.,33.]]+[(g,6.) for g in [20.,25.,33.]]
    with ProcessPoolExecutor(max_workers=9,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,a) for a in args]):print(json.dumps(f.result()),flush=True)
