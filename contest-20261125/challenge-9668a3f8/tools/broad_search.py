"""Native coarse-model broad thermal troughs; fixed held-out fine trajectories."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'workspace'))
from library_search import replay_metrics
import eval as ev
OUT=ROOT/'runs/broad';OUT.mkdir(exist_ok=True)
def commands(hold,width,depth):
 prev=np.array([28.,15.],np.float32);rows=[]
 for k in range(60):
  t=k*.5;drop=max(0,depth-3.8*max(abs(t-10.5)-width/2,0));late=np.clip((t-27)*.5,0,1);e=max(5.,20.-3.8*max(t-hold,0))+3.9*max(t-27,0);target=np.array([33.-drop-late,min(e,20)],np.float32);prev=(prev+np.clip(target-prev,np.float32(-1.999),np.float32(1.999))).astype(np.float32);rows.append(prev.copy())
 return np.array(rows)
def job(world):
 name,seed,override=world;idx=(multiprocessing.current_process()._identity[0]-1)%7 if multiprocessing.current_process()._identity else 0;cpus=[int(x) for x in os.environ.get('DRONE_SCAN_CPUS','5,6,7,8,9,15,16').split(',')];os.sched_setaffinity(0,{cpus[idx%len(cpus)]})
 from fast_native import NativeScan
 import plant
 case=ev.read_json(ROOT/'workspace/case.json');physical=ev.scenario(seed,case)
 if override:physical.update(override)
 s=NativeScan(physical,'coarse');initial=plant.diagnostics(s.env);fine=np.array(json.load(open(ROOT/'runs/staging/fine-h6.5-p8.0-d0.5.json'))['commands_MW'],np.float32)
 noise=np.clip(np.random.Generator(np.random.PCG64(physical['sensor_seed'])).standard_normal((60,9)),-3,3)*np.array(case['observation']['noise_relative_sigma']);qmid=np.float32(1.8308736068151363);weights=[];qsum=np.float32(0);count=np.float32(0)
 for k in range(60):
  if k<15:qsum=np.float32(qsum+np.float32(np.float32(np.float32(initial['q_min']*(1+noise[k,5]))-qmid)*100));count=np.float32(count+1)
  weights.append(np.float32(1/(1+np.exp(np.float32(-40)*np.float32(qsum/count)))))
 results=[]
 for hold,width,depth in itertools.product([6.5,7.],[1.5,2.5],[4.,5.]):
  key=f'h{hold}-w{width}-d{depth}';c=commands(hold,width,depth);prev=np.array([28.,15.],np.float32);cs=[]
  for k in range(60):
   target=(fine[k]+weights[k]*(c[k]-fine[k])).astype(np.float32);prev=(prev+np.clip(target-prev,np.float32(-1.999),np.float32(1.999))).astype(np.float32);cs.append(prev.copy())
  d,seconds=s.run(np.vstack(([28.,15.],[28.,15.],np.array(cs)[:59])));m,v,trace=replay_metrics(d,initial,s.dt);r={'world':name,'seed':seed,'physical':physical,'key':key,'level':'coarse','commands_MW':c.tolist(),'state':'native_complete','metrics':m,'violation':v,'trace':trace,'wall_time_s':seconds};np.save(OUT/(name+'-'+key+'.npy'),d);(OUT/(name+'-'+key+'.json')).write_text(json.dumps(r,indent=2));print(json.dumps({'world':name,'key':key,'Q_surplus_MJ':(m['fusion_energy_J']-3340322557.573245)/1e6,'E_MJ':m['J_supplied_total_J']/1e6,'violation':v}),flush=True);results.append(r)
 s.close();return name
if __name__=='__main__':
 worlds=[(f'dev{i+1}',s,None) for i,s in enumerate([2026100801,2026100802,2026100803,2026100804])]+[('lowZ',20261125,None),('mid_event',17,{'event_start_s':12.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98}),('late_event',991,{'event_start_s':16.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98})]
 with ProcessPoolExecutor(max_workers=7,mp_context=multiprocessing.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(job,w) for w in worlds]):print('completed',f.result(),flush=True)
