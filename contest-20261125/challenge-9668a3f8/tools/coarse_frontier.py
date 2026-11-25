"""Cold-corner native search for more efficient coarse-model startup/base heating.
The old fine trace is a ranking proxy only; every selected graph needs fresh paired validation.
"""
import os,sys,json,multiprocessing,subprocess,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));from block_eval import job
if __name__=='__main__':
 out=ROOT/'runs/coarse-frontier';out.mkdir(exist_ok=False);base=json.load(open(ROOT/'runs/early_reserve-parameters.json'));tasks=[]
 for ebase,hold in itertools.product([5.25,5.75,6.25],[3.5,4.5,5.5]):
  tag=f'E{ebase}-H{hold}';prev=np.array([28.,15.],np.float32);cs=[]
  for k in range(60):
   t=k*.5;dip=max(0,4.-3.8*max(abs(t-10.5)-1.25,0));pg=33.-dip-np.clip(t-27.,0,2);pe=max(ebase,20.-3.8*max(t-hold,0))+2.2*max(t-27.,0);target=np.array([pg,min(20.,pe)],np.float32);prev=(prev+np.clip(target-prev,np.float32(-1.999),np.float32(1.999))).astype(np.float32);cs.append(prev.copy())
  cfg=dict(base);cfg.update(coarse_commands_MW=np.array(cs).tolist(),coarse_base_E_MW=ebase,coarse_warm_hold_s=hold);params=out/(tag+'-parameters.json');params.write_text(json.dumps(cfg,indent=2));candidate=out/(tag+'.onnx');subprocess.run([sys.executable,str(ROOT/'tools/build_controller.py'),str(params),str(candidate)],check=True)
  with (out/(tag+'-original.log')).open('w') as log:subprocess.run([sys.executable,'-I',str(ROOT/'workspace/eval.py'),str(candidate),'--resources',str(ROOT/'workspace'),'--development','--workdir',str(out/(tag+'-original')),'--reward-dir',str(out/(tag+'-original-reward'))],stdout=log,stderr=subprocess.STDOUT,check=True)
  tasks.append((tag,17,{'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98},'coarse',str(candidate),str(out)))
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(job,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
