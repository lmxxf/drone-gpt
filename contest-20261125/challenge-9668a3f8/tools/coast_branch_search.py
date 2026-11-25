"""Test low ECRH coasting and a full terminal heat reserve on the exact cold corner."""
import os,sys,json,multiprocessing,subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));from block_eval import job
if __name__=='__main__':
 out=ROOT/'runs/coast-branch';out.mkdir(exist_ok=False);base=json.load(open(ROOT/'runs/robust-parameters.json'));tasks=[]
 for floor in [.5,1.5,2.5]:
  tag=f'f{floor}';prev=np.array([28.,15.],np.float32);cs=[]
  for k in range(60):
   t=k*.5;a=max(0,t-6.5);r=(20.-8.)/3.8
   if a<=r:e=20.-3.8*a
   elif a<=r+.5:e=8.
   else:e=max(floor,8.-3.8*(a-r-.5))
   e+=3.9*max(t-24.5,0);target=np.array([33.,min(20.,e)],np.float32);prev=(prev+np.clip(target-prev,np.float32(-1.999),np.float32(1.999))).astype(np.float32);cs.append(prev.copy())
  cfg=dict(base);cfg.update(fine_commands_MW=np.array(cs).tolist(),fine_coast_floor_MW=floor,fine_terminal_start_s=24.5);param=out/(tag+'-parameters.json');param.write_text(json.dumps(cfg,indent=2));candidate=out/(tag+'.onnx');subprocess.run([sys.executable,str(ROOT/'tools/build_controller.py'),str(param),str(candidate)],check=True)
  with (out/(tag+'-original.log')).open('w') as log:subprocess.run([sys.executable,'-I',str(ROOT/'workspace/eval.py'),str(candidate),'--resources',str(ROOT/'workspace'),'--development','--workdir',str(out/(tag+'-original')),'--reward-dir',str(out/(tag+'-original-reward'))],stdout=log,stderr=subprocess.STDOUT,check=True)
  tasks.append((tag,17,{'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98},'fine',str(candidate),str(out)))
 with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(job,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
