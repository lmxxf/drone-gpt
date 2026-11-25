import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev
p=Path(sys.argv[1]);candidate=Path(sys.argv[2]);seed=int(p.parts[-3]);case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];history=[json.loads(l) for l in p.read_text().splitlines()];cm=ev.module('controller');noise=np.clip(np.random.Generator(np.random.PCG64(ev.scenario(seed,case)['sensor_seed'])).standard_normal((60,9)),-3,3)*np.array(case['observation']['noise_relative_sigma']);mem=np.zeros(32);applied=np.array([28.,15.]);command=applied.copy();queued=applied.copy()
with cm.Controller(candidate,worker=True) as ctl:
 for k in range(len(history)-1):
  target=np.interp(history[k]['time_s'],ref['reference_time_s'],ref['reference_fusion_W'])/1e6;x=ev.features(history,k,target,applied,command,mem,noise,case);action,mem=ctl.step(x);applied=queued.copy();queued=action.copy();command=action.copy()
  if k%4==0 or k==len(history)-2:print(json.dumps({'t':x[0],'PG':action[0],'PE':action[1],'Ef_MJ':100*mem[0],'Er_MJ':100*mem[1],'error_filtered_MW':10*mem[5],'integral_heat_MW':10*mem[7],'W_error_MJ':10*mem[6],'W_heat_MW':10*mem[11]}))
