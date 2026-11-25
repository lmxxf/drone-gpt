"""Cross-check the compiled native path in a low-heating regime against saved Gym replay."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));sys.path.insert(0,str(ROOT/'tools'));import eval as ev
from fast_native import NativeScan
if __name__=='__main__':
    seed=8516;level='fine';case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];hist=[json.loads(x) for x in (ROOT/'runs/coast-local'/str(seed)/level/'trace.jsonl').read_text().splitlines()];cm=ev.module('controller');noise=np.clip(np.random.Generator(np.random.PCG64(ev.scenario(seed,case)['sensor_seed'])).standard_normal((60,9)),-3,3)*np.array(case['observation']['noise_relative_sigma']);mem=np.zeros(32);prev=np.array([28.,15.]);applied=prev.copy();queued=prev.copy();actions=[]
    with cm.Controller(ROOT/'runs/coast.onnx',worker=True) as ctl:
        for k in range(len(hist)-1):
            target=np.interp(hist[k]['time_s'],ref['reference_time_s'],ref['reference_fusion_W'])/1e6;x=ev.features(hist,k,target,applied,prev,mem,noise,case);a,mem=ctl.step(x);actions.append(a);applied=queued.copy();queued=a.copy();prev=a.copy()
    actions+= [actions[-1]]*(60-len(actions));curve=np.vstack(([28.,15.],[28.,15.],np.asarray(actions)[:59]));horizon=round(hist[-1]['time_s']*2)/2;s=NativeScan(ev.scenario(seed,case),level,horizon);d,seconds=s.run(curve);stride=int(round(.5/s.dt));errors=[]
    for k in range(1,len(hist)):
        errors.append(abs(d[k*stride-1,1]-hist[k]['P_fusion_W'])/hist[k]['P_fusion_W'])
    report={'horizon_s':horizon,'native_steps':len(d),'wall_time_s':seconds,'maximum_relative_fusion_difference':float(max(errors)),'final_relative_fusion_difference':float(errors[-1]),'passed_1e-7':bool(max(errors)<1e-7)};np.save(ROOT/'runs/fast-cold-parity.npy',d);print(json.dumps(report),flush=True);(ROOT/'runs/fast-cold-parity.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);s.close()
