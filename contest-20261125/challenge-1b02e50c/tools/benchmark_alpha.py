import os
os.sched_setaffinity(0,{16,17,18,19});os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
import sys,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'workspace'));import magnetic_eval as m
m.N_ALPHA=16
folder=root/'runs/alpha-runtime-benchmark';folder.mkdir(parents=True,exist_ok=True)
wout=root/'runs/proxy_v2/63947299a918c6/wout_eq.nc';bx=json.loads((root/'runs/baseline-neo-screen/result.json').read_text())['boozer']
started=time.time()
try:
 a=m.run_simple(wout,m.alpha_starts(m._wout(wout),2),5.7/bx['B00_axis'],folder,str(root/'tools/bin/simple.x'),4,1800)
 r={'status':'measured','particles':16,'seconds':time.time()-started,'alpha':a,'official_feasible':None}
except Exception as e:r={'status':'failed','reason':str(e),'seconds':time.time()-started}
(folder/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
