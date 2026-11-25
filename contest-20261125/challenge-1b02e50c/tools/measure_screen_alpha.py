import os
os.sched_setaffinity(0,{16,17,18,19});os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
import sys,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'workspace'))
import magnetic_eval as m
folder=root/'runs/baseline-alpha-screen';folder.mkdir(parents=True,exist_ok=True)
wout=root/'runs/proxy_v2/63947299a918c6/wout_eq.nc'
try:
 w=m._wout(wout);bx=json.loads((root/'runs/baseline-neo-screen/result.json').read_text())['boozer']
 r={'status':'running','radial_surfaces':int(w['ns']),'estimated_memory_gib':m.simple_memory_gib(w),'official_feasible':None}
 (folder/'result.json').write_text(json.dumps(r,indent=2))
 a=m.run_simple(wout,m.alpha_starts(w,2),5.7/bx['B00_axis'],folder,str(root/'tools/bin/simple.x'),4,43200)
 a['wilson_95']=[m.wilson(a['N_L'],4096)[0],m.wilson(a['N_L']+a['N_U'],4096)[1]]
 r.update(status='measured',alpha=a)
except Exception as exc:r={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
(folder/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
