import os
os.sched_setaffinity(0,{16,17,18,19})
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import sys,json,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'workspace'))
import magnetic_eval as m
folder=root/'runs/baseline-neo-screen';folder.mkdir(parents=True,exist_ok=True)
try:
 b=m.run_boozer(root/'runs/proxy_v2/63947299a918c6/wout_eq.nc',2,folder)
 r={'status':'boozer_measured','boozer':b}
 (folder/'result.json').write_text(json.dumps(r,default=str,indent=2))
 r['neo']=m.run_neo(b,folder,str(root/'tools/bin/xneo'),7200);r['status']='measured'
except Exception as exc:r={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
(folder/'result.json').write_text(json.dumps(r,default=str,indent=2));print(json.dumps(r,default=str))
