"""Line search in the measured descent direction; no QS-to-NEO substitution."""
import sys,os,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import solve_turbulence as s
base=json.loads((ROOT/'workspace/reference_configurations/landreman_paul_2022_QA_well.json').read_text())
records=[]
for step in (-.012,-.024,-.04):
 data=json.loads(json.dumps(base))
 for v in data['modes']:
  if v['m']>0:v['rbc']*=math.exp(-step);v['zbs']*=math.exp(step)
 data=s.normalize(data);r=s.evaluate(data);r['elongation_step']=step
 if r.get('screen_pass'):
  folder=ROOT/r['folder'];nfolder=folder/'neo-screen';nfolder.mkdir(exist_ok=True)
  try:
   bx=s.mag.run_boozer(folder/'wout_eq.nc',2,nfolder)
   r['neo']=s.mag.run_neo(bx,nfolder,str(ROOT/'tools/bin/xneo'),7200)
  except Exception as exc:r['neo_error']=f'{type(exc).__name__}: {exc}'
 records.append(r);s.save(ROOT/'results/refinement.json',records)
 if r.get('screen_pass') and r.get('neo',{}).get('eps_eff_max',1)<.01:
  existing=ROOT/'results/refined-best.json'
  if not existing.exists() or r['J_proxy']<json.loads(existing.read_text())['J_proxy']:
   s.save(ROOT/'refined_boundary.json',data);s.save(existing,r)
 print(json.dumps(r),flush=True)
