import sys,json,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev
case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];out=[]
for hold,width,depth in itertools.product([6.5,7.],[1.5,2.5],[4.,5.]):
 key=f'h{hold}-w{width}-d{depth}';rs=[]
 for world in ['dev1','dev2','dev3','dev4','lowZ','mid_event','late_event','cold8']:
  c=ROOT/'runs/broad'/(world+'-'+key+'.json');f=ROOT/'runs/pair-pulse-heldout'/(world+'-fine.json') if world!='cold8' else ROOT/'runs/staging/fine-h6.5-p8.0-d0.5.json'
  if not c.exists() or not f.exists():continue
  a=json.load(open(c));b=json.load(open(f));delta=np.abs(np.array([x['P_fusion_W'] for x in a['trace']])-np.array([x['P_fusion_W'] for x in b['trace']]));r={'world':world,'coarse':str(c),'max_gap_W':float(delta.max()),'max_gap_time_s':a['trace'][delta.argmax()]['time_s'],'min_Q_surplus_J':min(x['metrics']['fusion_energy_J'] for x in [a,b])-ref['required_fusion_energy_J'],'violation':[a['violation'],b['violation']]}
  try:r.update(feasible=not any(r['violation']),J_upper_J=ev.compare_resolutions(case,ref,[(x['metrics'],x['violation'],x['trace']) for x in [a,b]]))
  except ev.Unresolved as e:r.update(feasible=False,reason=str(e))
  rs.append(r)
 row={'key':key,'completed':len(rs),'all_feasible':len(rs)==8 and all(x['feasible'] for x in rs),'max_gap_W':max([r['max_gap_W'] for r in rs],default=0),'min_Q_surplus_J':min([r['min_Q_surplus_J'] for r in rs],default=0),'J':max([r.get('J_upper_J',0) for r in rs],default=0),'worlds':rs};out.append(row)
(ROOT/'runs/broad-ranking.json').write_text(json.dumps({'scope':'Offline native screening; cold8 fine uses unblended FF and other fine curves use previous coarse table; final ONNX native replay is required','results':out},indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='worlds'} for r in out],indent=2))
