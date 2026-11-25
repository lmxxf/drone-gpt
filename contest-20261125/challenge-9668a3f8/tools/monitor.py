from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev
ref=json.load(open(ROOT/'workspace/reference/baseline.json'))['development']
for arg in sys.argv[1:]:
 print(arg)
 for p in sorted((ROOT/'runs'/arg).glob('*/*/trace.jsonl')):
  rows=[json.loads(l) for l in p.read_text().splitlines()];r=rows[-1];ts=[x['time_s'] for x in rows];ps=[x['P_fusion_W'] for x in rows]
  target=np.interp(ts,ref['reference_time_s'],ref['reference_fusion_W'])
  de=np.trapezoid(np.array(ps)-target,ts)/1e6 if len(rows)>1 else 0
  print(p.parts[-3],p.parts[-2],f"t={ts[-1]:.1f} P={ps[-1]/1e6:.3f} W={r['W_thermal_total_J']/1e6:.3f} E_diff_sample_MJ={de:.3f}")
