"""Rank actual native full-waveform results; distinguish control-grid and native-grid gaps."""
import sys,json,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev
case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];coarse=[];fine=[]
for directory in ['library','staging','shaping','pulses']:
 for p in (ROOT/'runs'/directory).glob('*.json'):
  if p.stem=='ranking':continue
  r=json.load(open(p))
  if not isinstance(r,dict) or r.get('state')!='native_complete' or r.get('violation'):continue
  if r['metrics']['fusion_energy_J']<ref['required_fusion_energy_J']+1e6:continue
  if r['trace'][-1]['W_thermal_total_J']<.985*ref['reference_terminal_W_thermal_J']:continue
  r['file']=str(p);r['native_file']=str(p.with_suffix('.npy'))
  (coarse if r['level']=='coarse' else fine).append(r)
rows=[]
for c,f in itertools.product(coarse,fine):
 try:J=ev.compare_resolutions(case,ref,[(r['metrics'],r['violation'],r['trace']) for r in [c,f]])
 except ev.Unresolved:continue
 cn=np.load(c['native_file']);fn=np.load(f['native_file'])[1::2];assert np.max(abs(cn[:,0]-fn[:,0]))<1e-8;nd=abs(cn[:,1]-fn[:,1]);mask=cn[:,0]>3
 cd=np.array([r['P_fusion_W'] for r in c['trace']]);fd=np.array([r['P_fusion_W'] for r in f['trace']]);delta=abs(cd-fd)
 rows.append({'coarse_file':c['file'],'fine_file':f['file'],'J_upper_J':J,'min_Q_surplus_J':min(c['metrics']['fusion_energy_J'],f['metrics']['fusion_energy_J'])-ref['required_fusion_energy_J'],'control_grid_max_gap_W':float(delta.max()),'control_grid_after3_max_gap_W':float(delta[7:].max()),'native_grid_after3_max_gap_W':float(nd[mask].max()),'heat_difference_J':abs(c['metrics']['J_supplied_total_J']-f['metrics']['J_supplied_total_J'])})
rows.sort(key=lambda x:x['J_upper_J']);safe=[r for r in rows if r['native_grid_after3_max_gap_W']<3.35e6]
report={'scope':'Offline native full 30 s, one exact extreme corner; not final development/formal evaluation','scenario':{'initial_temperature_multiplier':.98,'event_start_s':8,'Z_eff_after_event':1.75},'feasible_pairs':rows,'native_gap_buffered_pairs':safe};(ROOT/'runs/waveform-ranking.json').write_text(json.dumps(report,indent=2));print(json.dumps({'pairs':len(rows),'native_gap_buffered':len(safe),'best_buffered':safe[:5],'best_official_grid':rows[:5]},indent=2))
