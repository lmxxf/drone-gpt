"""Aggregate completed native episodes without altering evaluator thresholds."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev
case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development']
run=Path(sys.argv[1]);r=json.load(open(run/'report.json'));episodes=r['episodes'];result={'runtime':r['runtime'],'release_gate_passed':False,'candidate_sha256':r['candidate_sha256'],'groups':{}}
for name,seeds in [('development',case['scenario_distribution']['development_seeds']),('stress',[8516,7487,20261125])]:
 group=[x for x in episodes if x['seed'] in seeds]
 if not group:continue
 v={};J=0;checks=[]
 for seed in sorted({x['seed'] for x in group}):
  es=sorted([x for x in group if x['seed']==seed],key=lambda x:x['level']);assert [x['level'] for x in es]==['coarse','fine']
  traces=[ev.read_json(run/str(seed)/x['level']/'trace.json') for x in es]
  runs=[(e['metrics'],e['violation'],t) for e,t in zip(es,traces)]
  obj=ev.compare_resolutions(case,ref,runs);J=max(J,obj)
  for e in es:
   for k,x in e['violation'].items():v[k]=max(v.get(k,0),x)
  e0,e1=[e['metrics']['J_supplied_total_J'] for e in es]
  pd=float(np.max(abs(np.array([x['P_fusion_W'] for x in traces[0]])-np.array([x['P_fusion_W'] for x in traces[1]]))))
  checks.append({'seed':seed,'J_upper_J':obj,'grid_energy_difference_J':abs(e0-e1),'grid_energy_difference_fraction_Jref':abs(e0-e1)/ref['J_reference_J'],'grid_power_difference_W':pd,'grid_power_difference_fraction_reference_peak':pd/max(ref['reference_fusion_W'])})
 result['groups'][name]={'feasible':not v,'local_score':ev.score(J,ref['J_reference_J'],v),'J_upper_J':J,'J_reference_J':ref['J_reference_J'],'heat_saving_fraction':1-J/ref['J_reference_J'],'violation':v,'min_Q_ratio':min(e['metrics']['fusion_energy_J']/ref['required_fusion_energy_J'] for e in group),'max_rms':max(e['metrics']['fusion_rms_relative'] for e in group),'max_peak':max(e['metrics']['fusion_peak_relative'] for e in group),'min_terminal_W_ratio':min(1-e['metrics']['terminal_energy_deficit_fraction'] for e in group),'max_beta_N':max(e['metrics']['beta_N_max'] for e in group),'max_temperature_keV':max(e['metrics']['temperature_max_keV'] for e in group),'max_positive_pedestal_J':max(e['metrics']['E_pedestal_positive_J'] for e in group),'grid_checks':checks,'episodes':group}
print(json.dumps(result,indent=2));(run/'summary.json').write_text(json.dumps(result,indent=2))
