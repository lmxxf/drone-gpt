#!/usr/bin/env python3
"""Summarize actual evaluator JSON; never infer feasibility from screens."""
import argparse,json
from pathlib import Path

def summarize(path):
 r=json.loads(Path(path).read_text())
 out={k:v for k,v in r.items() if k!='cases'};out['cases']={}
 for scenario,levels in r.get('cases',{}).items():
  for level,c in levels.items():
   modes=c['stability']['modes'];allm=list(modes.values())
   for phase in c.get('phases',{}).values():allm+=list(phase['modes'].values())
   out['cases'][scenario+'/'+level]={
    'average_pressure_Pa':c['equilibrium']['average_pressure_Pa'],
    'q_min':min(m['globals']['qmin'] for m in allm),
    'minimum_energy_by_n':{n:m['minimum_energy'][0] for n,m in modes.items()},
    'minimum_energy_all_phases':min(m['minimum_energy'][0] for m in allm),
    'mercier_max':max(m['mercier_di_max'] for m in allm),
    'ballooning_nodes_max':max(m['ballooning_nodes_max'] for m in allm),
    'internal_zero_crossings':sum(m['internal_zero_crossings'] for m in allm),
    'incomplete_ballooning_integrations':sum(m['ballooning_incomplete'] for m in allm),
    'gs_rms_max':max(m['gs_error_rms'] for m in allm),
    'gs_error_max':max(m['gs_error_max'] for m in allm),
    'dcon_runs':len(allm),
    'violations':c['violations']}
 out['refinement']={}
 for scenario,levels in r.get('cases',{}).items():
  if 'coarse' not in levels or 'fine' not in levels:continue
  c,f=levels['coarse'],levels['fine'];cm=c['stability']['modes'];fm=f['stability']['modes']
  out['refinement'][scenario]={
   'pressure_relative':abs(f['equilibrium']['average_pressure_Pa']/c['equilibrium']['average_pressure_Pa']-1),
   'q_absolute_max':max(abs(cm[n]['globals']['qmin']-fm[n]['globals']['qmin']) for n in cm),
   'energy_absolute_by_n':{n:abs(cm[n]['minimum_energy'][0]-fm[n]['minimum_energy'][0]) for n in cm}}
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();s=json.dumps(summarize(a.report),indent=2,allow_nan=False)
 if a.output:a.output.write_text(s+'\n')
 else:print(s)
