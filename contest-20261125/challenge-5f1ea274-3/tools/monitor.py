import json,pathlib,re,datetime,subprocess,sys,os
os.sched_setaffinity(0,{12,13,14,15})
root=pathlib.Path(__file__).resolve().parents[1]
s=json.loads((root/'workflow_status.json').read_text());out={'time':datetime.datetime.now().astimezone().isoformat(),'workflow':s}
if s.get('candidate'):
 work=(root/s['candidate']).parent/'work'
 if (work/'simple/log.txt').exists():
  lines=(work/'simple/log.txt').read_text(errors='replace').splitlines();progress=[x.strip() for x in lines if '[progress]' in x]
  out['physics_stage']='SIMPLE';out['latest']=progress[-1] if progress else (lines[-1] if lines else '')
  q=subprocess.run(['sudo','-n',sys.executable,str(root/'tools/read_simple_progress.py')],capture_output=True,text=True)
  if q.returncode==0:out['particles']=json.loads(q.stdout)
  else:out['counter_observation_error']=q.stderr[-300:]
 elif (work/'neo/neo_out.eq').exists():out['physics_stage']='NEO or starting SIMPLE'
 else:
  logs=list(work.glob('vmec_attempt*/log.txt'))
  if logs:
   p=max(logs,key=lambda q:q.stat().st_mtime);lines=p.read_text(errors='replace').splitlines();iters=[x.strip() for x in lines if re.match(r'^\s*\d+\s+[-+0-9.]+E',x)]
   out.update(physics_stage='VMEC',attempt=p.parent.name,latest=iters[-1] if iters else (lines[-1] if lines else ''))
if (root/'certified_result.json').exists():out['certified']=json.loads((root/'certified_result.json').read_text())
(root/'live_monitor.json').write_text(json.dumps(out,indent=2))
if '--brief' in sys.argv:
 brief={'stage':out.get('physics_stage',s['stage']),'run':pathlib.Path(s.get('candidate','')).parent.name}
 if 'particles' in out:
  q=out['particles'];brief.update(completed=q['completed'],total=q['total'],percent=round(100*q['fraction'],2),eta_hours=round(q.get('linear_eta_s',0)/3600,2),lost_observed=q.get('observed_loss_count_lower_bound'),unresolved_observed=q.get('observed_unresolved_lower_bound'))
 else:brief['latest']=out.get('latest',s)
 if 'certified' in out:brief['certified']=out['certified']
 print(json.dumps(brief))
else:print(json.dumps(out))
