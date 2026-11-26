"""Read the actual platform result; never post a self-reported score."""
import argparse,json,subprocess,sys,datetime,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();out=ROOT/'runs'/f'platform-{a.attempt}-status.json';r=subprocess.run(['/home/lmxxf/.local/bin/playground','status','--attempt-id',a.attempt,'--json'],capture_output=True,text=True);assert r.returncode==0,r.stderr;x=json.loads(r.stdout);previous=json.loads(out.read_text()) if out.exists() else None;key=lambda z:(z.get('execStatus'),(z.get('scoringState') or {}).get('state'),(z.get('scoringState') or {}).get('scoreIsFinal'),(z.get('scoringState') or {}).get('displayScore'));changed=previous is None or key(previous)!=key(x);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');events=ROOT/'runs'/f'platform-{a.attempt}-state-changes';events.mkdir(exist_ok=True);
 if changed:
  (events/(stamp+'-new.json')).write_text(r.stdout)
  if previous is not None:(events/(stamp+'-previous.json')).write_text(json.dumps(previous,indent=2)+'\n')
 tmp=Path(str(out)+'.tmp');tmp.write_text(r.stdout);os.replace(tmp,out);st=x.get('scoringState') or {};final=bool(st.get('scoreIsFinal'));small={'attempt_id':a.attempt,'time_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':st.get('state',x.get('status')),'score_is_final':final,'display_score':st.get('displayScore'),'zero_reason':st.get('zeroReason'),'exec_status':x.get('execStatus'),'exec_log':x.get('execLog'),'harbor_reward':(x.get('resultsJson') or {}).get('harbor_reward'),'harbor_status':(x.get('resultsJson') or {}).get('harbor_status')}
 with (ROOT/'runs/platform-status-history.jsonl').open('a') as log:log.write(json.dumps(small)+'\n')
 for reportpath in [ROOT/'evaluation_report.json',ROOT/'runs/submissions'/a.attempt/'evaluation_report.json']:
  if not reportpath.exists():continue
  report=json.load(open(reportpath));f=report['formal_submission']
  if str(f.get('attempt_id'))!=a.attempt:continue
  f.update(state='scored' if final else small['state'],platform_score_is_final=final,platform_status_path=str(out),platform_scoring_state=st,platform_execution_status=x.get('execStatus'),platform_execution_log=x.get('execLog'),platform_results=x.get('resultsJson'),benchmark_reward=(x.get('resultsJson') or {}).get('harbor_reward'))
  if final:f.update(score=st.get('displayScore',x.get('score')),platform_scoring_details=x.get('scoringDetails'),platform_taskeval=x.get('taskeval'))
  reportpath.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(small),flush=True)
 if final:print(json.dumps({'scoringDetails':x.get('scoringDetails'),'taskeval':x.get('taskeval'),'scorecard':x.get('scorecard'),'resultsJson':x.get('resultsJson')},ensure_ascii=False),flush=True)

