"""Read the actual platform result; never post a self-reported score."""
import argparse,json,subprocess,sys,datetime,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();out=ROOT/'runs'/f'platform-{a.attempt}-status.json';r=subprocess.run(['/home/lmxxf/.local/bin/playground','status','--attempt-id',a.attempt,'--json'],capture_output=True,text=True);assert r.returncode==0,r.stderr;x=json.loads(r.stdout);tmp=Path(str(out)+'.tmp');tmp.write_text(r.stdout);os.replace(tmp,out);st=x.get('scoringState') or {};final=bool(st.get('scoreIsFinal'));small={'attempt_id':a.attempt,'time_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':st.get('state',x.get('status')),'score_is_final':final,'display_score':st.get('displayScore'),'zero_reason':st.get('zeroReason')}
 with (ROOT/'runs/platform-status-history.jsonl').open('a') as log:log.write(json.dumps(small)+'\n')
 for reportpath in [ROOT/'evaluation_report.json',ROOT/'runs/submissions'/a.attempt/'evaluation_report.json']:
  if not reportpath.exists():continue
  report=json.load(open(reportpath));f=report['formal_submission']
  if str(f.get('attempt_id'))!=a.attempt:continue
  f.update(state='scored' if final else small['state'],platform_score_is_final=final,platform_status_path=str(out),platform_scoring_state=st,platform_execution_status=x.get('execStatus'),platform_execution_log=x.get('execLog'),platform_results=x.get('resultsJson'))
  if final:f.update(score=st.get('displayScore',x.get('score')),platform_scoring_details=x.get('scoringDetails'),platform_taskeval=x.get('taskeval'))
  reportpath.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(small),flush=True)
 if final:print(json.dumps({'scoringDetails':x.get('scoringDetails'),'taskeval':x.get('taskeval'),'scorecard':x.get('scorecard'),'resultsJson':x.get('resultsJson')},ensure_ascii=False),flush=True)

 if not final and small['state']=='failed':raise SystemExit(2)
