"""Submit only a fully checked artifact with the authentic current Codex trace."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRACE=Path('/home/lmxxf/.codex/sessions/2026/11/25/rollout-2026-11-25T22-15-37-01a2130f-e199-7e90-8883-e3f2d85168e0.jsonl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 reportpath=ROOT/'evaluation_report.json';report=json.load(open(reportpath));source=Path(report['artifact']['path']);expected=report['artifact']['sha256'];assert sha(source)==expected
 assert report['development']['feasible'] and report['all_tested_scenarios']['feasible'];assert report['original_evaluator_replay']['scenario_count']>=9;assert report['native_stress']['feasible'] and report['native_stress']['scenario_count']>=16 and report['native_stress']['candidate_sha256']==expected
 assert report['formal_submission']['state']=='not_submitted'
 out=ROOT/'submission';out.mkdir(exist_ok=True);assert all(p.name=='controller.onnx' and p.is_file() for p in out.iterdir())
 for target in [ROOT/'controller.onnx',out/'controller.onnx']:
  tmp=Path(str(target)+'.tmp');shutil.copyfile(source,tmp);os.replace(tmp,target);assert sha(target)==expected
 shutil.copyfile(TRACE,ROOT/'session_trace.jsonl')
 validation=subprocess.run(['/home/lmxxf/.local/bin/playground','trace','validate','--trace',str(TRACE)],capture_output=True,text=True);(ROOT/'runs/final-trace-validation.log').write_text(validation.stdout+validation.stderr);assert validation.returncode==0
 report['artifact'].update(validated_source_path=str(source),path=str(ROOT/'controller.onnx'));report['formal_submission'].update(state='submitting',trace_path=str(TRACE),submitted_controller_sha256=expected)
 reportpath.write_text(json.dumps(report,indent=2)+'\n')
 args=['/home/lmxxf/.local/bin/playground','submit','--challenge-id','challenge-9668a3f8','--outputs',str(out),'--trace',str(TRACE),'--harness','codex','--bundle-out',str(ROOT/'runs/final-bundle.zip')]
 with (ROOT/'runs/final-submit.log').open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
 data=(ROOT/'runs/final-submit.log').read_text();record=None
 for i,ch in enumerate(data):
  if ch=='{':
   try:
    value,_=json.JSONDecoder().raw_decode(data[i:])
    if isinstance(value,dict) and value.get('attempt_id'):record=value;break
   except json.JSONDecodeError:pass
 if r.returncode or record is None:
  report['formal_submission'].update(state='submission_error',exit_code=r.returncode,log=str(ROOT/'runs/final-submit.log'));reportpath.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'state':'submission_error','exit_code':r.returncode,'log':'runs/final-submit.log'}),flush=True);return r.returncode or 1
 small={k:record.get(k) for k in ['status','challenge_id','attempt_id','attempt_url','bundle','bundle_sha256']};(ROOT/'runs/formal-submission-summary.json').write_text(json.dumps(small,indent=2)+'\n');report['formal_submission'].update(small,state='submitted',trace_backup_path=str(ROOT/'session_trace.jsonl'));reportpath.write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(TRACE,ROOT/'session_trace.jsonl');print(json.dumps(small),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
