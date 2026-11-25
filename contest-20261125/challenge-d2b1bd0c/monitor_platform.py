#!/usr/bin/env python3
"""Finish the already-submitted task by recording its asynchronous platform score."""
import argparse,datetime,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.sched_setaffinity(0,{4,5,6,7})
ENV={**os.environ,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':str(ROOT/'native/tmp')}

def save(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt-id',required=True);ap.add_argument('--hours',type=float,default=24);a=ap.parse_args()
    (ROOT/'runs/platform_monitor.pid').write_text(str(os.getpid())+'\n')
    local=json.loads((ROOT/'submission_outputs/evaluation_report.json').read_text())['formal']
    end=time.monotonic()+a.hours*3600
    while time.monotonic()<end:
        stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            p=subprocess.run([str(Path.home()/'.local/bin/playground'),'status','--attempt-id',a.attempt_id,'--json'],cwd=ROOT,env=ENV,capture_output=True,text=True,timeout=45)
            if p.returncode:raise RuntimeError(p.stderr.strip())
            r=json.loads(p.stdout)
            if str(r.get('id'))!=a.attempt_id:raise RuntimeError('Unexpected attempt id')
            save(ROOT/'platform_attempt.json',r)
            s=r.get('scoringState',{});final=bool(s.get('scoreIsFinal'))
            result={'attempt_id':a.attempt_id,'updated_at_utc':stamp,'state':s.get('state'),'score_is_final':final,'platform_score':r.get('score') if final else None,'local_formal_score':local['reward'],'local_average_pressure_Pa':local['worst_average_pressure_Pa'],'scoring_state':s,'scoring_queue':r.get('scoringQueue'),'monitor_pid':os.getpid()}
            save(ROOT/'platform_result.json',result)
            print(json.dumps({k:result[k] for k in ('updated_at_utc','state','score_is_final','platform_score')}),flush=True)
            if final:return
        except Exception as exc:
            print(json.dumps({'updated_at_utc':stamp,'query_error':str(exc)}),flush=True)
        time.sleep(50)
    print(json.dumps({'monitor_expired':True,'score_not_fabricated':True}),flush=True)
if __name__=='__main__':main()
