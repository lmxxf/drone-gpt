"""Finalize a specific candidate without overwriting the currently submitted report."""
import argparse,json,time,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def ready_report(path):
 try:r=json.load(open(path))
 except (FileNotFoundError,json.JSONDecodeError):return None
 return r if 'wall_time_s' in r else None
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--params',required=True);p.add_argument('--primary',required=True);p.add_argument('--extra',action='append',default=[]);p.add_argument('--stress',required=True);p.add_argument('--report',required=True);p.add_argument('--plot',required=True);a=p.parse_args();dirs=[a.primary]+a.extra
 while True:
  rs=[ready_report(Path(d)/'report.json') for d in dirs]
  if all(r is not None for r in rs):
   assert all(r['feasible'] and not r.get('error') for r in rs),[(d,r.get('error'),r.get('violation')) for d,r in zip(dirs,rs)];assert sum(len(r['episodes']) for r in rs)==18;break
  time.sleep(10)
 ps=[s.split(None,2) for s in subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True).splitlines()[1:]];match=f'tools/block_eval.py {a.candidate} {a.stress} ';roots=[int(pid) for pid,ppid,args in ps if args.startswith(sys.executable+' '+match)];changes=[]
 for root in roots:
  workers=sorted(int(pid) for pid,ppid,args in ps if int(ppid)==root and 'from multiprocessing.spawn import spawn_main' in args)
  for worker,cpu in zip(workers,[5,6,7,8,9,16,17]):
   ids={worker}
   while True:
    more=ids|{int(pid) for pid,ppid,args in ps if int(ppid) in ids}
    if more==ids:break
    ids=more
   for pid in ids:
    try:
     before=sorted(os.sched_getaffinity(pid))
     for t in Path('/proc',str(pid),'task').iterdir():os.sched_setaffinity(int(t.name),{cpu})
     changes.append({'pid':pid,'old':before,'new':[cpu]})
    except (FileNotFoundError,ProcessLookupError):pass
 Path(str(a.report)+'.affinity.json').write_text(json.dumps(changes,indent=2))
 cmd=[sys.executable,str(ROOT/'tools/finalize_report.py'),'--candidate',a.candidate,'--params',a.params,'--run',a.primary,'--out',a.report]
 for d in a.extra:cmd+=['--extra',d]
 subprocess.run(cmd,check=True);subprocess.run([sys.executable,str(ROOT/'tools/audit_memory.py'),'--report',a.report],check=True);subprocess.run([sys.executable,str(ROOT/'tools/plot_report.py'),'--report',a.report,'--output',a.plot],check=True);print('ORIGINAL_REPORT_READY',flush=True)
 while True:
  try:r=json.load(open(Path(a.stress)/'summary.json'))
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
  assert r['feasible'] and len(r['pairs'])>=16,r;break
 subprocess.run([sys.executable,str(ROOT/'tools/attach_stress.py'),a.stress,'--report',a.report],check=True);print('ALL_LOCAL_VALIDATION_READY',flush=True)
