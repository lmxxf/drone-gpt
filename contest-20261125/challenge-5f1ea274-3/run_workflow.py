#!/usr/bin/env python3
"""Continue capped local verification; submit only a genuinely feasible result."""
import os
os.sched_setaffinity(0,{12,13,14,15})
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import fcntl,hashlib,json,pathlib,re,shutil,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent
TRACE=pathlib.Path('/home/lmxxf/.codex/sessions/2026/11/26/rollout-2026-11-26T00-36-37-01a21390-f861-7282-ae9b-d35ef063f98d.jsonl')
sys.path.insert(0,str(ROOT));from solve_boundary import write_json
# solve_boundary sets search thread defaults; verifier MPI workers use one each.
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.chdir(ROOT)
def state(stage,**kw):write_json(ROOT/'workflow_status.json',{'stage':stage,'updated':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw})
def full(p,folder):
    reward=folder/'result/reward.json'
    if not reward.exists():
        folder.mkdir(parents=True,exist_ok=True)
        snapshot=folder/'boundary.json'
        if snapshot.exists() and snapshot.read_bytes()!=p.read_bytes():raise RuntimeError('snapshot mismatch')
        shutil.copyfile(p,snapshot)
        if (folder/'work').exists() and any((folder/'work').iterdir()):raise RuntimeError('incomplete existing evaluator run requires explicit investigation')
        state('full_verification',candidate=str(snapshot.relative_to(ROOT)),result=str(reward.relative_to(ROOT)))
        with open(folder/'log.txt','w') as log:
            subprocess.run([sys.executable,'-I',str(ROOT/'workspace/eval.py'),str(snapshot),'--workdir',str(folder/'work'),'--reward-dir',str(folder/'result'),'--threads','4'],stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads(reward.read_text())
def main():
    lock=open(ROOT/'workflow.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid()));lock.flush()
    start=time.time()
    state('waiting_for_optimizer')
    while not (ROOT/'optimizer_status.json').exists():
        if time.time()-start>7200:raise TimeoutError('optimizer has not finished within two hours')
        active=False
        for proc in pathlib.Path('/proc').iterdir():
            if not proc.name.isdigit():continue
            try:
                cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ')
                if b'python solve_boundary.py --optimize --max-evals 24' in cmd and (proc/'cwd').resolve()==ROOT:
                    active=True;break
            except (OSError,RuntimeError):continue
        if not active:
            state('optimizing_or_resuming_cached_search')
            with open(ROOT/'optimization.log','a') as log:
                subprocess.run([sys.executable,'solve_boundary.py','--optimize','--max-evals','24'],stdout=log,stderr=subprocess.STDOUT,check=True)
        else:time.sleep(15)
    rows=[]
    for rp in (ROOT/'runs').glob('*/screen.json'):
        r=json.loads(rp.read_text())
        if not r.get('screen_pass') or r.get('qs_rms',1)>.0031 or r['profile']['well']<.00025:continue
        if r['geometry']['aspect_ratio']>=6:continue
        p=rp.parent/'boundary.json';rows.append((r['geometry']['aspect_ratio'],p))
    rows.sort(key=lambda x:x[0]);unique=[];seen=set()
    for A,p in rows:
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest);unique.append((A,p,digest))
    # Bound total extra cost to at most two original-evaluator runs (24 hours).
    selected=unique[:1]
    if unique:
        safer=[]
        for item in unique[1:]:
            row=json.loads((item[1].parent/'screen.json').read_text())
            if item[0]>=unique[0][0]+.1 and row['qs_rms']<=.002 and row['profile']['well']>=.0005:safer.append(item)
        if safer:selected.append(safer[0])
        elif len(unique)>1:selected.append(unique[1])
    write_json(ROOT/'verification_queue.json',[{'aspect_ratio':A,'boundary':str(p.relative_to(ROOT)),'sha256':digest} for A,p,digest in selected])
    verified=[]
    for A,p,digest in selected:
        folder=ROOT/'runs'/('certify_'+digest[:12]);r=full(p,folder)
        if r.get('valid')==1 and r.get('feasible')==1 and r.get('evaluator_error')==0:
            details=json.loads((folder/'result/details.json').read_text())
            verified.append((details['metrics']['aspect_ratio'],folder/'boundary.json',r,folder/'result/details.json'))
            break # candidates are sorted by A
    # A fresh baseline is the fallback after the interrupted scheduling trial.
    base_reward=ROOT/'results/full_qa_well/reward.json'
    if not verified:
        baseline=ROOT/'workspace/reference_configurations/landreman_paul_2022_QA_well.json'
        folder=ROOT/'runs/certify_baseline'
        r=full(baseline,folder)
        if r.get('valid')==1 and r.get('feasible')==1 and r.get('evaluator_error')==0:
            d=folder/'result/details.json';details=json.loads(d.read_text())
            verified.append((details['metrics']['aspect_ratio'],folder/'boundary.json',r,d))
    if base_reward.exists():
        r=json.loads(base_reward.read_text())
        if r.get('valid')==1 and r.get('feasible')==1 and r.get('evaluator_error')==0:
            d=ROOT/'results/full_qa_well/details.json';details=json.loads(d.read_text())
            verified.append((details['metrics']['aspect_ratio'],ROOT/'workspace/reference_configurations/landreman_paul_2022_QA_well.json',r,d))
    if not verified:
        state('no_verified_feasible_boundary',submitted=False);return
    A,p,r,d=min(verified,key=lambda x:x[0])
    out=ROOT/'submission_outputs';out.mkdir(exist_ok=True)
    if any(x.name not in {'boundary.json','solve_boundary.py','stage_report.json'} for x in out.iterdir()):raise RuntimeError('submission_outputs contains unexpected files')
    data=json.loads(p.read_text());write_json(ROOT/'boundary.json',data);write_json(out/'boundary.json',data)
    shutil.copyfile(TRACE,ROOT/'session_trace.jsonl')
    write_json(ROOT/'certified_result.json',{'reward':r,'details_path':str(d.relative_to(ROOT)),'sha256':hashlib.sha256((out/'boundary.json').read_bytes()).hexdigest()})
    if (ROOT/'manual_submission_control').exists():
        state('certified_ready_for_submission',aspect_ratio=A,local_reward=r,details_path=str(d.relative_to(ROOT)));return
    state('submitting',aspect_ratio=A,local_reward=r)
    cmd=[str(pathlib.Path.home()/'.local/bin/playground'),'submit','--challenge-id','challenge-5f1ea274-3','--outputs',str(out),'--trace',str(TRACE)]
    with open(ROOT/'submission_receipt.txt','w') as log:done=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    shutil.copyfile(TRACE,ROOT/'session_trace.jsonl')
    if done.returncode:raise RuntimeError('platform submission failed; inspect submission_receipt.txt, do not create a duplicate attempt blindly')
    receipt=(ROOT/'submission_receipt.txt').read_text()
    match=re.search(r'["\']?attempt_id["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]+)',receipt)
    attempt=match.group(1) if match else None
    state('submitted',attempt_id=attempt,local_reward=r,receipt='submission_receipt.txt')
    if attempt:
        for _ in range(1440):
            result=subprocess.run([str(pathlib.Path.home()/'.local/bin/playground'),'status','--attempt-id',attempt,'--json'],capture_output=True,text=True)
            (ROOT/'platform_status.json').write_text(result.stdout)
            try:record=json.loads(result.stdout)
            except ValueError:record={}
            status=str(record.get('status',record.get('attempt',{}).get('status',''))).lower()
            if status in {'completed','succeeded','success','finished','failed','error','scored'}:
                state('platform_finished',attempt_id=attempt,platform_status=status,local_reward=r);return
            time.sleep(30)
if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('error',error=f'{type(exc).__name__}: {exc}');raise
