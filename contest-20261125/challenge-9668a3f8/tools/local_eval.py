"""Local architecture port: uses unchanged evaluator physics and checks.
Not the trusted runtime release gate; all results are labelled local_untrusted_runtime.
"""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',str(__import__('pathlib').Path(__file__).resolve().parents[1]/'runs/jax-cache'))
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,argparse,multiprocessing,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspace'))
import eval as ev

def job(args):
    seed, candidate, out, levels=args
    allowed=sorted(os.sched_getaffinity(0))
    idx=multiprocessing.current_process()._identity[0]-1
    chunk=max(1,len(allowed)//4)
    custom=os.environ.get('DRONE_EVAL_CPUS')
    os.sched_setaffinity(0,{int(custom.split(',')[idx%len(custom.split(','))])} if custom else allowed[(idx%4)*chunk:(idx%4+1)*chunk])
    import jax
    jax.config.update('jax_compilation_cache_dir',os.environ['JAX_COMPILATION_CACHE_DIR'])
    jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
    case=ev.read_json(ROOT/'workspace/case.json')
    ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development']
    cm,pl=ev.module('controller'),ev.module('plant')
    results=[]
    for level in levels:
        t=time.time()
        try:
            if os.environ.get('DRONE_CORNER')=='1':
                (Path(out)/str(seed)/level).mkdir(parents=True)
                physical=ev.scenario(seed,case);physical.update(event_start_s=float(os.environ.get('DRONE_CORNER_EVENT','8')),Z_eff_after_event=1.75,initial_temperature_multiplier=.98)
                env=pl.CoreHeatEnv(ROOT/'workspace',physical,level)
                try:r=ev._episode_in_env(cm,pl,Path(candidate),Path(out)/str(seed)/level,seed,ref,case,env)
                finally:env.close()
            else:
                r=ev.episode(cm,pl,Path(candidate),ROOT/'workspace',Path(out)/str(seed)/level,seed,level,ref)
        except Exception as error:
            import traceback
            raise RuntimeError(traceback.format_exc()) from None
        results.append(r)
        print(json.dumps({'seed':seed,'level':level,'seconds':time.time()-t,'metrics':r[0],'violation':r[1]}),flush=True)
    objective=ev.compare_resolutions(case,ref,results) if len(levels)==2 else results[0][0]['J_supplied_total_J']
    return seed,results,objective

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('output');p.add_argument('--seeds',nargs='+',type=int,default=[2026100801,2026100802,2026100803,2026100804]);p.add_argument('--levels',nargs='+',default=['coarse','fine']);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'controller.onnx';shutil.copyfile(a.candidate,snapshot)
    ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];t=time.time()
    report={'physical_override':{'event_start_s':float(os.environ.get('DRONE_CORNER_EVENT','8')),'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98} if os.environ.get('DRONE_CORNER')=='1' else None,'runtime':'local_untrusted_runtime','release_gate_passed':False,'candidate_sha256':ev.sha(a.candidate),'episodes':[]};v={};J=0
    try:
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(job,(s,str(snapshot),str(out),a.levels)) for s in a.seeds]
            for f in as_completed(futures):
                seed,rs,obj=f.result();J=max(J,obj)
                for level,r in zip(a.levels,rs):
                    report['episodes'].append({'seed':seed,'level':level,'metrics':r[0],'violation':r[1]})
                    for k,x in r[1].items():v[k]=max(v.get(k,0),x)
                report.update(J=J,violation=v,local_score=ev.score(J,ref['J_reference_J'],v),feasible=not v)
                (out/'report.json').write_text(json.dumps(report,indent=2))
    except Exception as e:
        import traceback
        report.update(error=str(e),traceback=traceback.format_exc(),feasible=False)
        raise
    finally:
        report['wall_time_s']=time.time()-t
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
