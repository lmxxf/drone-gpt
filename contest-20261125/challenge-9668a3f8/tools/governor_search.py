"""Native closed-loop search for a diagnostic-based cooling-rate governor."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));sys.path.insert(0,str(ROOT/'tools'));import eval as ev
from library_search import replay_metrics
OUT=ROOT/'runs/governor';OUT.mkdir(exist_ok=True)
def job(arg):
    level,K,limit=arg;idx=(multiprocessing.current_process()._identity[0]-1)%3;os.sched_setaffinity(0,{[17,18,19][idx]})
    from fast_native import NativeScan
    from closed_native import run
    import plant
    case=ev.read_json(ROOT/'workspace/case.json');physical=ev.scenario(17,case);physical.update(event_start_s=8.,Z_eff_after_event=1.75,initial_temperature_multiplier=.98);noise=np.clip(np.random.Generator(np.random.PCG64(physical['sensor_seed'])).standard_normal((60,9)),-3,3)*np.array(case['observation']['noise_relative_sigma']);c=np.array(json.load(open(ROOT/'runs/library/coarse-e5.0-h5.0.json'))['commands_MW'],np.float32);f=np.array(json.load(open(ROOT/'runs/library/fine-e3.5-h6.0.json'))['commands_MW'],np.float32);s=None;key=f'{level}-k{K}-r{limit}';r={'key':key,'level':level,'gain':K,'rate_limit':limit}
    try:
        s=NativeScan(physical,level);initial=plant.diagnostics(s.env);d,seconds=run(s,noise,c,f,[K,limit,2.]);m,v,trace=replay_metrics(d,initial,s.dt);r.update(state='native_complete',metrics=m,violation=v,trace=trace,wall_time_s=seconds,issued_commands_MW=d[::int(round(.5/s.dt)),15:17].tolist());np.save(OUT/(key+'.npy'),d)
    except Exception as error:
        import traceback
        r.update(state='unresolved',reason=str(error),traceback=traceback.format_exc())
    finally:
        if s is not None:s.close()
    (OUT/(key+'.json')).write_text(json.dumps(r,indent=2));return {k:v for k,v in r.items() if k not in ['trace','issued_commands_MW','traceback']}
if __name__=='__main__':
    params=[(K,r) for K in [4.,8.,12.] for r in [.3,.5]]
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,(level,K,r)) for level in ['fine','coarse'] for K,r in params]):print(json.dumps(f.result()),flush=True)
    case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];rows=[]
    for K,r in params:
        pair=[json.load(open(OUT/f'{l}-k{K}-r{r}.json')) for l in ['coarse','fine']];row={'gain':K,'rate_limit':r}
        if any(x['state']!='native_complete' or x.get('violation') for x in pair):row.update(feasible=False,states=[(x['state'],x.get('violation'),x.get('reason')) for x in pair])
        else:
            try:J=ev.compare_resolutions(case,ref,[(x['metrics'],x['violation'],x['trace']) for x in pair]);row.update(feasible=True,J_upper_J=J,min_Q_surplus_J=min(x['metrics']['fusion_energy_J'] for x in pair)-ref['required_fusion_energy_J'])
            except ev.Unresolved as error:row.update(feasible=False,reason=str(error))
        rows.append(row)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
