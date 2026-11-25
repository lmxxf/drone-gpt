"""Held-out native waveform tests with actual delayed q_min sensor calibration."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,argparse,multiprocessing,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));sys.path.insert(0,str(ROOT/'tools'));import eval as ev
from library_search import replay_metrics

def job(args):
    name,seed,override,level,cfile,ffile,out=args
    idx=(multiprocessing.current_process()._identity[0]-1)%10;os.sched_setaffinity(0,{[5,6,7,8,9,15,16,17,18,19][idx]})
    from fast_native import NativeScan
    import plant
    case=ev.read_json(ROOT/'workspace/case.json');physical=ev.scenario(seed,case)
    if override:physical.update(override)
    record={'name':name,'seed':seed,'physical':physical,'level':level,'coarse_source':cfile,'fine_source':ffile,'screening_scope':'q-calibrated feedforward only; bounded PI is checked in final evaluator replay'};s=None
    try:
        s=NativeScan(physical,level);initial=plant.diagnostics(s.env);qtrue=initial['q_min'];coarse=np.array(json.load(open(cfile))['commands_MW'],np.float32);fine=np.array(json.load(open(ffile))['commands_MW'],np.float32)
        noise=np.clip(np.random.Generator(np.random.PCG64(physical['sensor_seed'])).standard_normal((60,9)),-3,3)*np.array(case['observation']['noise_relative_sigma']);qmid=np.float32((1.8337676430354448+1.8279795705948279)/2);qsum=np.float32(0);count=np.float32(0);commands=[];weights=[];prev=np.array([28.,15.],np.float32)
        for k in range(60):
            if k<15:qsum=np.float32(qsum+np.float32(np.float32(np.float32(qtrue*(1+noise[k,5]))-qmid)*100));count=np.float32(count+1)
            gain=np.float32(1/(1+np.exp(np.float32(-40)*np.float32(qsum/count))));target=(fine[k]+gain*(coarse[k]-fine[k])).astype(np.float32);cmd=(prev+np.clip(target-prev,np.float32(-1.999),np.float32(1.999))).astype(np.float32);commands.append(cmd);weights.append(float(gain));prev=cmd
        curve=np.vstack(([28.,15.],[28.,15.],np.array(commands)[:59]));d,seconds=s.run(curve);assert np.max(abs(d[:,6]-qtrue))<1e-8;m,v,trace=replay_metrics(d,initial,s.dt)
        record.update(state='native_complete',metrics=m,violation=v,trace=trace,q_weight=weights,wall_time_s=seconds);np.save(Path(out)/(name+'-'+level+'.npy'),d)
    except Exception as e:record.update(state='unresolved',reason=str(e))
    finally:
        if s is not None:s.close()
    (Path(out)/(name+'-'+level+'.json')).write_text(json.dumps(record,indent=2));return {k:v for k,v in record.items() if k not in ['trace','q_weight']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('coarse');p.add_argument('fine');p.add_argument('out');a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    worlds=[(f'dev{i+1}',s,None) for i,s in enumerate([2026100801,2026100802,2026100803,2026100804])]+[('lowZ',20261125,None),('mid_event',17,{'event_start_s':12.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98}),('late_event',991,{'event_start_s':16.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98})]
    args=[(name,seed,over,level,str(Path(a.coarse).resolve()),str(Path(a.fine).resolve()),str(out)) for level in ['fine','coarse'] for name,seed,over in worlds]
    with ProcessPoolExecutor(max_workers=10,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,x) for x in args]):print(json.dumps(f.result()),flush=True)
    case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];results=[]
    for name,_,_ in worlds:
        rows=[json.load(open(out/(name+'-'+l+'.json'))) for l in ['coarse','fine']];r={'name':name}
        if any(x['state']!='native_complete' or x.get('violation') for x in rows):r.update(feasible=False,rows=rows)
        else:
            try:J=ev.compare_resolutions(case,ref,[(x['metrics'],x['violation'],x['trace']) for x in rows]);r.update(feasible=True,J_upper_J=J,min_Q_surplus_J=min(x['metrics']['fusion_energy_J'] for x in rows)-ref['required_fusion_energy_J'])
            except ev.Unresolved as e:r.update(feasible=False,reason=str(e))
        results.append(r)
    (out/'summary.json').write_text(json.dumps({'scope':'offline native held-out tests; original evaluator replay still required','feasible':all(x['feasible'] for x in results),'pairs':results},indent=2));print(json.dumps(results),flush=True)
