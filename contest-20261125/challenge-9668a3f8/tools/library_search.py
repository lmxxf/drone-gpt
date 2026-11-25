"""Factorized two-model native waveform search; never substitutes for final evaluator replay."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing,itertools,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'))
import eval as ev
OUT=ROOT/'runs/library';OUT.mkdir(exist_ok=True)

def commands(elow,hold,cool_rate=3.8,terminal_rate=3.9,terminal_start=27.):
    prev=np.array([28.,15.],np.float32);out=[]
    for k in range(60):
        t=k*.5;e=max(elow,20-cool_rate*max(t-hold,0))+terminal_rate*max(t-terminal_start,0)
        desired=np.array([33.,min(20.,e)],np.float32);a=np.clip(prev+np.clip(desired-prev,np.float32(-1.999),np.float32(1.999)),[0,0],[33,20]).astype(np.float32);out.append(a);prev=a
    return np.asarray(out,np.float32)

def replay_metrics(d,initial,dt):
    case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];accepted=[initial]
    for x in d:
        accepted.append({'time_s':float(x[0]),'P_fusion_W':float(x[1]),'W_thermal_total_J':float(x[2]),'beta_N':float(x[5]),'q_min':float(x[6]),'q95':float(x[7]),'P_external_W':float(x[8]),'E_external_total_J':float(x[9]),'P_pedestal_W':float(x[10]),'temperature_max_keV':float(x[11]),'fgw_n_e_line_avg':initial['fgw_n_e_line_avg']})
    stride=int(round(.5/dt));trace=accepted[::stride];assert len(trace)==61
    energy=accepted[-1]['E_external_total_J']-initial['E_external_total_J'];check=float(np.trapezoid([x['P_external_W'] for x in accepted],[x['time_s'] for x in accepted]));assert abs(check-energy)<1e-8*energy
    m,v=ev.violations(case,trace,ref,accepted);m.update(ev.supplied_heat_metrics(accepted,energy));return m,v,trace

def job(arg):
    level,elow,hold=arg;idx=(multiprocessing.current_process()._identity[0]-1)%10;os.sched_setaffinity(0,{[5,6,7,8,9,15,16,17,18,19][idx]})
    from fast_native import NativeScan
    import plant
    key=f'{level}-e{elow}-h{hold}';record={'key':key,'level':level,'elow':elow,'hold':hold,'commands_MW':commands(elow,hold).tolist(),'protocol':'float32 commands, slew 1.999 MW, native fixed timestep, native domain check at every step'}
    s=None
    try:
        s=NativeScan({'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98},level);initial=plant.diagnostics(s.env);curve=np.vstack(([28.,15.],[28.,15.],np.asarray(record['commands_MW'])[:59]));d,seconds=s.run(curve);m,v,trace=replay_metrics(d,initial,s.dt)
        record.update(state='native_complete',wall_time_s=seconds,metrics=m,violation=v,trace=trace);np.save(OUT/(key+'.npy'),d)
    except Exception as error:
        import traceback
        record.update(state='unresolved',reason=str(error),traceback=traceback.format_exc())
    finally:
        if s is not None:s.close()
    (OUT/(key+'.json')).write_text(json.dumps(record,indent=2));return {k:v for k,v in record.items() if k not in ['commands_MW','trace','traceback']}

def rank():
    case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];rows=[json.load(open(p)) for p in OUT.glob('*.json') if p.stem!='ranking'];coarse=[r for r in rows if r.get('level')=='coarse' and r.get('state')=='native_complete'];fine=[r for r in rows if r.get('level')=='fine' and r.get('state')=='native_complete'];pairs=[]
    for c,f in itertools.product(coarse,fine):
        if c['violation'] or f['violation']:continue
        if min(c['metrics']['fusion_energy_J'],f['metrics']['fusion_energy_J'])<ref['required_fusion_energy_J']+2e6:continue
        if min(c['trace'][-1]['W_thermal_total_J'],f['trace'][-1]['W_thermal_total_J'])<.985*ref['reference_terminal_W_thermal_J']:continue
        try:J=ev.compare_resolutions(case,ref,[(r['metrics'],r['violation'],r['trace']) for r in [c,f]])
        except ev.Unresolved:continue
        delta=np.abs(np.array([r['P_fusion_W'] for r in c['trace']])-np.array([r['P_fusion_W'] for r in f['trace']]))
        pairs.append({'coarse':c['key'],'fine':f['key'],'J_upper_J':J,'min_Q_surplus_J':min(c['metrics']['fusion_energy_J'],f['metrics']['fusion_energy_J'])-ref['required_fusion_energy_J'],'grid_power_max_W':float(delta.max()),'grid_power_after3s_max_W':float(delta[7:].max())})
    pairs.sort(key=lambda r:r['J_upper_J']);(OUT/'ranking.json').write_text(json.dumps(pairs,indent=2));print(json.dumps({'candidate_pairs':len(pairs),'best':pairs[:10]},indent=2),flush=True)
if __name__=='__main__':
    args=[('coarse',e,h) for e in [4.5,5.,5.5] for h in [5.,6.,7.]]+[('fine',e,h) for e in [3.,3.5,3.8] for h in [6.,7.,8.,9.]]
    with ProcessPoolExecutor(max_workers=10,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,a) for a in args]):print(json.dumps(f.result()),flush=True)
    rank()
