"""Produce an auditable report from completed, unchanged native evaluator runs."""
import argparse,hashlib,json,platform,sys,importlib.metadata
from pathlib import Path
import numpy as np
import onnx
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def collect(directory,candidate_hash):
    directory=Path(directory);raw=json.load(open(directory/'report.json'))
    assert raw.get('feasible') and not raw.get('error'),(directory,raw.get('error'),raw.get('violation'))
    assert raw['candidate_sha256']==candidate_hash
    case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];pairs=[]
    for seed in sorted({e['seed'] for e in raw['episodes']}):
        es=sorted([e for e in raw['episodes'] if e['seed']==seed],key=lambda e:e['level']);assert [e['level'] for e in es]==['coarse','fine']
        traces=[ev.read_json(directory/str(seed)/e['level']/'trace.json') for e in es]
        runs=[(e['metrics'],e['violation'],t) for e,t in zip(es,traces)];J=ev.compare_resolutions(case,ref,runs)
        states=[]
        for e,t in zip(es,traces):
            assert not e['violation'];accepted=ev.read_json(directory/str(seed)/e['level']/'accepted-trace.json')
            expected=6001 if e['level']=='coarse' else 12001;assert len(accepted)==expected;assert abs(accepted[-1]['time_s']-30)<1e-8;assert all(s['solver_error_state']==0 for s in accepted)
            states.append(len(accepted));del accepted
        scenario=ev.scenario(seed,case)
        if raw.get('physical_override'):
            scenario.update(raw['physical_override'] if isinstance(raw['physical_override'],dict) else dict(event_start_s=8.,Z_eff_after_event=1.75,initial_temperature_multiplier=.98))
        pairs.append({'seed':seed,'run_directory':str(directory.resolve()),'physical_override':raw.get('physical_override'),'scenario':scenario,'J_upper_J':J,'grid_heat_difference_J':abs(es[0]['metrics']['J_supplied_total_J']-es[1]['metrics']['J_supplied_total_J']),'grid_power_difference_W':float(np.max(np.abs(np.array([x['P_fusion_W'] for x in traces[0]])-np.array([x['P_fusion_W'] for x in traces[1]])))),'accepted_state_counts':states,'terminal_W_J':[t[-1]['W_thermal_total_J'] for t in traces],'episodes':es})
    return pairs

def summary(pairs):
    ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];es=[e for p in pairs for e in p['episodes']]
    return {'feasible':True,'scenario_count':len(pairs),'trajectory_count':len(es),'native_accepted_steps':sum(sum(n-1 for n in p['accepted_state_counts']) for p in pairs),'worst_supplied_heat_J':max(p['J_upper_J'] for p in pairs),'minimum_fusion_energy_ratio':min(e['metrics']['fusion_energy_J']/ref['required_fusion_energy_J'] for e in es),'minimum_terminal_W_ratio':min(w/ref['reference_terminal_W_thermal_J'] for p in pairs for w in p['terminal_W_J']),'maximum_fusion_rms_relative':max(e['metrics']['fusion_rms_relative'] for e in es),'maximum_fusion_peak_relative':max(e['metrics']['fusion_peak_relative'] for e in es),'maximum_beta_N':max(e['metrics']['beta_N_max'] for e in es),'maximum_temperature_keV':max(e['metrics']['temperature_max_keV'] for e in es),'maximum_positive_pedestal_heat_J':max(e['metrics']['E_pedestal_positive_J'] for e in es),'maximum_grid_heat_difference_J':max(p['grid_heat_difference_J'] for p in pairs),'maximum_grid_power_difference_W':max(p['grid_power_difference_W'] for p in pairs)}

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--candidate',required=True);a.add_argument('--params',required=True);a.add_argument('--run',required=True);a.add_argument('--extra',action='append',default=[]);a.add_argument('--out',required=True);args=a.parse_args()
    candidate=Path(args.candidate);sha=digest(candidate);primary=collect(args.run,sha);extras=[p for d in args.extra for p in collect(d,sha)];allpairs=primary+extras
    case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];dev=[p for p in primary if p['seed'] in case['scenario_distribution']['development_seeds']];assert len(dev)==4
    development=summary(dev);development.update(score=ev.score(development['worst_supplied_heat_J'],ref['J_reference_J'],{}),reference_supplied_heat_J=ref['J_reference_J'])
    m=onnx.load(candidate);versions={p:importlib.metadata.version(p) for p in case['versions']};assert versions==case['versions']
    base=ev.read_json(ROOT/'runs/baseline-local/summary.json')['groups']['development']
    report={'schema':'fusion-control-engineering-report/v1','artifact':{'path':str(candidate.resolve()),'sha256':sha,'bytes':candidate.stat().st_size,'nodes':len(m.graph.node),'opset':m.opset_import[0].version,'IR':m.ir_version,'operators':sorted({n.op_type for n in m.graph.node})},'parameters':json.load(open(args.params)),'runtime':{'validation_scope':'Local native physics replay, not the trusted release gate','unmodified_eval_entrypoint':{k:v for k,v in ev.read_json(candidate.parent/(candidate.stem+'-original-reward')/'details.json').items() if k in ['state','reason','wall_time_s']},'python':platform.python_version(),'architecture':platform.machine(),'versions':versions,'fingerprint_audit':ev.read_json(ROOT/'runs/runtime-audit.json')},'resources_sha256':{f:digest(ROOT/'workspace'/f) for f in ['eval.py','plant.py','controller.py','case.json','torax-config.json','runtime-lock.json','reference/baseline.json']},'baseline':{k:v for k,v in base.items() if k not in ['episodes','grid_checks']},'development':development,'all_tested_scenarios':summary(allpairs),'pairs':allpairs,'memory':{'0':'Delayed fusion-power trapezoidal integral, units 100 MJ','1':'Delay-aligned target-power trapezoidal integral, units 100 MJ','2-4':'Previous power/target values for delayed quadrature, units 10 MW','5':'Filtered fusion-power error, units 10 MW','6':'Filtered thermal-energy error, units 10 MJ','7-8':'PI heat correction and base heating command, units 10 MW','9-11':'Thermal energy, energy debt, terminal heat correction','12-14':'Slow/fast fusion-power filters and signed AC heat budget','15-16':'Centered q_min calibration sum and sample count','17-31':'Reserved zeros'},'estimator_note':'Memory integrals are exact discrete delayed trapezoidal recurrences up to float32 rounding. They estimate the unobserved native-step integral; feasibility and all reported Q values use the original evaluator accepted-step integration. The energy loop compares matched target and fusion quadratures.','q_calibration_note':'Gain scheduling uses only past noisy q_min observations, averaged during the first 7.5 seconds; the midpoint is calibrated from the two public native radial resolutions. This numerical-model calibration is explicit; no scenario/seed identifier is supplied to the graph.','formal_submission':{'state':'not_submitted','attempt_id':None,'score':None}}
    if 'coarse_commands_MW' in report['parameters']:
        for k in ['12-14','15-16','17-31']:report['memory'].pop(k,None)
        report['memory'].update({'12':'Centered q_min calibration sum','13':'q_min sample count','14':'Calibrated coarse-model mixing weight','15-31':'Reserved zeros'})
        if 'feedback_hold_s' in report['parameters']:
            report['memory'].pop('15-31',None);report['memory'].update({'15':'Held terminal feedback correction, units 10 MW','16-31':'Reserved zeros'})
    elif not report['parameters'].get('ac_gain'):
        for k in ['12-14','15-16','17-31']:report['memory'].pop(k,None)
        report['memory']['12-31']='Reserved zeros'
        report['q_calibration_note']=None
    Path(args.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'sha256':sha,'development':development,'all_tested_scenarios':report['all_tested_scenarios']},indent=2))
