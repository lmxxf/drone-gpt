#!/usr/bin/env python3
"""Package measured results, exact design, and replay code; no inferred scores."""
import hashlib,json,platform,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if len(sys.argv)!=2:raise SystemExit('usage: build_report.py VALIDATED_RUN_NAME')
    name=sys.argv[1];reward=read(ROOT/'results'/name/'reward.json');details=read(ROOT/'results'/name/'details.json')
    assert reward['valid']==1 and reward['feasible']==1 and details['status']=='feasible',reward
    baseline=read(ROOT/'results/uniform-full/details.json');baseline_reward=read(ROOT/'results/uniform-full/reward.json')
    frozen=read(ROOT/'workspace/reference/baseline.json')
    dest=ROOT/'submission_outputs';dest.mkdir(exist_ok=True)
    if any(dest.iterdir()):raise RuntimeError('submission_outputs must be empty')
    shutil.copyfile(ROOT/'runs'/name/'submission.json',dest/'blanket.json')
    for filename in ['solve_blanket.py','local_eval.py','build_report.py']:
        shutil.copyfile(ROOT/filename,dest/filename)
    shutil.copyfile(ROOT/'runtime/native_manifest.json',dest/'runtime_manifest.json')
    for filename in ['details.json','reward.json']:
        shutil.copyfile(ROOT/'results'/name/filename,dest/filename)
        shutil.copyfile(ROOT/'results/uniform-full'/filename,dest/('baseline_'+filename))
    comparisons={}
    for source,measured in baseline['outcomes'].items():
        comparisons[source]={
            'frozen_mean':frozen['cases'][source]['mean'],'reproduced_mean':measured['mean'],
            'difference':[a-b for a,b in zip(measured['mean'],frozen['cases'][source]['mean'])],
            'candidate_yield_mean':details['outcomes'][source]['mean'][0],
            'candidate_yield_lower':details['outcomes'][source]['lower'][0],
            'max_upper_flux_to_limit_ratio':max(v/b for v,b in zip(details['outcomes'][source]['upper'][1:],frozen['cases'][source]['screening_limits']))}
    history=[]
    for p in sorted((ROOT/'search').glob('*/result.json')):
        r=read(p);history.append({'name':p.parent.name,'particles':r['particles_per_batch'],'batches':r['batches'],
                                 'outcomes':r['outcomes'],'elapsed_seconds':r['elapsed_seconds'],'seeds':r['seeds'],'design':r['design']})
    report={'protocol':'workspace calibrated fusionx-n0 v1, explicitly authorized by user',
            'runtime_architecture':platform.machine(),'threads':4,'cpu_affinity':[8,9,10,11],
            'official_evaluator_sha256':digest(ROOT/'workspace/eval.py'),
            'official_case_sha256':digest(ROOT/'workspace/case.json'),
            'local_adapter_scope':'Only immutable-container path/fingerprint checks adapted; original prepare_model, require_data_coverage, batch_observations, summarize_replicates, score_result, main execute unchanged.',
            'baseline_reward':baseline_reward,'candidate_reward':reward,
            'objective_lower':details['objective_lower'],
            'relative_yield_lower_improvement':details['objective_lower']/frozen['objective_lower']-1,
            'baseline_comparisons':comparisons,'search_history':history,
            'inventory_margin_fraction':{k:1-details['inventory'][k]/frozen['inventory_limits'][k] for k in details['inventory']},
            'completed_search_trials':len(history),
            'search_source_histories':sum(h['particles']*h['batches']*len(h['seeds'])*len(h['outcomes']) for h in history),
            'replay_command':'source runtime/environment.sh; taskset -c 8-11 runtime/native-env/bin/python -I local_eval.py blanket.json --workdir runs/replay-clean --reward-dir results/replay-clean --threads 4',
            'blanket_sha256':digest(dest/'blanket.json'),
            'limitation':'Local validation is a native ARM64 rebuild; platform validation uses its independent pristine runtime and hidden seed pair.'}
    (dest/'evaluation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['candidate_reward','objective_lower','relative_yield_lower_improvement']},indent=2))
if __name__=='__main__':main()
