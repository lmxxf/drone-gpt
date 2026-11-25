import os,sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'workspace'))
from fast_validate_pair import job
import eval as ev
if __name__=='__main__':
    out=ROOT/'runs/options-heldout';out.mkdir(exist_ok=False)
    c=ROOT/'runs/library/coarse-e5.0-h5.0.json'
    fs={'zvd':ROOT/'runs/shaping/fine-e3.5-h5.5-order2.json','plateau':ROOT/'runs/staging/fine-h6.0-p8.0-d1.0.json','plain':ROOT/'runs/library/fine-e3.8-h6.0.json','zv38':ROOT/'runs/shaping/fine-e3.8-h6.0-order1.json'}
    worlds=[('dev2',2026100802,None),('mid',17,{'event_start_s':12.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98})]
    args=[(tag+'-'+name,seed,over,level,str(c),str(f),str(out)) for level in ['fine','coarse'] for tag,f in fs.items() for name,seed,over in worlds]
    with ProcessPoolExecutor(max_workers=10,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(job,a) for a in args]):print(json.dumps(future.result()),flush=True)
    case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];results=[]
    for tag in fs:
        for name,_,_ in worlds:
            key=tag+'-'+name;rows=[json.load(open(out/(key+'-'+level+'.json'))) for level in ['coarse','fine']];r={'option':tag,'world':name}
            if any(x['state']!='native_complete' or x.get('violation') for x in rows):r.update(feasible=False,rows=rows)
            else:
                try:J=ev.compare_resolutions(case,ref,[(x['metrics'],x['violation'],x['trace']) for x in rows]);r.update(feasible=True,J_upper_J=J)
                except ev.Unresolved as e:r.update(feasible=False,reason=str(e))
            results.append(r)
    (out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results),flush=True)
