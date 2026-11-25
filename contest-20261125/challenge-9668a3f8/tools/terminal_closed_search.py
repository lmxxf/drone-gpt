import os,sys,json,multiprocessing,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'workspace'))
from block_eval import job
import eval as ev
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--tags',nargs='+',default=['gentle','gradual','relaxed']);parser.add_argument('--out',default='runs/terminal-closed');args=parser.parse_args();out=(ROOT/args.out).resolve();out.mkdir(exist_ok=False);worlds=[('dev1',2026100801,None),('dev3',2026100803,None),('cold8',17,{'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98})];tags=args.tags;args=[(tag+'-'+name,seed,o,level,str(ROOT/'runs'/(tag+'.onnx')),str(out)) for level in ['fine','coarse'] for tag in tags for name,seed,o in worlds]
 with ProcessPoolExecutor(max_workers=9,mp_context=multiprocessing.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(job,a) for a in args]):print(json.dumps(f.result()),flush=True)
 case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];rows=[]
 for tag in tags:
  for name,_,_ in worlds:
   rs=[json.load(open(out/(tag+'-'+name+'-'+l+'.json'))) for l in ['coarse','fine']];r={'tag':tag,'world':name,'violation':[s['violation'] for s in rs]}
   try:r.update(feasible=not any(r['violation']),J=ev.compare_resolutions(case,ref,[(s['metrics'],s['violation'],s['trace']) for s in rs]))
   except ev.Unresolved as e:r.update(feasible=False,reason=str(e))
   rows.append(r)
 (out/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows),flush=True)
