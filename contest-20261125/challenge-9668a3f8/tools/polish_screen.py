import os,sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'workspace'))
from fast_validate_pair import job
import eval as ev
if __name__=='__main__':
 out=ROOT/'runs/polish-screen';out.mkdir(exist_ok=False);c=ROOT/'runs/polished-coarse-source.json';f=ROOT/'runs/staging/fine-h6.5-p8.0-d0.5.json';worlds=[(f'dev{i+1}',s,None) for i,s in enumerate([2026100801,2026100802,2026100803,2026100804])]+[('lowZ',20261125,None),('mid_event',17,{'event_start_s':12.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98}),('late_event',991,{'event_start_s':16.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98}),('cold8',17,{'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98})]
 with ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context('spawn')) as pool:
  for fut in as_completed([pool.submit(job,(n,s,o,'coarse',str(c),str(f),str(out))) for n,s,o in worlds]):print(json.dumps(fut.result()),flush=True)
