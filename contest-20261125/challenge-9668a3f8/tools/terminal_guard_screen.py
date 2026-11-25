import sys,json,multiprocessing,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));from block_eval import job
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--candidate',default='runs/terminal_guard.onnx');parser.add_argument('--out',default='runs/terminal-guard-screen');a=parser.parse_args();out=(ROOT/a.out).resolve();out.mkdir(exist_ok=False);candidate=(ROOT/a.candidate).resolve();stress=json.load(open(ROOT/'runs/early_reserve-stress/stress11-coarse.json'));worlds=[('cold8',17,{'event_start_s':8.,'Z_eff_after_event':1.75,'initial_temperature_multiplier':.98}),('stress11',stress['seed'],None)]
 with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
  for f in as_completed([pool.submit(job,(n,s,o,'coarse',str(candidate),str(out))) for n,s,o in worlds]):print(json.dumps(f.result()),flush=True)
