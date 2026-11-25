"""Deterministic initial-temperature scan of the shared full-power startup."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'))
def job(arg):
    temperature,level=arg
    import plant,jax
    jax.config.update('jax_compilation_cache_dir',str(ROOT/'runs/jax-cache'));jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
    env=plant.CoreHeatEnv(ROOT/'workspace',{'event_start_s':12.,'Z_eff_after_event':1.6,'initial_temperature_multiplier':temperature},level,horizon=3.)
    env.reset();trace=[plant.diagnostics(env)];cmd=np.array([28.,15.]);queued=cmd.copy()
    for k in range(6):
        new=cmd+np.clip(np.array([33.,20.])-cmd,-2,2);env.advance(queued);cmd=new;queued=new.copy();trace.append(plant.diagnostics(env))
    env.close();return temperature,level,trace
if __name__=='__main__':
    rows={};results=[]
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(job,(float(t),r)) for t in np.linspace(.98,1.02,9) for r in ['coarse','fine']]):
            t,r,trace=f.result();rows.setdefault(str(t),{})[r]=trace
            if len(rows[str(t)])==2:
                c=rows[str(t)]['coarse'];fi=rows[str(t)]['fine'];gap=max(abs(x['P_fusion_W']-y['P_fusion_W']) for x,y in zip(c,fi));results.append({'initial_multiplier':t,'max_grid_gap_W':gap});print(json.dumps(results[-1]),flush=True)
            (ROOT/'runs/startup-check.json').write_text(json.dumps({'pairs':results,'traces':rows},indent=2))
