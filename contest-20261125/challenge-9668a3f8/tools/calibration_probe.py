import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import plant
import jax
jax.config.update('jax_compilation_cache_dir',str(ROOT/'runs/jax-cache'));jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
level=sys.argv[1];s={'event_start_s':12.,'Z_eff_after_event':1.6,'initial_temperature_multiplier':1.0}
env=plant.CoreHeatEnv(ROOT/'workspace',s,level,horizon=7.5);env.reset();trace=[plant.diagnostics(env)];command=np.array([28.,15.]);queued=command.copy()
for k in range(15):
    next_command=command+np.clip(np.array([33.,4.])-command,-2,2)
    env.advance(queued);queued=next_command.copy();command=next_command.copy()
    d=plant.diagnostics(env);d['PG'],d['PE']=env.previous_power.tolist();trace.append(d)
    (ROOT/'runs'/f'calibration-{level}.json').write_text(json.dumps(trace,indent=2));print(json.dumps(d),flush=True)
env.close()
