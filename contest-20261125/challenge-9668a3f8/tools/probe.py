import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'))
import plant
import jax
jax.config.update('jax_compilation_cache_dir',str(ROOT/'runs/jax-cache'));jax.config.update('jax_persistent_cache_min_compile_time_secs',0)
target=np.array([float(sys.argv[1]),float(sys.argv[2])]);out=ROOT/'runs'/f'probe-{target[0]}-{target[1]}.json'
s={'event_start_s':12.,'Z_eff_after_event':1.6,'initial_temperature_multiplier':1.0}
env=plant.CoreHeatEnv(ROOT/'workspace',s,'coarse',horizon=12.);env.reset();trace=[plant.diagnostics(env)];power=np.array([28.,15.])
for k in range(24):
    power+=np.clip(target-power,-2,2);env.advance(power)
    d=plant.diagnostics(env);d['PG'],d['PE']=power.tolist();trace.append(d)
    out.write_text(json.dumps(trace,indent=2));print(json.dumps(d),flush=True)
env.close()
