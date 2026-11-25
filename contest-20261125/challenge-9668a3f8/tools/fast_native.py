"""Offline native TORAX scan. Search only; final candidates use unchanged eval.py episodes.
Every native solver step, model-domain check and pedestal heat diagnostic is retained.
"""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,time,functools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'))
import eval as ev
import plant,jax,jax.numpy as jnp
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.torax_pydantic.interpolated_param_1d import TimeVaryingScalarUpdate
from torax._src.transport_model import qlknn_transport_model
jax.config.update('jax_compilation_cache_dir',str(ROOT/'runs/jax-cache'));jax.config.update('jax_persistent_cache_min_compile_time_secs',0)

@functools.partial(jax.jit,static_argnames=('steps','native','ped'))
def scan(sf,state,post,dt,indices,lo,hi,*,steps,native,ped):
    def body(carry,_):
        old,pp=carry;s,p=sf(old,pp,max_dt=dt);rp=sf.runtime_params_provider(s.t)
        vals=plant.native_model_inputs(native,rp.transport,s.geometry,s.core_profiles,rp.neoclassical.poloidal_velocity_multiplier)
        v=vals[indices];domain_bad=jnp.any((v<lo)|(v>hi)|~jnp.isfinite(v))
        ti=s.core_profiles.T_i.value;te=s.core_profiles.T_e.value
        d=jnp.stack([s.t,p.P_fusion,p.W_thermal_total,te[0],ti[0],p.beta_N,p.q_min,p.q95,p.P_external_total,p.E_external_total,ped(ti,te),jnp.maximum(jnp.max(ti),jnp.max(te)),s.solver_numeric_outputs.solver_error_state,domain_bad.astype(jnp.float64),jnp.abs(s.t-old.t-dt)])
        return (s,p),d
    return jax.lax.scan(body,(state,post),None,length=steps)

class NativeScan:
    def __init__(self,scenario,level,horizon=30.):
        self.env=plant.CoreHeatEnv(ROOT/'workspace',scenario,level);self.env.reset();self.dt=self.env.native_dt;self.horizon=horizon;self.level=level
        plant.native_pedestal_power(self.env)
        self.native=self.env.torax_app.step_fn.solver.models.transport_model;model=qlknn_transport_model.get_model(self.native.path,self.native.name)
        rho=np.asarray(self.env.torax_app.current_sim_state.geometry.rho_face_norm);self.indices=jnp.asarray(np.flatnonzero((rho>.2+1e-7)&(rho<.9-1e-7)))
        ranges=[self.env.core_domain[n] for n in model.inputs_and_ranges];self.lo=jnp.asarray([x['min'] if x['min'] is not None else -np.inf for x in ranges]);self.hi=jnp.asarray([x['max'] if x['max'] is not None else np.inf for x in ranges])
        assert not any(x['outside_faces'] for x in plant.transport_domain(self.env).values())
    def run(self,endpoints):
        sf=self.env.torax_app.step_fn;times=jnp.arange(len(endpoints),dtype=jnp.float64)*.5;endpoints=jnp.asarray(endpoints,dtype=jnp.float64)*1e6
        rp=sf.runtime_params_provider.update_provider_from_mapping({'sources.generic_heat.P_total':TimeVaryingScalarUpdate(time=times,value=endpoints[:,0]),'sources.ecrh.P_total':TimeVaryingScalarUpdate(time=times,value=endpoints[:,1]),'numerics.t_final':self.horizon})
        sf=SimulationStepFn(solver=sf.solver,time_step_calculator=sf.time_step_calculator,runtime_params_provider=rp,geometry_provider=sf.geometry_provider)
        start=time.time();(state,post),diag=scan(sf,self.env.torax_app.current_sim_state,self.env.torax_app.current_sim_output,jnp.asarray(self.dt),self.indices,self.lo,self.hi,steps=int(round(self.horizon/self.dt)),native=self.native,ped=self.env._native_pedestal_integral)
        d=np.asarray(diag);seconds=time.time()-start
        assert np.isfinite(d).all();assert np.all(d[:,12]==0),'native solver failure';assert np.all(d[:,13]==0),'native model-domain failure';assert np.max(d[:,14])<1e-8,'native time-step failure';assert abs(d[-1,0]-self.horizon)<1e-8
        return d,seconds
    def close(self):self.env.close()

if __name__=='__main__':
    level=sys.argv[1] if len(sys.argv)>1 else 'coarse';horizon=float(sys.argv[2]) if len(sys.argv)>2 else 2.;seed=2026100801
    case=ev.read_json(ROOT/'workspace/case.json');s=NativeScan(ev.scenario(seed,case),level,horizon);endpoints=np.tile([28.,15.],(61,1));d,seconds=s.run(endpoints)
    base=[json.loads(x) for x in (ROOT/'runs/baseline-local'/str(seed)/level/'trace.jsonl').read_text().splitlines()];stride=int(round(.5/s.dt));errs=[]
    for k in range(1,int(round(horizon/.5))+1):
        row=d[k*stride-1];r=base[k];errs.append(abs(row[1]-r['P_fusion_W'])/r['P_fusion_W'])
    report={'level':level,'horizon_s':horizon,'wall_time_s':seconds,'maximum_relative_fusion_difference':max(errs),'native_steps':len(d),'native_solver_and_domain_checks_passed':True};print(json.dumps(report),flush=True)
    if horizon==30.:
        actions=json.load(open(ROOT/'runs/anchor-local'/str(seed)/level/'actions.json'));cmd=np.array([x['command_MW'] for x in actions]);curve=np.vstack(([28.,15.],[28.,15.],cmd[:59]));da,wa=s.run(curve);anchor=[json.loads(x) for x in (ROOT/'runs/anchor-local'/str(seed)/level/'trace.jsonl').read_text().splitlines()];err=max(abs(da[k*stride-1,1]-anchor[k]['P_fusion_W'])/anchor[k]['P_fusion_W'] for k in range(1,61));report['cached_anchor_wall_time_s']=wa;report['anchor_maximum_relative_fusion_difference']=err;print(json.dumps({'cached_anchor_wall_time_s':wa,'anchor_maximum_relative_fusion_difference':err}),flush=True)
    np.save(ROOT/'runs'/f'fast-baseline-{level}.npy',d);(ROOT/'runs'/f'fast-baseline-{level}.json').write_text(json.dumps(report,indent=2));s.close()
