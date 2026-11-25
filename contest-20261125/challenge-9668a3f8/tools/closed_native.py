"""Causal, delayed temperature-rate governor inside the maintained native solver scan."""
import os
os.environ['MPLBACKEND']='Agg';os.environ['JAX_ENABLE_X64']='true';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,functools,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));sys.path.insert(0,str(ROOT/'tools'))
import jax,jax.numpy as jnp
import plant
from fast_native import NativeScan
from torax._src.orchestration.step_function import SimulationStepFn
from torax._src.torax_pydantic.interpolated_param_1d import TimeVaryingScalarUpdate

@functools.partial(jax.jit,static_argnames=('steps','R','native','ped','kind'))
def closed_scan(sf,s0,p0,dt,indices,lo,hi,noise,coarse,fine,parameters,*,steps,R,native,ped,kind='reactive'):
    initial=jnp.array([28.,15.],jnp.float32);qmid=jnp.float32((1.8337676430354448+1.8279795705948279)/2)
    carry=(s0,p0,s0.core_profiles.T_e.value[0],p0.q_min,jnp.float32(s0.core_profiles.T_e.value[0]),jnp.float32(0),jnp.float32(0),jnp.float32(0),jnp.float32(15),initial,initial,initial,initial.astype(jnp.float64),initial.astype(jnp.float64),jnp.float32(0),jnp.float32(0),jnp.float32(15),jnp.float32(15),jnp.float32(15))
    def body(carry,i):
        s,p,old_te,old_q,tef,qs,qc,ug,old_ff,cmd,applied,queued,start,end,weight,dy_old,uf2,uf3,uf4=carry;j=i%R;k=i//R
        controls=(old_te,old_q,tef,qs,qc,ug,old_ff,cmd,applied,queued,start,end,weight,dy_old,uf2,uf3,uf4)
        def update(v):
            old_te,old_q,tef,qs,qc,ug,old_ff,cmd,applied,queued,start,end,weight,dy_old,uf2,uf3,uf4=v
            q=jnp.float32(old_q*(1+noise[k,5]));qs=jnp.where(k<15,qs+(q-qmid)*jnp.float32(100),qs);qc=jnp.where(k<15,qc+jnp.float32(1),qc);weight=jax.nn.sigmoid(jnp.float32(40)*qs/jnp.maximum(qc,jnp.float32(1)))
            ff=fine[k]+weight*(coarse[k]-fine[k]);te=jnp.float32(old_te*(1+noise[k,2]));new_tef=jnp.where(k<3,te,jnp.float32(.5)*tef+jnp.float32(.5)*te);rate=jnp.where(k<3,jnp.float32(0),(new_tef-tef)/jnp.float32(.5));cooling=ff[1]<old_ff-jnp.float32(.001)
            dy=new_tef-tef;uf1=jnp.float32(.5)*cmd[1]+jnp.float32(.5)*uf2
            if kind=='predictive':
                a1=jnp.float32(.5614564596843078);a2=jnp.float32(-.19290482522123412);b1=jnp.float32(.06871025720897991);b2=jnp.float32(.005700036016133524)
                du1=uf1-uf2;du2=uf2-uf3;du3=uf3-uf4
                d0=a1*dy+a2*dy_old+b1*du2+b2*du3
                d1=a1*d0+a2*dy+b1*du1+b2*du2
                prediction=a1*d1+a2*d0+b1*jnp.float32(.5)*(ff[1]-uf1)+b2*du1
                need=jnp.maximum((-parameters[1]*jnp.float32(.5)-prediction)/(b1*jnp.float32(.5)),0)
                target_ug=jnp.clip(parameters[0]*need,0,parameters[2])*(1-weight)
                new_ug=jnp.where((k>=8)&(k<=40),target_ug,jnp.float32(0))
            else:
                target_ug=jnp.where(cooling,jnp.clip(parameters[0]*jnp.maximum(-rate-parameters[1],0),0,parameters[2])*(1-weight),jnp.float32(0));new_ug=jnp.float32(.5)*ug+jnp.float32(.5)*target_ug
            target=ff+jnp.array([0.,new_ug],jnp.float32);new_cmd=jnp.clip(cmd+jnp.clip(target-cmd,jnp.float32(-1.999),jnp.float32(1.999)),jnp.array([0.,0.]),jnp.array([33.,20.])).astype(jnp.float32)
            return (s.core_profiles.T_e.value[0],p.q_min,new_tef,qs,qc,new_ug,ff[1],new_cmd,queued,new_cmd,applied.astype(jnp.float64),queued.astype(jnp.float64),weight,dy,uf1,uf2,uf3)
        controls=jax.lax.cond(j==0,update,lambda x:x,controls)
        old_te,old_q,tef,qs,qc,ug,old_ff,cmd,applied,queued,start,end,weight,dy_old,uf2,uf3,uf4=controls
        power0=start+(j.astype(jnp.float64)/R)*(end-start);power1=start+((j.astype(jnp.float64)+1)/R)*(end-start);times=jnp.stack([s.t,s.t+dt]);rp=sf.runtime_params_provider.update_provider_from_mapping({'sources.generic_heat.P_total':TimeVaryingScalarUpdate(time=times,value=jnp.stack([power0[0],power1[0]])*1e6),'sources.ecrh.P_total':TimeVaryingScalarUpdate(time=times,value=jnp.stack([power0[1],power1[1]])*1e6)})
        nsf=SimulationStepFn(solver=sf.solver,time_step_calculator=sf.time_step_calculator,runtime_params_provider=rp,geometry_provider=sf.geometry_provider);ns,npst=nsf(s,p,max_dt=dt);runtime=rp(ns.t);vals=plant.native_model_inputs(native,runtime.transport,ns.geometry,ns.core_profiles,runtime.neoclassical.poloidal_velocity_multiplier);v=vals[indices];bad=jnp.any((v<lo)|(v>hi)|~jnp.isfinite(v));ti=ns.core_profiles.T_i.value;te=ns.core_profiles.T_e.value
        d=jnp.stack([ns.t,npst.P_fusion,npst.W_thermal_total,te[0],ti[0],npst.beta_N,npst.q_min,npst.q95,npst.P_external_total,npst.E_external_total,ped(ti,te),jnp.maximum(jnp.max(ti),jnp.max(te)),ns.solver_numeric_outputs.solver_error_state,bad.astype(jnp.float64),jnp.abs(ns.t-s.t-dt),cmd[0],cmd[1],ug,tef,weight])
        return (ns,npst,*controls),d
    return jax.lax.scan(body,carry,jnp.arange(steps))

def run(s,noise,coarse,fine,parameters,kind='reactive'):
    base=s.env.torax_app.step_fn;provider=base.runtime_params_provider.update_provider_from_mapping({'numerics.t_final':s.horizon});step=SimulationStepFn(solver=base.solver,time_step_calculator=base.time_step_calculator,runtime_params_provider=provider,geometry_provider=base.geometry_provider)
    t=time.time();carry,d=closed_scan(step,s.env.torax_app.current_sim_state,s.env.torax_app.current_sim_output,jnp.asarray(s.dt),s.indices,s.lo,s.hi,jnp.asarray(noise),jnp.asarray(coarse,np.float32),jnp.asarray(fine,np.float32),jnp.asarray(parameters,np.float32),steps=int(round(s.horizon/s.dt)),R=int(round(.5/s.dt)),native=s.native,ped=s.env._native_pedestal_integral,kind=kind);d=np.asarray(d)
    assert np.isfinite(d).all();assert np.all(d[:,12:14]==0),'native solver/domain error';assert max(d[:,14])<1e-8;assert abs(d[-1,0]-s.horizon)<1e-8
    return d,time.time()-t
if __name__=='__main__':
    import eval as ev
    from library_search import replay_metrics
    level=sys.argv[1] if len(sys.argv)>1 else 'fine';horizon=float(sys.argv[2]) if len(sys.argv)>2 else 2.;seed=17;case=ev.read_json(ROOT/'workspace/case.json');physical=ev.scenario(seed,case);physical.update(event_start_s=8.,Z_eff_after_event=1.75,initial_temperature_multiplier=.98);s=NativeScan(physical,level,horizon);noise=np.clip(np.random.Generator(np.random.PCG64(physical['sensor_seed'])).standard_normal((60,9)),-3,3)*np.array(case['observation']['noise_relative_sigma']);c=np.array(json.load(open(ROOT/'runs/library/coarse-e5.0-h5.0.json'))['commands_MW'],np.float32);f=np.array(json.load(open(ROOT/'runs/library/fine-e3.5-h6.0.json'))['commands_MW'],np.float32);d,seconds=run(s,noise,c,f,[0.,.4,2.]);R=int(round(.5/s.dt));issued=d[::R,15:17];issued=np.vstack([issued,np.tile(issued[-1],(60-len(issued),1))]);reference,seconds2=s.run(np.vstack(([28.,15.],[28.,15.],issued[:59])));error=float(np.max(abs(d[:,1]-reference[:,1])/reference[:,1]));r={'level':level,'horizon_s':horizon,'closed_wall_s':seconds,'replay_wall_s':seconds2,'max_relative_fusion_difference':error,'passed':error<1e-7};print(json.dumps(r),flush=True);(ROOT/'runs/closed-native-parity.json').write_text(json.dumps(r,indent=2));s.close()
