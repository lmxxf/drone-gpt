#!/usr/bin/env python3
"""Independent magnetic measurements; never changes official verification gates."""
import os
os.sched_setaffinity(0,{16,17,18,19})
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,json,sys,time,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'workspace'))
import magnetic_eval as m

def write(p,r):p.write_text(json.dumps(r,indent=2,default=str,allow_nan=False)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('boundary',type=Path);ap.add_argument('--workdir',type=Path,required=True);ap.add_argument('--backend',choices=['native','vmecpp'],default='native');ap.add_argument('--skip-alpha',action='store_true');a=ap.parse_args()
    folder=a.workdir.resolve();folder.mkdir(parents=True,exist_ok=True)
    r={'backend':a.backend,'official_feasible':None,'full_constraints_pass':None,'stages':{}}
    started=time.time()
    try:
        b=m.read_boundary(a.boundary);g=m.geometry(b);b=b.scaled(1.7/g['minor_radius']);g=m.geometry(b)
        r['stages']['geometry']=g
        r['volume_relative_error']=abs(g['volume']/581.8735601800271-1)
        if a.backend=='native':
            m.VMEC_NS_FINAL=(257,)
            v=m.run_vmec(b,folder,shutil.which('xvmec2000'),14400)
        else:
            import solve_turbulence as solver
            _,w,wout,res=solver.equilibrium(json.loads(a.boundary.read_text()),folder,True)
            v=dict(wout=wout,wout_data=w,residuals=res,ns=257)
        r['stages']['vmec']={k:v[k] for k in ('residuals','ns')}
        w=v['wout_data'];wout=v['wout']
        if a.backend=='vmecpp':
            import numpy as np
            jac=m._cos_series(w['gmnc'][1:],w['xm_nyq'],w['xn_nyq'],*m._period_grid(2*m.VMEC_NTHETA,2))
            r['jacobian_sign_pass']=bool(np.all(np.sign(jac)==np.sign(w['signgs'])))
            if not r['jacobian_sign_pass']:raise ValueError('Jacobian changes sign')
            del jac
        r['wout']=str(wout);r['wout_sha256']=hashlib.sha256(wout.read_bytes()).hexdigest()
        r['stages']['profile']=m.profile_metrics(w)
        r['stages']['coil']=m.coil_room(wout,2)
        write(folder/'measurement.json',r)
        bx=m.run_boozer(wout,2,folder)
        r['stages']['boozer']={k:bx[k] for k in ('B00_axis','e_B','history','s_neo')}
        write(folder/'measurement.json',r)
        r['stages']['neo']=m.run_neo(bx,folder,shutil.which('xneo'),7200)
        r['simple_memory_gib']=m.simple_memory_gib(w)
        write(folder/'measurement.json',r)
        if not a.skip_alpha:
            alpha=m.run_simple(wout,m.alpha_starts(w,2),5.7/bx['B00_axis'],folder,shutil.which('simple.x'),4,43200)
            alpha['wilson_95']=[m.wilson(alpha['N_L'],m.N_ALPHA)[0],m.wilson(alpha['N_L']+alpha['N_U'],m.N_ALPHA)[1]]
            r['stages']['alpha']=alpha
            metrics=dict(volume_relative_error=r['volume_relative_error'],f_alpha_upper=alpha['wilson_95'][1],eps_eff_max=r['stages']['neo']['eps_eff_max'],iota_min=r['stages']['profile']['iota_min'],well=r['stages']['profile']['well'],kappa_max=g['kappa_max'],lgradb_min=r['stages']['coil']['lgradb_min'])
            sys.path.insert(0,str(ROOT/'workspace'));import eval as ev
            case=json.loads((ROOT/'workspace/case.json').read_text());r['metrics']=metrics;r['violations']=ev.violations(metrics,case['constraints'])
            r['full_constraints_pass']=all(x==0 for x in r['violations'].values()) and alpha['N_U']/m.N_ALPHA<=.001 and bx['e_B']<=1e-10
        r['status']='measured'
    except Exception as exc:r.update(status='failed',error=f'{type(exc).__name__}: {exc}')
    r['seconds']=time.time()-started;write(folder/'measurement.json',r);print(json.dumps(r,default=str),flush=True)
if __name__=='__main__':main()
