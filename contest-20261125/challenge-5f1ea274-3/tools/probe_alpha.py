"""Small, explicitly non-certifying SIMPLE timing/confinement probe."""
import os
os.sched_setaffinity(0,{12,13,14,15});os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
import argparse,json,pathlib,sys,time,numpy as np,booz_xform
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev
ap=argparse.ArgumentParser();ap.add_argument('wout',type=pathlib.Path);ap.add_argument('--particles',type=int,default=64);ap.add_argument('--time',type=float,default=.002);ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args()
w=ev._wout(a.wout);b=booz_xform.Booz_xform();b.verbose=0;b.read_wout(str(a.wout));b.compute_surfs=[0,1];b.mboz=32;b.nboz=24;b.run();m=np.array(b.xm_b);n=np.array(b.xn_b);row=np.array(b.bmnc_b)[(m==0)&(n==0)][0];baxis=1.5*row[0]-.5*row[1]
ev.N_ALPHA=a.particles;ev.ALPHA_TIME=a.time
a.out.mkdir(parents=True,exist_ok=False)
r=ev.run_simple(a.wout,ev.alpha_starts(w,int(w['nfp'])),float(ev.AXIS_FIELD/baxis),a.out,str(ROOT/'tools/bin/simple.x'),4,1800)
r.update(full_verified=False,particle_count=a.particles,trace_time=a.time,radial_surfaces=int(w['ns']),purpose='timing and early-loss screening only; neither a full-time nor full-resolution acceptance test')
(a.out/'probe.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
