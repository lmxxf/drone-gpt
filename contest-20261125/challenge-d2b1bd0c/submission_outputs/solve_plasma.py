#!/usr/bin/env python3
"""Native, cached seven-variable constrained search; formal evaluation certifies incumbents.

Run with `source env.sh; taskset -c 4-7 python solve_plasma.py ...`.
Screening uses the prescribed coarse equilibrium and selected native DCON modes,
not a feasibility certificate. No numerical failure is assigned a physical margin.
"""
import os
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
os.sched_setaffinity(0, {4,5,6,7})
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
ROOT = Path(__file__).resolve().parent
if not (ROOT/'workspace/physics/eval.py').exists():ROOT=ROOT.parent
if not (ROOT/'workspace/physics/eval.py').exists():raise RuntimeError('Run inside the original challenge directory, with workspace/physics present')
sys.path.insert(0, str(ROOT/'workspace/physics'))
import eval as ev
P = json.loads((ROOT/'workspace/physics/assets/protocol.json').read_text())
KEYS = list(P['shape_bounds']) + ['axis_pressure_Pa']
BOUNDS = list(P['shape_bounds'].values()) + [P['axis_pressure_Pa_bounds']]
BINARY = ROOT/'native/GPEC/bin/dcon'

def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)

def design(x):
    return dict(zip(KEYS, map(float, x)))

def screen(x, modes=(1,5), scenarios=('broad',), keep=False, level='coarse'):
    d = design(x)
    cache_input=[d,list(modes),list(scenarios)]
    if level!='coarse':cache_input.append(level)
    key = hashlib.sha256(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()[:16]
    work = ROOT/'runs/screen'/key
    report = work/'screen.json'
    if report.exists():
        cached=json.loads(report.read_text())
        if cached['status']!='invalidated_blas':return cached
        work.rename(work.with_name(work.name+'-rejected-'+str(time.time_ns())))
    work.mkdir(parents=True, exist_ok=True)
    save(work/'plasma.json', d)
    out = {'design':d, 'screen_only':True, 'level':level, 'cases':{}, 'status':'pending'}
    start = time.monotonic()
    try:
        ev.read_design(work/'plasma.json', P)
        for scenario in P['scenarios']:
            if scenario['name'] not in scenarios: continue
            case = work/scenario['name']; case.mkdir()
            job = {k:P[k] for k in ('R0_m','B0_T','volume_m3','current_A')}
            job.update(shape={k:d[k] for k in P['shape_bounds']}, axis_pressure_Pa=d['axis_pressure_Pa'], resolution=P[level], ffprime_hollow=scenario['ffprime_hollow'])
            save(case/'job.json', job)
            ev.execute([sys.executable,'-I',str(ROOT/'workspace/physics/equilibrium.py'),str(case/'job.json'),str(case/'equilibrium')],case/'equilibrium.log',900)
            eq=json.loads((case/'equilibrium/report.json').read_text()); ev.check_equilibrium(eq,P,d['axis_pressure_Pa'])
            command=[sys.executable,'-I',str(ROOT/'workspace/physics/dcon.py'),str(case/'equilibrium/equilibrium.geqdsk'),str(case/'modes'),'--binary',str(BINARY),'--template',str(ROOT/'workspace/physics/assets/templates'),'--normalize-ballooning','--refinement',str(1 if level=='coarse' else 2),'--ballooning-rtol',str(P[level]['ballooning_rtol']),'--poloidal-padding',str(P[level]['poloidal_padding']),'--vacuum-points',str(P[level]['vacuum_points']),'--ballooning-extent',str(P[level]['ballooning_extent']),'--ballooning-cap',str(P[level]['ballooning_cap']),'--jacobian','pest','--modes',*map(str,modes)]
            ev.execute(command,case/'modes.log',3600)
            stability=json.loads((case/'modes/report.json').read_text())
            violations=ev.check_stability(stability,P,modes=list(modes))
            out['cases'][scenario['name']]={'equilibrium':eq,'stability':stability,'violations':violations}
        entries=[m for c in out['cases'].values() for m in c['stability']['modes'].values()]
        out['margins']={'energy':min(m['minimum_energy'][0]-.02 for m in entries),'q':min(m['globals']['qmin']-1.1 for m in entries),'mercier':min(-.01-m['mercier_di_max'] for m in entries),'nodes':-max(m['ballooning_nodes_max'] for m in entries),'crossings':-max(m['internal_zero_crossings'] for m in entries)}
        out['pressure']=min(c['equilibrium']['average_pressure_Pa'] for c in out['cases'].values())
        out['status']='pass' if all(not c['violations'] for c in out['cases'].values()) else 'infeasible'
    except (ev.InvalidDesign, ev.NumericalFailure, OSError, ValueError, KeyError) as exc:
        out['status']=type(exc).__name__;out['reason']=str(exc)
    window=ROOT/'runs/openblas-window.json'
    if window.exists():
        w=json.loads(window.read_text())
        touched=[p for pattern in ('*/job.json','*/modes/*/dcon.in') for p in work.glob(pattern) if w['start']<=p.stat().st_mtime<=w['end']]
        if touched:out['status']='invalidated_blas';out['invalidated_library_experiment']=True
    out['seconds']=time.monotonic()-start
    save(report,out)
    with (ROOT/'runs/search_history.jsonl').open('a') as f:f.write(json.dumps(out,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='cases'}),flush=True)
    if not keep:
        for case in work.iterdir():
            if case.is_dir():
                for p in case.rglob('*'):
                    if p.is_file() and p.suffix not in ('.json','.log','.in','.dat'):p.unlink()
    return out


def equilibrium_probe(x):
    """Cheap native-equilibrium probe; q_axis is a screening upper bound, never a certificate."""
    d=design(x);key=hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()[:16]
    work=ROOT/'runs/equilibrium-probes'/key;report=work/'probe.json'
    if report.exists():return json.loads(report.read_text())
    work.mkdir(parents=True,exist_ok=False);save(work/'plasma.json',d)
    out={'design':d,'screen_only':True}
    try:
        ev.read_design(work/'plasma.json',P)
        job={k:P[k] for k in ('R0_m','B0_T','volume_m3','current_A')}
        job.update(shape={k:d[k] for k in P['shape_bounds']},axis_pressure_Pa=d['axis_pressure_Pa'],resolution=P['coarse'],ffprime_hollow=.2)
        save(work/'job.json',job)
        ev.execute([sys.executable,'-I',str(ROOT/'workspace/physics/equilibrium.py'),str(work/'job.json'),str(work/'equilibrium')],work/'equilibrium.log',900)
        eq=json.loads((work/'equilibrium/report.json').read_text());ev.check_equilibrium(eq,P,d['axis_pressure_Pa'])
        out.update(status='equilibrium_only',q_axis=eq['stats']['q_0'],pressure=eq['average_pressure_Pa'],equilibrium=eq)
    except (ev.InvalidDesign,ev.NumericalFailure,OSError,ValueError,KeyError) as exc:
        out.update(status=type(exc).__name__,reason=str(exc))
    save(report,out);print(json.dumps(out),flush=True)
    return out



def project_radial_envelope(d,minor_scale=1.4995):
    """Set kappa analytically on the fixed-volume radial-envelope manifold.

    For fixed a and the other coefficients, toroidal volume is linear in
    vertical scaling kappa. The public geometry verifier still checks every
    proposal before a native equilibrium is run.
    """
    from shapely.geometry import Polygon
    theta=np.linspace(0,2*np.pi,4096,endpoint=False)
    phase=theta+np.arcsin(d['delta'])*np.sin(theta)+d['c2']*np.sin(2*theta)+d['c3']*np.sin(3*theta)
    z=np.sin(theta)+d['s2']*np.sin(2*theta)+d['s3']*np.sin(3*theta)
    domain=Polygon(np.column_stack((4+minor_scale*np.cos(phase),minor_scale*z)))
    if not domain.is_valid or domain.area<=0:raise ValueError('Non-simple projected section')
    volume_at_unit_kappa=2*np.pi*domain.centroid.x*domain.area
    out=dict(d);out['kappa']=max(BOUNDS[0][0],P['volume_m3']/volume_at_unit_kappa)
    return out

def projected_scan(seed,workers=2):
    proposals={}
    for dc2 in (0.,-.0125,-.025,-.05,-.0875):
        for dc3 in (0.,-.0125):
            trial=dict(seed)
            trial['c2']=float(np.clip(seed['c2']+dc2,*P['shape_bounds']['c2']))
            trial['c3']=float(np.clip(seed['c3']+dc3,*P['shape_bounds']['c3']))
            trial=project_radial_envelope(trial)
            proposals[json.dumps(trial,sort_keys=True)]=trial
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs=[pool.submit(screen,[d[k] for k in KEYS],(1,),('broad','peaked')) for d in proposals.values()]
        results=[f.result() for f in jobs]
    save(ROOT/'runs/projected_scan_results.json',results)
    passing=[r for r in results if r['status']=='pass']
    if passing:save(ROOT/'runs/projected_scan_incumbent.json',max(passing,key=lambda r:r['pressure']))
    return passing

def pressure_boundary(d,step=20000.,tolerance=1000.):
    """Continue a fixed shape to a local q-limited boundary, then bracket it.

    This is deliberately not a global monotonic-feasibility assumption: it
    stops on a numerical failure or a different active constraint.
    """
    if step<=0 or tolerance<=0:raise ValueError('Positive step and tolerance required')
    x=np.array([d[k] for k in KEYS],float)
    def query(pressure):
        trial=x.copy();trial[-1]=pressure
        return screen(trial,modes=(1,),scenarios=('broad','peaked'))
    def q_failure(result):
        m=result.get('margins',{})
        return result['status']=='infeasible' and m.get('q',0)<0 and all(m.get(k,-1)>=0 for k in ('energy','mercier','nodes','crossings'))
    lo=float(x[-1]);passing=query(lo)
    if passing['status']!='pass':raise ValueError('A completed passing initial screen is required')
    hi=None;failing=None
    while lo<BOUNDS[-1][1]:
        point=min(lo+step,BOUNDS[-1][1]);r=query(point)
        if r['status']=='pass':lo=point;passing=r
        elif q_failure(r):hi=point;failing=r;break
        else:raise RuntimeError('Pressure continuation hit a numerical or non-q limiting event: '+r['status'])
    if hi is not None:
        while hi-lo>tolerance:
            point=round((hi+lo)/(2*tolerance))*tolerance
            if not lo<point<hi:break
            r=query(point)
            if r['status']=='pass':lo=point;passing=r
            elif q_failure(r):hi=point;failing=r
            else:raise RuntimeError('Local q bracket interrupted by another constraint: '+r['status'])
    x[-1]=lo
    save(ROOT/'runs/pressure_boundary_candidate.json',design(x))
    save(ROOT/'runs/pressure_boundary_report.json',{'screen_only':True,'passing':passing,'failing':failing,'bracket_Pa':[lo,hi],'tolerance_Pa':tolerance})
    return design(x)

def objective(x):
    r=screen(x)
    if 'margins' not in r: return 100.0
    m=r['margins']; penalty=sum(max(0,-m[k])*w for k,w in [('energy',10),('q',10),('mercier',10),('nodes',5),('crossings',5)])
    return -r['pressure']/200000 + penalty

def formal(d, label):
    path=ROOT/'runs'/f'{label}.json'
    if path.exists() and json.loads(path.read_text())!=d:
        raise ValueError('Formal label already belongs to a different design; choose a fresh --label')
    save(path,d)
    work=ROOT/'runs'/label
    if not (work/'details.json').exists():
        subprocess.run([sys.executable,'-I',str(ROOT/'workspace/physics/eval.py'),str(path),'--binary',str(BINARY),'--workdir',str(work),'--mode','formal'],check=True)
    result=json.loads((work/'details.json').read_text())
    if result['status']=='feasible':
        best=ROOT/'best_formal.json'
        with (ROOT/'runs/best_formal.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not best.exists() or result['reward']>json.loads(best.read_text())['reward']:
                save(best,{'design':d,'reward':result['reward'],'report':str(work/'details.json')})
                save(ROOT/'plasma.json',d)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--screen',type=Path);ap.add_argument('--modes',type=int,nargs='+',default=[1,5]);ap.add_argument('--scenarios',nargs='+',default=['broad']);ap.add_argument('--formal',type=Path);ap.add_argument('--label',default='optimized-formal');ap.add_argument('--de',type=int,default=0);ap.add_argument('--local',type=Path);ap.add_argument('--iterations',type=int,default=50);ap.add_argument('--latin',type=int,default=0);ap.add_argument('--seed',type=int,default=502);ap.add_argument('--climb',type=Path);ap.add_argument('--audit',type=int,default=0);ap.add_argument('--pressure-boundary',type=Path);ap.add_argument('--pressure-step',type=float,default=20000.);ap.add_argument('--workers',type=int,choices=[1,2],default=1);ap.add_argument('--expand',action='store_true');ap.add_argument('--projected-scan',type=Path);ap.add_argument('--resume-climb',action='store_true');ap.add_argument('--level',choices=['coarse','fine'],default='coarse')
    a=ap.parse_args()
    if a.projected_scan:projected_scan(json.loads(a.projected_scan.read_text()),a.workers)
    if a.pressure_boundary:pressure_boundary(json.loads(a.pressure_boundary.read_text()),a.pressure_step)
    if a.audit:
        from scipy.stats import qmc
        bounds=np.array(BOUNDS,float);bounds[-1]=[1600000,4000000]
        for x in qmc.scale(qmc.LatinHypercube(7,seed=911).random(a.audit),bounds[:,0],bounds[:,1]):
            r=equilibrium_probe(x)
            if r.get('q_axis',0)>=1.08:screen(x,modes=(1,))
    if a.latin:
        from scipy.stats import qmc
        low=[1.65,-.2,-.12,-.12,-.1,-.1,1150000]
        high=[2.15,.6,.12,.12,.1,.1,1500000]
        for x in qmc.scale(qmc.LatinHypercube(7,seed=a.seed).random(a.latin),low,high):
            screen(x,modes=(1,))
    if a.climb:
        x=np.array([json.loads(a.climb.read_text())[k] for k in KEYS]);steps=np.array([.04,.04,.025,.025,.02,.02,25000.])
        iteration_base=0
        if a.resume_climb:
            previous=json.loads((ROOT/'runs/climb_progress.json').read_text())
            if previous['incumbent']['design']!=design(x):raise ValueError('Resume seed differs from saved incumbent')
            steps=np.array(previous['steps'],float);incumbent=previous['incumbent'];iteration_base=previous['iteration']
        else:incumbent=screen(x,modes=(1,),scenarios=('broad','peaked'))
        save(ROOT/'runs/climb_incumbent.json',incumbent)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for iteration in range(a.iterations):
                improved=False
                for j in range(7):
                    trials=[]
                    for sign in (-1,1):
                        trial=x.copy();trial[j]=float(np.clip(trial[j]+sign*steps[j],*BOUNDS[j]))
                        if not np.array_equal(trial,x):trials.append(trial)
                    results=[f.result() for f in [pool.submit(screen,t,(1,),('broad','peaked')) for t in trials]]
                    good=[r for r in results if r['status']=='pass']
                    accepted=False
                    if good:
                        r=max(good,key=lambda r:r['pressure'])
                        if incumbent['status']!='pass' or r['pressure']>incumbent['pressure']:
                            direction=np.sign(r['design'][KEYS[j]]-x[j])
                            x=np.array([r['design'][k] for k in KEYS]);incumbent=r;improved=True;accepted=True
                            save(ROOT/'runs/climb_incumbent.json',r)
                    if a.expand and accepted:
                        # Probe farther along an empirically improving direction;
                        # every proposed point still runs the native constraints.
                        for expansion in range(2):
                            proposals=[];values=set()
                            for multiplier in (2.,4.):
                                trial=x.copy();trial[j]=float(np.clip(x[j]+direction*multiplier*steps[j],*BOUNDS[j]))
                                if not np.array_equal(trial,x) and trial[j] not in values:
                                    proposals.append(trial);values.add(trial[j])
                            if not proposals:break
                            tested=[f.result() for f in [pool.submit(screen,t,(1,),('broad','peaked')) for t in proposals]]
                            eligible=[r for r in tested if r['status']=='pass' and r['pressure']>incumbent['pressure']]
                            if not eligible:steps[j]*=.5;break
                            r=max(eligible,key=lambda r:r['pressure'])
                            x=np.array([r['design'][k] for k in KEYS]);incumbent=r;steps[j]*=2.
                            save(ROOT/'runs/climb_incumbent.json',r)
                    elif a.expand:steps[j]*=.5
                if not improved and not a.expand:steps*=.5
                save(ROOT/'runs/climb_progress.json',{'iteration':iteration_base+iteration+1,'improved':improved,'steps':steps.tolist(),'incumbent':incumbent})
                if np.all(steps<=np.array([.005,.005,.003,.003,.0025,.0025,2000.])):break
    if a.screen:screen([json.loads(a.screen.read_text())[k] for k in KEYS],a.modes,a.scenarios,level=a.level)
    if a.formal:formal(json.loads(a.formal.read_text()),a.label)
    if a.de:
        bounds=np.array(BOUNDS,float)
        result=differential_evolution(objective,bounds,popsize=5,maxiter=a.de,seed=734,polish=False,x0=[1.8,.4,0,0,0,0,4e5]);save(ROOT/'runs/de_result.json',{'design':design(result.x),'objective':float(result.fun)})
    if a.local:
        x=np.array([json.loads(a.local.read_text())[k] for k in KEYS]); scale=np.array([.1,.1,.05,.05,.04,.04,1e5])
        bounds=[((lo-v)/s,(hi-v)/s) for (lo,hi),v,s in zip(BOUNDS,x,scale)]
        result=minimize(lambda y:objective(x+y*scale),np.zeros(7),method='Powell',bounds=bounds,options={'maxfev':a.iterations,'xtol':.025,'ftol':1e-4});save(ROOT/'runs/local_result.json',{'design':design(x+result.x*scale),'objective':float(result.fun)})
if __name__=='__main__':main()
