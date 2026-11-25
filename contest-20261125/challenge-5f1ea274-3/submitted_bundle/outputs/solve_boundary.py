#!/usr/bin/env python3
"""Reduced boundary search. Screening results are NOT full-evaluator certificates."""
import os
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
os.sched_setaffinity(0, {12,13,14,15})
import argparse, hashlib, json, math, pathlib, subprocess, sys, time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'workspace'))
import eval as ev

def write_json(p, data):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(p)

def deform(data, x):
    # One common thickening; independent axis corrugation, elongation and higher
    # shaping preserve the seed's signed QA/well mode combinations.
    thick, axis, elong, shaping=x
    out=json.loads(json.dumps(data))
    for a in out['modes']:
        m,n=a['m'],a['n']
        if (m,n)==(0,0):continue
        scale=(axis if m==0 else thick*(shaping if m>=2 else 1))
        a['rbc']*=scale/math.sqrt(elong) if m>0 else scale
        a['zbs']*=scale*math.sqrt(elong) if m>0 else scale
    return out

def screen(data, folder):
    import vmecpp, booz_xform
    folder.mkdir(parents=True,exist_ok=True)
    p=folder/'boundary.json';write_json(p,data)
    b=ev.read_boundary(p);g=ev.geometry(b)
    result={'geometry':g,'full_verified':False}
    if g['kappa_max']>8:result['rejected']='elongation';return result
    b=b.scaled(ev.MINOR_RADIUS/g['minor_radius'])
    attempt=ev.VmecAttempt((17,33,65),(1500,3000,15000),(1e-10,1e-10,1e-12),0.7)
    text=ev.vmec_input(b,attempt)
    mpol,ntor=ev.vmec_resolution(b)
    text=text.replace(f'MPOL = {mpol}, NTOR = {ntor}',f'MPOL = {max(b.m_max+2,8)}, NTOR = {max(b.n_max+1,6)}')
    text=text.replace("  PRECON_TYPE = 'none', PREC2D_THRESHOLD = 1e-12\n",'')
    inp=folder/'input.eq';inp.write_text(text)
    start=time.time()
    out=vmecpp.run(vmecpp.VmecInput.from_file(inp),max_threads=4,verbose=False)
    wout=folder/'wout_eq.nc';out.wout.save(wout)
    w=ev._wout(wout);result['residuals']=[float(w[k]) for k in ('fsqr','fsqz','fsql')]
    if max(result['residuals'])>1e-12:raise RuntimeError('screen equilibrium did not converge')
    result['profile']=ev.profile_metrics(w)
    result['coil']=ev.coil_room(wout,b.nfp)
    bx=booz_xform.Booz_xform();bx.verbose=0;bx.read_wout(str(wout));bx.compute_surfs=[15,31,47,62];bx.mboz=32;bx.nboz=24;bx.run()
    B=np.asarray(bx.bmnc_b);xm=np.asarray(bx.xm_b);xn=np.asarray(bx.xn_b)
    mask=xn!=0 if b.nfp==2 else min((xn!=b.nfp*xm, xn!=-b.nfp*xm), key=lambda k:float(np.sum(B[k]**2)))
    i00=np.flatnonzero((xm==0)&(xn==0))[0]
    result['qs_rms']=float(np.max(np.sqrt(np.sum(B[mask]**2,axis=0))/np.abs(B[i00])))
    result['seconds']=time.time()-start
    result['screen_pass']=result['qs_rms']<=0.02 and result['profile']['well']>=0 and result['profile']['iota_min']>=.3 and result['coil']['lgradb_min']>=2.4
    return result

def validate(p, folder):
    # Preserve the original evaluator, including its eight fixed MPI ranks. All
    # ranks inherit the four-core affinity, and each BLAS/OpenMP worker uses one.
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    subprocess.run([sys.executable,'-I',str(ROOT/'workspace/eval.py'),str(p),'--workdir',str(folder/'work'),'--reward-dir',str(folder/'result'),'--threads','4'],env=env,check=True)
    return json.loads((folder/'result/reward.json').read_text())

def optimize(max_evals):
    from scipy.optimize import minimize
    seed=json.loads((ROOT/'workspace/reference_configurations/landreman_paul_2022_QA_well.json').read_text())
    history=[];cache={}
    def evaluate_x(x):
        data=deform(seed,x);key=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:12]
        if key in cache:return cache[key]
        folder=ROOT/'runs'/('opt_'+key);rp=folder/'screen.json'
        if rp.exists():r=json.loads(rp.read_text())
        else:
            try:r=screen(data,folder)
            except Exception as exc:r={'error':f'{type(exc).__name__}: {exc}','full_verified':False}
            folder.mkdir(parents=True,exist_ok=True);write_json(rp,r)
        row={'parameters':list(map(float,x)),'folder':str(folder.relative_to(ROOT)),**r}
        cache[key]=row;history.append(row);write_json(ROOT/'optimization_results.json',history);print(json.dumps(row),flush=True)
        good=[z for z in history if z.get('screen_pass')]
        if good:
            best=min(good,key=lambda z:z['geometry']['aspect_ratio'])
            write_json(ROOT/'candidate_boundary.json',json.loads((ROOT/best['folder']/'boundary.json').read_text()))
        return row
    def objective(x):
        r=evaluate_x(x)
        return r['geometry']['aspect_ratio'] if 'error' not in r else 100.
    def constraints(x):
        r=evaluate_x(x)
        if 'error' in r:return np.full(4,-100.) # explicitly recorded failed solve
        return np.array([(r['profile']['well']-.0003)/.01,
            (r['profile']['iota_min']-.32)/.1,
            (r['coil']['lgradb_min']-2.6)/2.,(.003-r['qs_rms'])/.003])
    result=minimize(objective,[1.03,1.,1.,1.],method='COBYLA',bounds=[(1.,1.5),(.8,1.2),(.85,1.2),(.85,1.2)],
        constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':max_evals,'rhobeg':.02,'tol':.001,'catol':.0001})
    write_json(ROOT/'optimizer_status.json',{'success':bool(result.success),'message':str(result.message),'x':result.x.tolist(),'fun':float(result.fun),'nfev':int(result.nfev),'full_verified':False})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--optimize',action='store_true');ap.add_argument('--max-evals',type=int,default=30);ap.add_argument('--baseline',action='store_true');ap.add_argument('--search',action='store_true');ap.add_argument('--validate',type=pathlib.Path);ap.add_argument('--scales',default='1,1.02,1.04,1.06,1.08,1.1');a=ap.parse_args()
    if a.optimize:
        optimize(a.max_evals);return
    if a.validate:
        print(json.dumps(validate(a.validate.resolve(),ROOT/'runs'/('full_'+str(time.time_ns())))));return
    seeds=sorted((ROOT/'workspace/reference_configurations').glob('*.json')) if a.baseline else [ROOT/'workspace/reference_configurations/landreman_paul_2022_QA_well.json']
    records=[]
    for seed in seeds:
        consecutive_failures=0
        for t in ([1.] if a.baseline else list(map(float,a.scales.split(',')))):
            x=[t,1.,1.,1.];data=deform(json.loads(seed.read_text()),x)
            key=seed.stem+'_'+hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:12];folder=ROOT/'runs'/key;rp=folder/'screen.json'
            if rp.exists():r=json.loads(rp.read_text())
            else:
                try:r=screen(data,folder)
                except Exception as exc:r={'error':f'{type(exc).__name__}: {exc}','full_verified':False}
                write_json(rp,r)
            records.append({'seed':seed.name,'parameters':x,'folder':str(folder.relative_to(ROOT)),**r});print(json.dumps(records[-1]),flush=True)
            write_json(ROOT/('baseline_screening.json' if a.baseline else 'screening_results.json'),records)
            consecutive_failures=0 if r.get('screen_pass') else consecutive_failures+1
            if not a.baseline and consecutive_failures>=2:break
    good=[r for r in records if r.get('screen_pass')]
    if good:
        best=min(good,key=lambda r:r['geometry']['aspect_ratio'])
        # A screen-only candidate never replaces the full-verified incumbent.
        write_json(ROOT/'candidate_boundary.json',json.loads((ROOT/best['folder']/'boundary.json').read_text()))
if __name__=='__main__':main()
