#!/usr/bin/env python3
"""Inventory-conserving native OpenMC search for the calibrated workspace protocol.

Run under taskset -c 8-11 with OMP_NUM_THREADS=4. No transport approximation
is substituted for native runs; the GP only proposes the next design.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="4"
import numpy as np
from scipy.optimize import differential_evolution
from scipy.linalg import solve_triangular
from scipy.special import ndtr
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('official', ROOT/'workspace/eval.py')
E=importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
CASE=E.read_json(ROOT/'workspace/case.json')
BASE=E.read_json(ROOT/'workspace/reference/baseline.json')
V=np.array([E.region_volume(y,z) for l in range(6) for y in range(4) for z in range(4)])
DATA=ROOT/'runtime/data/cross_sections.xml'
EXE=ROOT/'runtime/native-env/bin/openmc'
ENV=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENMC_CROSS_SECTIONS=str(DATA),
         LD_LIBRARY_PATH=str(ROOT/'runtime/sysroot/usr/lib/aarch64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH',''))

def project(x,weights,target,lo,hi):
    x=np.asarray(x,dtype=float)
    left,right=lo-float(x.max())-1,hi-float(x.min())+1
    for _ in range(80):
        mid=(left+right)/2
        if np.dot(weights,np.clip(x+mid,lo,hi))>target: right=mid
        else:left=mid
    return np.clip(x+(left+right)/2,lo,hi)

def make_design(b,e,spatial=None):
    b=np.repeat(np.asarray(b,float),16) if len(b)==6 else np.asarray(b,float).copy()
    e=np.repeat(np.asarray(e,float),16) if len(e)==6 else np.asarray(e,float).copy()
    if spatial is not None:
        # Edge distance and diagnostic-port distance provide distinct lateral modes.
        y=np.tile(np.repeat([-30,-10,10,30],4),6); z=np.tile([-30,-10,10,30],24)
        edge=(np.abs(y)+np.abs(z)-40)/40
        port=np.exp(-((y-13)**2+z**2)/500)
        depth=np.repeat(np.linspace(1,0,6),16)
        b += spatial[0]*edge*depth+spatial[1]*port*depth
        e += spatial[2]*edge*depth+spatial[3]*port*depth
    b=project(b,V,.4*V.sum(),.2,.6)
    # Constant ceramic formula density makes Li6 inventory proportional to V*b*e.
    e=project(e,V*b,.12*V.sum()*(1-2e-12),.05,.9)
    return {'schema_version':'fusionx-n0-design/v1','case_id':CASE['case_id'],
            'regions':[{'id':rid,'breeder_fraction':float(bb),'li6_atom_fraction':float(ee)}
                       for rid,bb,ee in zip(E.region_ids(),b,e)]}

def encode(p):
    if len(p)==4:
        base=E.read_json(ROOT/'search/spatial_base.json')
        return make_design([r['breeder_fraction'] for r in base['regions']],
                           [r['li6_atom_fraction'] for r in base['regions']],
                           (np.asarray(p)-.5)*np.array([.3,.3,.8,.8]))
    return make_design(.2+.4*np.asarray(p[:6]),.05+.85*np.asarray(p[6:12]),
                       None if len(p)==12 else (np.asarray(p[12:16])-.5)*np.array([.3,.3,.8,.8]))

def measure(design,name,particles=50000,batches=8,sources=('cosine',),seeds=(4821901,)):
    runroot=ROOT/'search'/name
    if (runroot/'result.json').exists():
        result=E.read_json(runroot/'result.json')
        if result['design']!=design: raise RuntimeError('Cache name reused for a different design')
        return result
    runroot.mkdir(parents=True,exist_ok=False)
    E.write_json(runroot/'blanket.json',design)
    parsed=E.parse_design(runroot/'blanket.json',CASE)
    start=time.monotonic(); outcomes={}
    for source in sources:
        reps=[]
        for seed in seeds:
            model,inventory=E.prepare_model(parsed,CASE,source,seed,DATA,particles=particles,batches=batches)
            run=runroot/f'{source}-{seed}';run.mkdir()
            model.export_to_xml(run)
            with (run/'openmc.log').open('w') as log:
                subprocess.run([str(EXE),'-s','4'],cwd=run,env=ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
            logtext=(run/'openmc.log').read_text().lower()
            if 'lost particle' in logtext or 'could not be located' in logtext or list(run.glob('particle*.h5')):
                raise RuntimeError(f'Lost particle in {run}')
            # Exploratory runs need no geometry-summary linking when reading tallies.
            # Preserve the exact cumulative-batch differencing and native normalization.
            import openmc
            previous=np.zeros(17);observations=[]
            paths=sorted(run.glob('statepoint.*.h5'),key=lambda p:int(p.name.split('.')[1]))
            if len(paths)!=batches:raise RuntimeError('Incomplete native statepoints')
            for batch,path in enumerate(paths,1):
                with openmc.StatePoint(path,autolink=False) as state:
                    assert state.current_batch==batch and state.n_realizations==batch
                    assert state.run_mode=='fixed source'
                    cumulative=np.r_[float(state.get_tally(id=1).sum.sum()),state.get_tally(id=2).sum.reshape(-1)/2000.]
                    assert cumulative.shape==(17,) and np.isfinite(cumulative).all()
                    observation=cumulative-previous
                    assert np.all(observation>=-1e-12)
                    observations.append(observation);previous=cumulative
            observations=np.asarray(observations)
            np.save(run/'observations.npy',observations)
            reps.append(observations)
        samples=np.concatenate(reps)
        mean=samples.mean(0);se=samples.std(0,ddof=1)/np.sqrt(len(samples))
        outcomes[source]={'mean':mean.tolist(),'standard_error':se.tolist(),
                          'flux_mean_ratio':(mean[1:]/np.array(BASE['cases'][source]['screening_limits'])).tolist()}
    invratio={k:inventory[k]/BASE['inventory_limits'][k] for k in inventory}
    if max(invratio.values())>1+1e-10:raise RuntimeError(f'Inventory projection failed: {invratio}')
    result={'design':design,'inventory':inventory,'inventory_ratios':invratio,'outcomes':outcomes,
            'particles_per_batch':particles,'batches':batches,'seeds':list(seeds),
            'elapsed_seconds':time.monotonic()-start}
    E.write_json(runroot/'result.json',result)
    print(json.dumps({'name':name,'yield':{s:o['mean'][0] for s,o in outcomes.items()},
                      'max_flux_mean_ratio':{s:max(o['flux_mean_ratio']) for s,o in outcomes.items()},
                      'seconds':result['elapsed_seconds']}),flush=True)
    return result

def initial_profiles():
    yield 'uniform',make_design([.4]*6,[.3]*6)
    yield 'rear_enrichment',make_design([.4]*6,[.05,.1,.25,.4,.5,.5])
    yield 'front_ceramic_low_e',make_design([.6,.6,.5,.3,.2,.2],[.12,.2,.35,.55,.65,.65])
    yield 'front_be_rear_e',make_design([.2,.3,.4,.5,.5,.5],[.05,.1,.25,.45,.5,.5])
    yield 'front_enrichment',make_design([.4]*6,[.8,.55,.25,.1,.05,.05])
    yield 'front_ceramic',make_design([.6,.6,.5,.3,.2,.2],[.65,.45,.2,.1,.05,.05])
    yield 'front_be',make_design([.2,.3,.4,.5,.5,.5],[.8,.65,.3,.1,.05,.05])
    yield 'middle_capture',make_design([.3,.5,.6,.5,.3,.2],[.15,.65,.5,.2,.1,.05])
    yield 'max_front_be',make_design([.2,.2,.2,.6,.6,.6],[.9,.9,.9,.1,.1,.1])
    spare=make_design([.2,.2,.2,.6,.6,.6],[.9,.9,.9,.1,.1,.1])
    for r in spare['regions'][48:]:r['li6_atom_fraction']=.05
    yield 'unused_li6_rear_minimum',spare
    yield 'two_be_layers',make_design([.2,.2,.3,.5,.6,.6],[.9,.9,.6,.2,.1,.05])
    yield 'transition_005',make_design([.2,.2,.25,.55,.6,.6],[.9,.9,.9,.05,.05,.05])
    yield 'transition_010',make_design([.2,.2,.3,.5,.6,.6],[.9,.9,.9,.05,.05,.05])
    yield 'transition_020',make_design([.2,.2,.4,.4,.6,.6],[.9,.9,.7,.05,.05,.05])
    yield 'transition_030',make_design([.2,.2,.5,.3,.6,.6],[.9,.9,.57,.05,.05,.05])
    yield 'second_layer_capture',make_design([.2,.3,.2,.5,.6,.6],[.9,.9,.9,.05,.05,.05])
    yield 'rear_capture_frontmost',make_design([.2,.2,.2,.6,.6,.6],[.9,.9,.9,.2,.05,.05])
    yield 'rear_capture_middle',make_design([.2,.2,.2,.6,.6,.6],[.9,.9,.9,.05,.2,.05])
    center=np.array([.25 if y in (1,2) and z in (1,2) else .05 for y in range(4) for z in range(4)])
    yield 'rear_capture_center',make_design([.2,.2,.2,.6,.6,.6],np.r_[np.full(48,.9),np.tile(center,3)])
    center=np.array([.65 if y in (1,2) and z in (1,2) else .05 for y in range(4) for z in range(4)])
    yield 'rear_capture_front_center',make_design([.2,.2,.2,.6,.6,.6],np.r_[np.full(48,.9),center,np.full(32,.05)])
    yield 'front_both',make_design([.6,.5,.4,.3,.3,.3],[.9,.6,.2,.05,.05,.05])

def objective(result):
    outs=result['outcomes']
    val=min(o['mean'][0] for o in outs.values())
    # Exploratory flux estimates are noisy; full verification remains mandatory.
    violation=0.
    for source,o in outs.items():
        limits=np.array(BASE['cases'][source]['screening_limits'])
        uncertainty=np.array(o['standard_error'][1:])/limits
        # Do not mistake an exploratory upward fluctuation for a proven violation.
        violation+=sum(max(0,r-1.-1.645*err)**2 for r,err in zip(o['flux_mean_ratio'],uncertainty))
    return val-2*violation

def gp_proposal(X,Y,rng):
    X=np.asarray(X);Y=np.asarray(Y)
    # Matern-5/2 GP with observation noise. Kernel length selected by marginal likelihood.
    ym=Y.mean();ys=max(Y.std(),.005); yn=(Y-ym)/ys
    best=None
    for length in [.25,.5,.8,1.2,2.0]:
        distance=np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(2))/length
        kernel=(1+np.sqrt(5)*distance+5*distance**2/3)*np.exp(-np.sqrt(5)*distance)
        kernel+=np.eye(len(X))*(.0012/ys)**2
        try:L=np.linalg.cholesky(kernel)
        except np.linalg.LinAlgError:continue
        alpha=np.linalg.solve(L.T,np.linalg.solve(L,yn))
        nll=.5*yn@alpha+np.log(np.diag(L)).sum()
        if best is None or nll<best[0]:best=(nll,length,L,alpha)
    _,length,L,alpha=best
    def acquisition(p):
        dist=np.sqrt(((X-p)**2).sum(1))/length
        k=(1+np.sqrt(5)*dist+5*dist**2/3)*np.exp(-np.sqrt(5)*dist)
        mu=ym+ys*k@alpha
        vv=solve_triangular(L,k,lower=True,check_finite=False); sd=ys*np.sqrt(max(1e-10,1-vv@vv))
        improve=mu-max(Y)-.0005;z=improve/sd
        ei=improve*ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
        return -ei
    res=differential_evolution(acquisition,[(0,1)]*X.shape[1],maxiter=65,popsize=6,
                               polish=False,rng=rng,workers=1)
    return res.x

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['profiles','search','full'])
    p.add_argument('--design',type=Path);p.add_argument('--name',default='candidate')
    p.add_argument('--particles',type=int,default=50000);p.add_argument('--batches',type=int,default=8)
    p.add_argument('--iterations',type=int,default=30);p.add_argument('--dimension',type=int,choices=[4,12,16],default=12)
    args=p.parse_args()
    if args.mode=='full':
        subprocess.run([str(ROOT/'runtime/native-env/bin/python'),'-I',str(ROOT/'local_eval.py'),str(args.design.resolve()),
                        '--workdir',str(ROOT/'runs'/args.name),'--reward-dir',str(ROOT/'results'/args.name),'--threads','4'],
                       env=ENV,check=True)
    elif args.mode=='profiles':
        for name,d in initial_profiles():measure(d,name,args.particles,args.batches)
    else:
        folder=ROOT/'search';folder.mkdir(exist_ok=True)
        history_path=folder/f'bo{args.dimension}.json'
        history=E.read_json(history_path) if history_path.exists() else []
        for h in history:
            h['objective']=objective(E.read_json(folder/h['name']/'result.json'))
        training=[]
        if args.dimension==16 and (folder/'bo12.json').exists():
            for h in E.read_json(folder/'bo12.json'):
                training.append({'params':h['params']+[.5]*4,
                                 'objective':objective(E.read_json(folder/h['name']/'result.json'))})
        if args.dimension==4 and not (folder/'spatial_base.json').exists():
            measured=[E.read_json(q) for q in folder.glob('med-*/result.json')]
            if not measured:raise RuntimeError('Independent radial refinement required first')
            E.write_json(folder/'spatial_base.json',max(measured,key=objective)['design'])
        rng=np.random.default_rng(1062026)
        for index in range(len(history),args.iterations):
            if index<10:
                if index==0 and args.dimension==4:
                    params=np.full(4,.5)
                elif index==0:
                    best=max((E.read_json(x) for x in folder.glob('*/result.json')),key=objective)
                    bb=np.array([r['breeder_fraction'] for r in best['design']['regions']]).reshape(6,16).mean(1)
                    ee=np.array([r['li6_atom_fraction'] for r in best['design']['regions']]).reshape(6,16).mean(1)
                    params=np.r_[(bb-.2)/.4,(ee-.05)/.85,[.5]*max(0,args.dimension-12)]
                elif index==1 and args.dimension==12:
                    params=np.r_[np.array([0,0,0,1,1,1],float),(np.array([.9,.9,.9,.1,.1,.1])-.05)/.85]
                elif index==2 and args.dimension==12:
                    params=np.r_[(np.array([.2,.2,.3,.5,.6,.6])-.2)/.4,(np.array([.9,.9,.6,.2,.1,.05])-.05)/.85]
                else:
                    base=np.array(history[0]['params'])
                    params=np.clip(base+rng.normal(0,.26,args.dimension),0,1)
            else:params=gp_proposal([h['params'] for h in training+history],[h['objective'] for h in training+history],rng)
            result=measure(encode(params),f'bo{args.dimension}-{index:03d}',args.particles,args.batches)
            history.append({'params':params.tolist(),'objective':objective(result),'name':f'bo{args.dimension}-{index:03d}'})
            E.write_json(history_path,history)
            best=max(history,key=lambda h:h['objective'])
            E.write_json(ROOT/'blanket.json',E.read_json(folder/best['name']/'result.json')['design'])
            print('BEST',best['name'],best['objective'],flush=True)

if __name__=='__main__':main()
