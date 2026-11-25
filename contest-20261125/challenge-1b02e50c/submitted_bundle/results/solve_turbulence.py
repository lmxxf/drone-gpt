#!/usr/bin/env python3
"""VMEC++ / gx_geometry boundary search; proxy values are NOT GX heat flux.

Run with a Python environment containing vmecpp, simsopt, booz_xform, netCDF4.
The read-only sibling environment is reused; all search outputs stay here.
"""
import os
os.sched_setaffinity(0, {16,17,18,19})
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse, contextlib, hashlib, io, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.linalg import eigh
ROOT=Path(__file__).resolve().parent
if not (ROOT/'workspace').exists() and (ROOT.parent/'workspace').exists(): ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'workspace'))
sys.path.insert(0,str(ROOT/'tools/src/gx_geometry'))
import magnetic_eval as mag
import vmecpp, booz_xform
from gx_geometry.vmec import Vmec
from gx_geometry.vmec_diagnostics import vmec_fieldline_from_center, vmec_splines
from gx_geometry.gx_grid import add_gx_definitions, uniform_arclength, resample
from gx_geometry.eik_files import write_eik_netcdf


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def quick_geom(b):
    # Exact Fourier quadrature for area and volume, no polygon-intersection shortcut
    # is used for final verification; mag.geometry checks the complete boundary.
    theta=2*np.pi*np.arange(128)/128;phi=np.pi*np.arange(64)/64
    R,Z,Zt=mag._surface(b,theta,phi)
    area=2*np.pi*np.mean(R*Zt,axis=1)
    if np.min(area)<=0 or np.min(R)<=0:raise ValueError('invalid section')
    return math.sqrt(area.mean()/np.pi),float(2*np.pi**2*np.mean(R**2*Zt))


def normalize(data):
    b=mag.Boundary(2,{(v['m'],v['n']):v['rbc'] for v in data['modes']},
                   {(v['m'],v['n']):v['zbs'] for v in data['modes']})
    a,V=quick_geom(b);b=b.scaled(1.7/a)
    a,V=quick_geom(b)
    # Translating every cross-section radially leaves area unchanged; dV/dR00=2*pi*Sbar.
    b.rbc[(0,0)]+=(581.8735601800271-V)/(2*np.pi**2*a*a)
    return {'nfp':2,'modes':[dict(m=m,n=n,rbc=b.rbc[m,n],zbs=b.zbs[m,n]) for m,n in sorted(b.rbc)]}


def equilibrium(data,folder,final=False):
    save(folder/'boundary.json',data)
    b=mag.read_boundary(folder/'boundary.json')
    ns=257 if final else 65
    attempt=mag.VmecAttempt((17,33,65,129,257) if final else (17,33,65),
          (2000,4000,8000,16000,60000) if final else (1500,3000,16000),
          (1e-12,1e-12,1e-12,1e-12,1e-16) if final else (1e-10,1e-10,1e-12),.7)
    text=mag.vmec_input(b,attempt)
    if not final:
        m,n=mag.vmec_resolution(b)
        text=text.replace(f'MPOL = {m}, NTOR = {n}','MPOL = 8, NTOR = 6')
    text=text.replace("  PRECON_TYPE = 'none', PREC2D_THRESHOLD = 1e-12\n",'')
    inp=folder/'input.eq';inp.write_text(text)
    out=vmecpp.run(vmecpp.VmecInput.from_file(inp),max_threads=4,verbose=False)
    wout=folder/'wout_eq.nc';out.wout.save(wout)
    w=mag._wout(wout)
    residuals=[float(w[k]) for k in ('fsqr','fsqz','fsql')]
    if int(w['ier_flag'])!=0 or int(w['ns'])!=ns or max(residuals)>attempt.ftol[-1]:
        raise ValueError(f'VMEC++ not converged: {residuals}')
    return b,w,wout,residuals


def proxy_tube(wout,theta,phi,folder,nz=97):
    vm=Vmec(str(wout));vs=vmec_splines(vm)
    wanted=75*1.7/float(vm.wout.Aminor_p)
    def length(turns):
        return float(vmec_fieldline_from_center(vs,.5,theta,phi,turns,1001).length)
    turns=brentq(lambda t:length(t)-wanted,.05,10,xtol=1e-9)
    fl=vmec_fieldline_from_center(vs,.5,theta,phi,turns,1001)
    add_gx_definitions(fl,-1.)
    with contextlib.redirect_stdout(io.StringIO()):fl=resample(uniform_arclength(fl),nz)
    write_eik_netcdf(fl,str(folder))
    # Uniform arclength, a units; use N=nz-1 unique samples, forced periodic.
    B=np.asarray(fl.bmag)[:-1]; metric=np.asarray(fl.gds2)[:-1]
    curvature=np.asarray(fl.cvdrift)[:-1]
    # Local shear is already retained through the full gds2 variation along the tube.
    # Positive cvdrift is the bad-curvature branch in the sigma_Bxy=-1 GX convention.
    # Fluid interchange / mixing-length heuristic: gamma^2 = omega_*T*omega_d
    # minus parallel sound transit and FLR shielding. It is not a kinetic eigenproblem.
    N=len(B);ds=75/N
    D=np.zeros((N,N))
    for j in range(N):D[j,j]=-1/ds;D[j,(j+1)%N]=1/ds
    weight=1/B;weight/=weight.mean()
    W=np.diag(weight)
    parallel=D.T@W@D
    radial_metric=np.asarray(fl.grho)[:-1]**2
    rows={}
    for gradient in (3,4,5):
        best=0.;best_ky=None
        for ky in (.15,.25,.4,.65,1.):
            k2=ky*ky*metric
            drive=ky*ky*(gradient-.9*2/3)*curvature/(1+k2)
            operator=np.diag(weight*drive)-parallel
            lam,vec=eigh(operator,W,subset_by_index=[N-1,N-1])
            gamma=math.sqrt(max(float(lam[0]),0.))
            mode=vec[:,0]**2*weight;mode/=mode.sum()
            mixing=float(mode@k2)
            q=gradient*gamma*float(mode@radial_metric)/mixing
            if q>best:best=q;best_ky=ky
        rows[str(gradient)]={'Q_proxy':best,'ky':best_ky}
    return dict(points=rows,B_ratio=float(B.max()/B.min()),
                curvature_rms=float(np.sqrt(np.mean(curvature**2))),
                bad_curvature_mean=float(np.mean(np.maximum(curvature,0))),
                compression_mean=float(metric.mean()),poloidal_turns=turns,
                length_over_a=float(fl.length)*float(vm.wout.Aminor_p)/1.7)


def evaluate(data,final=False):
    digest=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()[:14]
    folder=ROOT/'runs'/('final_v2' if final else 'proxy_v2')/digest
    if (folder/'result.json').exists():return json.loads((folder/'result.json').read_text())
    folder.mkdir(parents=True,exist_ok=True);started=time.time()
    r={'folder':str(folder.relative_to(ROOT)),'final_resolution':final,'full_feasible':None}
    try:
        b,w,wout,res=equilibrium(data,folder,final)
        r.update(profile=mag.profile_metrics(w),residuals=res)
        r['coil']=mag.coil_room(wout,2)
        bx=booz_xform.Booz_xform();bx.verbose=0;bx.read_wout(str(wout))
        bx.compute_surfs=[int(i*(int(w['ns'])-2)) for i in (.1,.3,.5,.7,.9)];bx.mboz=32;bx.nboz=24;bx.run()
        B=np.asarray(bx.bmnc_b);xm=np.asarray(bx.xm_b);xn=np.asarray(bx.xn_b)
        i00=np.flatnonzero((xm==0)&(xn==0))[0]
        r['qs_rms']=float(np.max(np.sqrt(np.sum(B[xn!=0]**2,axis=0))/abs(B[i00])))
        r['screen_pass']=r['profile']['well']>=0 and r['profile']['iota_min']>=.3 and r['coil']['lgradb_min']>=2.4
        # QS RMS is a search guard, NEVER a substitute for NEO epsilon_eff or SIMPLE.
        r['tubes']={};r['points']={}
        for label,t,p in [('T00',0,0),('T10',np.pi,0),('T01',0,np.pi/2),('T11',np.pi,np.pi/2)]:
            tube=proxy_tube(wout,t,p,folder/(label+'.nc'))
            r['tubes'][label]=tube
            for g,value in tube['points'].items():r['points'][label+'-g'+g]=value['Q_proxy']
        r['J_proxy']=max(r['points'].values())
    except Exception as exc:r.update(error=f'{type(exc).__name__}: {exc}',screen_pass=False)
    r['seconds']=time.time()-started;save(folder/'result.json',r)
    print(json.dumps(r),flush=True)
    return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--search',type=int,default=0);ap.add_argument('--refine',action='store_true');ap.add_argument('--final',type=Path)
    args=ap.parse_args()
    if args.refine:
        import runpy
        runpy.run_path(str(ROOT/'tools/refine_proxy.py'),run_name='__main__')
        return
    if args.final:
        print(json.dumps(evaluate(json.loads(args.final.read_text()),True)));return
    base=normalize(json.loads((ROOT/'workspace/reference_configurations/landreman_paul_2022_QA_well.json').read_text()))
    baseline=evaluate(base);save(ROOT/'results/proxy-baseline.json',baseline)
    if 'J_proxy' not in baseline:raise RuntimeError('baseline proxy failed')
    records=[baseline];best=baseline;best_data=base
    # Orthogonal paths: elongation at fixed area, helical m=1 pairs, m=2 triangularity.
    directions=[('elongation',None),('axis',None)]+[(v,(m,n)) for m,n in [(1,0),(1,1),(1,-1),(2,0),(2,1),(2,-1)] for v in ('rbc','zbs')]
    for i in range(args.search):
        name,key=directions[(i//2)%len(directions)];sign=1 if i%2==0 else -1
        data=json.loads(json.dumps(base));step=sign*(.006 if name in ('elongation','axis') else .0005)
        for v in data['modes']:
            m,n=v['m'],v['n']
            if name=='elongation' and m>0:v['rbc']*=math.exp(-step);v['zbs']*=math.exp(step)
            elif name=='axis' and m==0 and n>0:v['rbc']*=1+step;v['zbs']*=1+step
            elif key==(m,n):v[name]+=step*1.7
        data=normalize(data);r=evaluate(data);r['direction']=[name,key,step];records.append(r)
        if r.get('screen_pass') and r.get('qs_rms',1)<max(.002,baseline.get('qs_rms',0)*2) and r.get('J_proxy',1e99)<best['J_proxy']:
            best=r;best_data=data;save(ROOT/'candidate_boundary.json',data)
        save(ROOT/'results/search.json',dict(baseline=baseline,best=best,records=records,
             proxy_reduction=1-best['J_proxy']/baseline['J_proxy'],full_feasible=None))
    if args.search:save(ROOT/'candidate_boundary.json',best_data)

if __name__=='__main__':main()
