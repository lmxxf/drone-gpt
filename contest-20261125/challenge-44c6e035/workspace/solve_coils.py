#!/usr/bin/env python3
"""Native SIMSOPT finite-build L-BFGS-B, with differentiated geometry penalties.
Final acceptance is exclusively workspace/eval.py, never this search model.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax import jit, value_and_grad, vjp
import eval as ev
from simsopt.geo import CurveXYZFourier, FrameRotation, FramedCurveCentroid, CurveFilament
from simsopt.geo.framedcurve import torsion_pure_centroid
from simsopt.field import BiotSavart, Coil, Current
from simsopt.objectives import SquaredFlux
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parent
CASE=json.loads((ROOT/'assets/case.json').read_text())
RPACK=np.hypot(.4,.6)/2

def basis(n, order=8, derivative=0):
    t=np.arange(n)/n
    b=np.zeros((2*order+1,n)); b[0]=1 if derivative==0 else 0
    for m in range(1,order+1):
        b[2*m-1]=(2*np.pi*m)**derivative*np.sin(2*np.pi*m*t+derivative*np.pi/2)
        b[2*m]=(2*np.pi*m)**derivative*np.cos(2*np.pi*m*t+derivative*np.pi/2)
    return b

def unpack(x):
    return x[:204].reshape(4,3,17),x[204:224].reshape(4,5),jnp.concatenate((jnp.array([CASE['first_current_Aturn']/1e6]),x[224:]))

def encode(data):
    return np.r_[np.array([[c[a] for a in 'xyz'] for c in data['coils']]).ravel(),np.array([c['rotation'] for c in data['coils']]).ravel(),[c['current']/1e6 for c in data['coils'][1:]]]

def decode(x):
    q,r,i=unpack(jnp.asarray(x)); q,r,i=map(np.asarray,(q,r,i))
    return dict(nfp=2,order=8,coils=[dict(x=c[0].tolist(),y=c[1].tolist(),z=c[2].tolist(),rotation=a.tolist(),current=float(v*1e6)) for c,a,v in zip(q,r,i)])

def write(path,x):
    path=Path(path); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(decode(x),indent=2,allow_nan=False)); tmp.replace(path)

class Search:
    def __init__(self, data,nphi=64,ntheta=32,count=128,section=3):
        self.x0=encode(data); self.count=count
        self.surface=ev.surface(CASE,ROOT/'assets',nphi,ntheta)
        self.points=self.surface.gamma().reshape(-1,3)
        self.normal=self.surface.unitnormal().reshape(-1,3)
        self.weights=np.linalg.norm(self.surface.normal(),axis=2).ravel(); self.weights/=self.weights.sum()
        self.phi=np.c_[-self.points[:,1],self.points[:,0],np.zeros(len(self.points))]/np.hypot(self.points[:,0],self.points[:,1])[:,None]
        self.base_index=[]; self.maps=[]
        _,_,physical=ev.reconstruct(data,count)
        options=[np.diag(d) for d in [(1,1,1),(1,-1,-1),(-1,-1,1),(-1,1,-1)]]
        q=self.x0[:204].reshape(4,3,17)
        for coil in physical:
            cc=coil.curve.local_full_x.reshape(3,17)
            matches=[(np.max(abs(cc-m@c)),i,m) for i,c in enumerate(q) for m in options]
            error,i,m=min(matches,key=lambda v:v[0]); assert error<1e-10
            self.base_index.append(i); self.maps.append(m)
        self.maps=np.array(self.maps); self.base_index=np.array(self.base_index)
        self.sign=np.array([c.current.get_value()/data['coils'][i]['current'] for c,i in zip(physical,self.base_index)])
        self.curves=[CurveXYZFourier(count,8) for _ in range(16)]
        self.angles=[FrameRotation(c.quadpoints,2,scale=1.) for c in self.curves]
        self.currents=[Current(1.) for _ in range(16)]
        nodes,w=leggauss(section); coils=[]
        for c,a,cur in zip(self.curves,self.angles,self.currents):
            frame=FramedCurveCentroid(c,a)
            for j in range(section):
                for k in range(section):
                    coils.append(Coil(CurveFilament(frame,.2*nodes[j],.3*nodes[k]),cur*(w[j]*w[k]/4)))
        self.field=BiotSavart(coils); self.flux=SquaredFlux(self.surface,self.field,definition="normalized")
        self.bank=np.load(ROOT/'assets/search_scenarios.npz')['errors']
        self.samples=[-1]
        self.temperature=.05
        self.B=jnp.asarray(basis(count)); self.D=jnp.asarray(basis(count,derivative=1))
        self.identity=np.tile(np.eye(3),(16,1,1))
        self.transform=jit(self._transform)
        self.transform_pull=jit(lambda x,rot,trans,gq,gr,gi: vjp(lambda z:self._transform(z,rot,trans),x)[1]((gq,gr,gi))[0])
        self.setup_geometry()
        self.calls=0; self.started=time.monotonic(); self.last={}

    def _transform(self,x,rot,trans):
        q,r,i=unpack(x)
        g=q@self.B; d=q@self.D
        speed=jnp.linalg.norm(d,axis=1)
        centers=jnp.sum(g*speed[:,None,:],axis=2)/jnp.sum(speed,axis=1)[:,None]
        phys=jnp.einsum('nij,njk->nik',jnp.asarray(self.maps),q[self.base_index])
        center=jnp.einsum('nij,nj->ni',jnp.asarray(self.maps),centers[self.base_index])
        moved=jnp.einsum('nij,njk->nik',rot,phys)
        moved=moved.at[:,:,0].add(center+trans-jnp.einsum('nij,nj->ni',rot,center))
        return moved,r[self.base_index],i[self.base_index]*jnp.asarray(self.sign)*1e6

    def field_value(self,x,sample=-1,grad=True):
        if sample<0: rot=self.identity; trans=np.zeros((16,3))
        else: rot=Rotation.from_rotvec(self.bank[sample,:,3:]).as_matrix(); trans=self.bank[sample,:,:3]
        q,r,curr=self.transform(jnp.asarray(x),rot,trans)
        for c,a,cur,cq,ar,ci in zip(self.curves,self.angles,self.currents,q,r,curr):
            c.local_full_x=np.asarray(cq).ravel(); a.local_full_x=np.asarray(ar); cur.local_full_x=np.array([float(ci)])
        field=self.field.B(); bn=np.sum(field*self.normal,axis=1)
        num=np.dot(self.weights,bn**2); den=np.dot(self.weights,np.sum(field**2,axis=1))
        e2=2*float(self.flux.J()); rms=np.sqrt(den)
        # Equivalent to 2*normalized SquaredFlux; scale to O(1).
        objective=e2*1e6
        drms=max(0,abs(rms/5.7-1)-.012)
        objective+=1e4*drms**2
        if not grad: return np.sqrt(e2),rms
        dB=2e6*self.weights[:,None]*(bn[:,None]*self.normal/den-num*field/den**2)
        if drms>0: dB+=2e4*drms*np.sign(rms/5.7-1)*self.weights[:,None]*field/(5.7*rms)
        norm=np.linalg.norm(field,axis=1)
        for index,gap,vector in [(int(np.argmin(norm)),3.02-norm.min(),-field[np.argmin(norm)]/norm.min()),(int(np.argmax(norm)),norm.max()-8.98,field[np.argmax(norm)]/norm.max()),(int(np.argmin(-np.sum(field*self.phi,axis=1))),3.02-np.min(-np.sum(field*self.phi,axis=1)),self.phi[np.argmin(-np.sum(field*self.phi,axis=1))])]:
            if gap>0:
                objective+=1e4*gap*gap; dB[index]+=2e4*gap*vector
        if sample<0 and e2>.00098**2:
            gap=np.sqrt(e2)-.00098; objective+=1e8*gap*gap
            dB+=2e8*gap*self.weights[:,None]*(bn[:,None]*self.normal/den-num*field/den**2)/np.sqrt(e2)
        derivative=self.field.B_vjp(np.ascontiguousarray(dB))
        gq=np.array([derivative.data[c].reshape(3,17) for c in self.curves])
        gr=np.array([derivative.data[a] for a in self.angles])
        gi=np.array([derivative.data[c][0] for c in self.currents])
        gx=np.asarray(self.transform_pull(jnp.asarray(x),rot,trans,gq,gr,gi))
        return objective,gx,np.sqrt(e2),rms

    def setup_geometry(self):
        n=512
        self.gb=np.asarray(basis(n)); self.gd=np.asarray(basis(n,derivative=1)); self.gdd=np.asarray(basis(n,derivative=2))
        self.ab=np.asarray(basis(n,2)); self.ad=np.asarray(basis(n,2,1))
        s=ev.surface(CASE,ROOT/'assets',1024,256)
        self.target=s.gamma().reshape(-1,3); self.stree=cKDTree(self.target)
        self.scover=ev.surface_cover_radius(s,1024,256)
        self.geo=jit(value_and_grad(self._geometry,has_aux=True))

    def nearest(self,x):
        q=x[:204].reshape(4,3,17); g=q@self.gb
        phys=np.einsum('nij,njk->nik',self.maps,g[self.base_index]).transpose(0,2,1)
        pairs=[]
        for i in range(16):
            tree=cKDTree(phys[i])
            for j in range(i):
                d,idx=tree.query(phys[j]); k=np.argmin(d); pairs.append((i,int(idx[k]),j,int(k)))
        target=self.target[self.stree.query(g.transpose(0,2,1).reshape(-1,3))[1]].reshape(4,-1,3)
        selfpairs=[]
        for i in range(4):
            speed=np.linalg.norm(q[i]@self.gd,axis=0)
            increments=.5*(speed+np.roll(speed,-1))/len(speed)
            arc=np.r_[0,np.cumsum(increments[:-1])]
            da=abs(arc[:,None]-arc[None,:]); eligible=np.minimum(da,increments.sum()-da)>=3.95
            distances=np.linalg.norm(g[i].T[:,None,:]-g[i].T[None,:,:],axis=2)
            distances[~eligible]=np.inf
            u,v=np.unravel_index(np.argmin(distances),distances.shape); selfpairs.append((u,v))
        return np.array(pairs),target,np.array(selfpairs)

    def _geometry(self,x,pairs,target,selfpairs):
        q,rotation,_=unpack(x); b=jnp.asarray(self.gb); d=jnp.asarray(self.gd); dd=jnp.asarray(self.gdd)
        g=q@b; gd=q@d; gdd=q@dd
        speed=jnp.linalg.norm(gd,axis=1); length=jnp.mean(speed,axis=1)
        centers=jnp.sum(g*speed[:,None,:],axis=2)/jnp.sum(speed,axis=1)[:,None]
        amplitudes=jnp.sqrt(jnp.sum(q[:,:,1::2]**2,axis=1)+1e-24)+jnp.sqrt(jnp.sum(q[:,:,2::2]**2,axis=1)+1e-24)
        freq=2*jnp.pi*jnp.arange(1,9)
        v=jnp.sum(amplitudes*freq,axis=1); a=jnp.sum(amplitudes*freq**2,axis=1); jerk=jnp.sum(amplitudes*freq**3,axis=1)
        cover=v/(2*16384)
        rmax=jnp.max(jnp.linalg.norm(g-centers[:,:,None],axis=1),axis=1)+cover+.001
        move=jnp.sqrt(3)*.01+2*rmax*jnp.sin(jnp.sqrt(3)*.001/2)
        # Extra dense-sampling margins during search; official bound uses N=16384.
        radius=jnp.min(jnp.linalg.norm(g[:,:2],axis=1),axis=1)-cover-move-RPACK
        vmin=jnp.min(speed,axis=1)-a/(2*16384)
        kap=jnp.linalg.norm(jnp.cross(gd.transpose(0,2,1),gdd.transpose(0,2,1)),axis=2)/speed**3
        ku=jnp.max(kap,axis=1)+(jerk/vmin**2+3*a*a/vmin**3)/(2*16384)
        twist=jax.vmap(torsion_pure_centroid)(g.transpose(0,2,1),gd.transpose(0,2,1),gdd.transpose(0,2,1),rotation@jnp.asarray(self.ab),rotation@jnp.asarray(self.ad))
        tu=jnp.max(jnp.abs(twist),axis=1)
        phys=jnp.einsum('nij,njk->nik',jnp.asarray(self.maps),g[self.base_index]).transpose(0,2,1)
        pd=jnp.linalg.norm(phys[pairs[:,0],pairs[:,1]]-phys[pairs[:,2],pairs[:,3]],axis=1)
        ci=jnp.asarray(self.base_index)[pairs[:,0]]; cj=jnp.asarray(self.base_index)[pairs[:,2]]
        pd=pd-cover[ci]-cover[cj]-move[ci]-move[cj]-2*RPACK
        plasma=jnp.min(jnp.linalg.norm(g.transpose(0,2,1)-target,axis=2),axis=1)-cover-move-self.scover-RPACK
        z=jnp.max(jnp.abs(g[:,2]),axis=1)+cover+move+RPACK
        rupper=jnp.max(jnp.linalg.norm(g[:,:2],axis=1),axis=1)+cover+move+RPACK
        selfdist=jnp.linalg.norm(g.transpose(0,2,1)[jnp.arange(4),selfpairs[:,0]]-g.transpose(0,2,1)[jnp.arange(4),selfpairs[:,1]],axis=1)-2*cover-2*RPACK
        gaps=jnp.concatenate((jnp.maximum(.12-selfdist,0),jnp.maximum(2.510-radius,0),jnp.maximum(ku-.545,0)*5,jnp.maximum(tu-.240,0)*5,jnp.maximum(.865-2*RPACK-pd,0),jnp.maximum(2.415-plasma,0),jnp.maximum(z-9.98,0),jnp.maximum(rupper-21.98,0),jnp.array([jnp.maximum(jnp.sum(length+a/(4*16384))-199.9,0)])))
        penalty=200*jnp.sum(gaps*gaps)
        # Small spectral regularizer controls unresolved high-frequency excursions.
        penalty+=1e-5*jnp.sum((q[:,:,1:]**2)*jnp.repeat(jnp.arange(1,9)**4,2)[None,None,:])
        aux=jnp.array([jnp.min(radius),jnp.max(ku),jnp.max(tu),jnp.min(pd),jnp.min(plasma),jnp.sum(length+a/(4*16384)),jnp.max(z),jnp.max(rupper),jnp.min(selfdist)])
        return penalty,aux

    def fun(self,x):
        pairs,target,selfpairs=self.nearest(x)
        (penalty,aux),gg=self.geo(jnp.asarray(x),pairs,target,selfpairs)
        vals=[]; grads=[]; es=[]; rms=[]
        for sample in self.samples:
            val,grad,e,br=self.field_value(x,sample)
            vals.append(val); grads.append(grad); es.append(e); rms.append(br)
        vals=np.array(vals)
        # Smooth worst-case normalized flux, with every selected sample active.
        temp=self.temperature
        z=np.exp((vals-vals.max())/temp); z/=z.sum()
        val=vals.max()+temp*np.log(np.exp((vals-vals.max())/temp).sum())+float(penalty)
        grad=np.asarray(gg)+z@np.array(grads)
        if not np.isfinite(val) or not np.isfinite(grad).all(): raise RuntimeError('nonfinite objective or gradient')
        self.calls+=1
        self.last=dict(call=self.calls,seconds=time.monotonic()-self.started,objective=val,penalty=float(penalty),e=max(es),rms=rms,geometry=np.asarray(aux).tolist(),samples=self.samples)
        if self.calls%10==0: print(json.dumps(self.last),flush=True)
        return val,grad

    def screen(self,x):
        values=[]
        for k in range(128):
            e,rms=self.field_value(x,k,False); values.append(e)
        result=dict(e=values,nominal=self.field_value(x,-1,False),worst=int(np.argmax(values)),J=max(values))
        print('SCREEN '+json.dumps(result),flush=True)
        return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='assets/reference/coils_nominal.json'); p.add_argument('--output',default='coils.json'); p.add_argument('--iterations',type=int,default=500); p.add_argument('--phases',type=int,default=3); p.add_argument('--nphi',type=int,default=64); p.add_argument('--ntheta',type=int,default=32); p.add_argument('--count',type=int,default=128); p.add_argument('--section',type=int,default=3); p.add_argument('--check-gradient',action='store_true'); p.add_argument('--screen-only',action='store_true'); p.add_argument('--robust-start',action='store_true'); p.add_argument('--method',choices=['L-BFGS-B','SLSQP'],default='L-BFGS-B'); p.add_argument('--variables',choices=['all','rotation'],default='all'); a=p.parse_args()
    data=ev.read_submission(Path(a.input),CASE); s=Search(data,a.nphi,a.ntheta,a.count,a.section); x=s.x0
    if a.screen_only:
        result=s.screen(x); Path(a.output+'.screen.json').write_text(json.dumps(result,indent=2)); return
    if a.robust_start:
        initial_screen=s.screen(x); s.samples=[-1]+np.argsort(initial_screen['e'])[-3:].tolist()
    if a.check_gradient:
        val,grad=s.fun(x); direction=np.random.default_rng(42).normal(size=len(x)); direction/=np.linalg.norm(direction)
        h=1e-5; fd=(s.fun(x+h*direction)[0]-s.fun(x-h*direction)[0])/(2*h); exact=grad@direction
        print('GRADIENT',val,fd,exact,abs(fd-exact)/max(1,abs(fd)),flush=True)
        if not np.isfinite(exact) or abs(fd-exact)>1e-4*max(1,abs(fd)): raise RuntimeError('gradient mismatch')
    bounds=[(-100,100)]*204+[(-np.pi,np.pi)]*20+[(1,24)]*3
    if a.method=='SLSQP' or a.variables=='rotation': s.temperature=.005
    for phase in range(a.phases):
        def callback(xk):
            write(a.output,xk)
        if a.variables=='rotation':
            anchor=x.copy(); indices=np.arange(204,224)
            def expand(v):
                out=anchor.copy(); out[indices]=v; return out
            def rotation_objective(v):
                value,gradient=s.fun(expand(v)); return value,gradient[indices]
            result=minimize(rotation_objective,x[indices],method='L-BFGS-B',jac=True,bounds=[bounds[k] for k in indices],callback=lambda v:callback(expand(v)),options=dict(maxiter=a.iterations,maxcor=20,ftol=1e-13,gtol=1e-7,maxls=30))
            result.x=expand(result.x)
        elif a.method=='L-BFGS-B':
            result=minimize(s.fun,x,method='L-BFGS-B',jac=True,bounds=bounds,callback=callback,options=dict(maxiter=a.iterations,maxcor=50,ftol=1e-13,gtol=2e-6,maxls=30))
        else:
            auxfun=jit(lambda z,p,t,sp:s._geometry(z,p,t,sp)[1])
            auxjac=jit(jax.jacrev(auxfun,argnums=0))
            signs=np.array([1,-1,-1,1,1,-1,-1,-1,1])
            limits=np.array([2.508,.547,.245,.860-2*RPACK,2.410,199.95,9.985,21.985,.115])
            def constraints(z,jacobian=False):
                p,t,sp=s.nearest(z)
                if jacobian: return signs[:,None]*np.asarray(auxjac(jnp.asarray(z),p,t,sp))
                return signs*(np.asarray(auxfun(jnp.asarray(z),p,t,sp))-limits)
            def flux_only(z):
                val,grad=s.fun(z); p,t,sp=s.nearest(z); (pen,aux),gg=s.geo(jnp.asarray(z),p,t,sp)
                return val-float(pen),grad-np.asarray(gg)
            local_bounds=[(max(lo,float(v)-(.12 if k<204 else .3)),min(hi,float(v)+(.12 if k<204 else .3))) for k,(v,(lo,hi)) in enumerate(zip(x,bounds))]
            result=minimize(flux_only,x,method='SLSQP',jac=True,bounds=local_bounds,callback=callback,constraints=[dict(type='ineq',fun=constraints,jac=lambda z:constraints(z,True))],options=dict(maxiter=a.iterations,ftol=1e-10,disp=True))
        x=result.x; write(a.output,x); write(str(Path(a.output).with_suffix(''))+f'.phase-{phase}.json',x)
        print('PHASE',phase,result.message,result.fun,flush=True)
        measured=s.screen(x); Path(a.output+'.screen.json').write_text(json.dumps(measured,indent=2))
        worst=np.argsort(measured['e'])[-3:].tolist(); s.samples=[-1]+worst
    Path(a.output+'.optimization.json').write_text(json.dumps(dict(last=s.last,success=bool(result.success),message=str(result.message)),indent=2))
if __name__=='__main__': main()
