"""Identify delayed small-signal native dynamics from paired causal probes."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
gains=[-2.,-1.,1.,2.];data={g:json.load(open(ROOT/'runs'/f'damping-{g}-fine.json')) for g in gains}
rows=[];ys=[];groups=[]
for i,g in enumerate(gains):
 for h in gains[i+1:]:
  p=np.array([r['P_fusion_W']/1e6 for r in data[g]['trace']])-np.array([r['P_fusion_W']/1e6 for r in data[h]['trace']]);u=np.array([r['PE'] for r in data[g]['actions']])-np.array([r['PE'] for r in data[h]['actions']]);X=[];y=[]
  for k in range(21,len(p)):
   X.append([p[k-1],p[k-2],u[k-2],u[k-3]]);y.append(p[k])
  groups.append((str((g,h)),np.array(X),np.array(y)));rows.extend(X);ys.extend(y)
X=np.array(rows);y=np.array(ys)
def fit(X,y):
 c=np.linalg.solve(X.T@X+.01*np.eye(4),X.T@y);err=y-X@c;return c,float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),float(np.sqrt(np.mean(err**2)))
c,r2,rmse=fit(X,y);models=[c];modelrows=[{'source':'all','coefficients':c.tolist(),'R2':r2,'rmse_MW':rmse}]
for name,xi,yi in groups:
 ci,ri,ei=fit(xi,yi)
 modelrows.append({'source':name,'coefficients':ci.tolist(),'R2':ri,'rmse_MW':ei})
 if ri>.75:models.append(ci)
def radius(c,a,b):
 a1,a2,b1,b2=c
 def step(x):
  pc,pp,up,upp,sl,fa,B,old=x;slow=.9*sl+.1*pp;fast=.4*fa+.6*pp;osc=fast-slow;u=a*osc+b*old-B/3
  return np.array([a1*pc+a2*pp+b1*up+b2*upp,pc,u,up,slow,fast,B+.5*u,osc])
 A=np.column_stack([step(np.eye(8)[i]) for i in range(8)])
 return float(max(abs(np.linalg.eigvals(A))))
trials=[]
for a in np.linspace(-1.8,1.8,37):
 for b in np.linspace(-1.8,1.8,37):
  if np.hypot(a,b)>1.8:continue
  rho=max(radius(m,a,b) for m in models);trials.append((rho+.005*np.hypot(a,b),rho,float(a),float(b)))
trials.sort();chosen=[]
for score,rho,a,b in trials:
 if all(np.hypot(a-z['now_gain'],b-z['lag_gain'])>.4 for z in chosen):chosen.append({'now_gain':a,'lag_gain':b,'worst_model_radius':rho})
 if len(chosen)==4:break
result={'models':modelrows,'condition_number':float(np.linalg.cond(X)),'baseline_radii':{'no_AC':[radius(m,0,0) for m in models],'tested_AC':[radius(m,1,0) for m in models]},'candidate_FIR_gains':chosen,'note':'Linear identification and pole-radius predictions are not native validation.'}
(ROOT/'runs/identified-dynamics.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
