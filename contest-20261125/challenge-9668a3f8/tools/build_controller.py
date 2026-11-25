"""Export a static causal PI energy controller, ONNX opset 18, IR 10."""
import json, argparse, os
from pathlib import Path
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
class Graph:
    def __init__(self):self.nodes=[];self.weights=[];self.i=0
    def const(self,x,dtype=np.float32):
        self.i+=1;n=f'c{self.i}';a=np.asarray(x,dtype=dtype)
        if a.ndim==0:a=a.reshape(1,1)
        self.weights.append(nh.from_array(a,n));return n
    def op(self,op,*xs,**attrs):
        self.i+=1;n=f'v{self.i}';self.nodes.append(h.make_node(op,list(xs),[n],**attrs));return n
    def add(self,a,b):return self.op('Add',a,b)
    def sub(self,a,b):return self.op('Sub',a,b)
    def mul(self,a,b):return self.op('Mul',a,b)
    def scale(self,a,k):return self.mul(a,self.const(k))
    def col(self,i):return self.op('Gather','input',self.const([i],np.int64),axis=1)
    def clip(self,a,lo,hi):return self.op('Min',self.op('Max',a,self.const(lo)),self.const(hi))

def build(cfg,dest):
    g=Graph();c=g.const
    use_curves='coarse_commands_MW' in cfg
    if use_curves:
        assert not cfg.get('ac_gain')
        for key in ['coarse_commands_MW','fine_commands_MW']:
            a=np.asarray(cfg[key],np.float32);assert a.shape==(60,2) and np.isfinite(a).all()
            assert (a>=[0,0]).all() and (a<=[33,20]).all()
            assert np.max(np.abs(np.diff(np.vstack(([28,15],a)),axis=0)))<=2+1e-6
    t,dt,p,w=g.col(0),g.col(1),g.col(2),g.col(3)
    target=g.col(11);prev_g=g.col(14);prev_e=g.col(15)
    m=[g.col(i+16) for i in range((16 if 'feedback_hold_s' in cfg else 14) if use_curves else 17 if cfg.get('ac_gain') else 12)]
    first=g.op('Less',t,c(.25));integrating=g.op('Greater',t,c(.75));initialize_filter=g.op('Less',t,c(1.25))
    delayed_target=g.op('Where',first,target,g.scale(m[3],10))
    delta_f=g.scale(g.mul(dt,g.add(g.scale(m[2],10),p)),.005)
    delta_r=g.scale(g.mul(dt,g.add(g.scale(m[4],10),delayed_target)),.005)
    ef=g.add(m[0],g.op('Where',integrating,delta_f,c(0)))
    er=g.add(m[1],g.op('Where',integrating,delta_r,c(0)))
    # Matched delayed trapezoids cancel the common startup quadrature bias in the debt.
    # Both estimates use delayed measurements; this is not an exact native integral.
    err=g.sub(delayed_target,p)
    ep=g.op('Where',integrating,err,c(0))
    epf=g.op('Where',initialize_filter,ep,g.add(g.scale(m[5],10*(1-cfg['filter'])),g.scale(ep,cfg['filter'])))
    # Energy deficit in MJ, including modest planned surplus accumulation.
    deficit=g.add(g.scale(g.sub(er,ef),100),g.scale(g.op('Max',g.sub(t,c(.5)),c(0)),cfg['margin_MW']))
    integral=g.clip(g.scale(deficit,cfg['ki']),-10,12)
    # Heat tracking becomes active near the end, leaving noise tolerance at 98% Wref.
    werr=g.sub(c(cfg['W_target_MJ']),w)
    wf=g.op('Where',initialize_filter,werr,g.add(g.scale(m[6],10*.7),g.scale(werr,.3)))
    terminal=g.clip(g.scale(g.sub(t,c(20)),.1),0,1)
    wt=g.scale(g.mul(terminal,g.op('Max',wf,c(0))),cfg['kw'])
    if use_curves:
        q_update=g.op('Less',t,c(7.5))
        qsum=g.op('Where',q_update,g.add(m[12],g.scale(g.sub(g.col(7),c(cfg['q_mid'])),100)),m[12])
        qcount=g.op('Where',q_update,g.add(m[13],c(1)),m[13])
        curve_gain=g.op('Sigmoid',g.scale(g.op('Div',qsum,g.op('Max',qcount,c(1))),40.))
        idx=g.op('Cast',g.clip(g.scale(t,2),0,59),to=T.INT64)
        idx=g.op('Reshape',idx,c([1],np.int64))
        coarse=g.op('Gather',c(np.asarray(cfg['coarse_commands_MW'],np.float32)),idx,axis=0)
        fine=g.op('Gather',c(np.asarray(cfg['fine_commands_MW'],np.float32)),idx,axis=0)
        command_ff=g.add(fine,g.mul(curve_gain,g.sub(coarse,fine)))
        pg_ff=g.op('Gather',command_ff,c([0],np.int64),axis=1);pe_ff=g.op('Gather',command_ff,c([1],np.int64),axis=1)
        startup_ff=g.add(pg_ff,pe_ff)
    else:
        startup_ff=g.sub(c(cfg.get('startup_MW',43)),g.scale(g.clip(g.sub(t,c(cfg.get('startup_hold_s',0))),0,cfg.get('startup_ramp_s',1)),(cfg.get('startup_MW',43)-cfg['feedforward_MW'])/cfg.get('startup_ramp_s',1))) if cfg.get('startup_hold_s') is not None else c(cfg['feedforward_MW'])
    heat=g.add(startup_ff,g.add(g.scale(g.add(epf,c(cfg['margin_MW'])),cfg['kp']),g.add(integral,wt)))
    if 'feedback_lower_MW' in cfg:
        heat=g.add(startup_ff,g.clip(g.sub(heat,startup_ff),cfg['feedback_lower_MW'],cfg['feedback_upper_MW']))
    if use_curves:
        heat=g.op('Where',g.op('GreaterOrEqual',t,c(cfg.get('feedback_start_s',16.))),heat,startup_ff)
    if use_curves and 'feedback_hold_s' in cfg:
        feedback_correction=g.op('Where',g.op('Less',t,c(cfg['feedback_hold_s'])),g.sub(heat,startup_ff),g.scale(m[15],10))
        heat=g.add(startup_ff,feedback_correction)
    # Optional slow feedforward change, continuous in time.
    heat=g.add(heat,g.scale(g.clip(g.scale(t,1/30),0,1),cfg.get('ff_slope',0)))
    if 'openloop_until_s' in cfg:
        heat=g.op('Where',g.op('Less',t,c(cfg['openloop_until_s'])),startup_ff,heat)
    if cfg.get('terminal_boost_MW',0):
        heat=g.add(heat,g.scale(g.clip(g.scale(g.sub(t,c(cfg['terminal_start_s'])),1/cfg['terminal_ramp_s']),0,1),cfg['terminal_boost_MW']))
    heat=g.clip(heat,cfg.get('minimum_heat_MW',0),53)
    if use_curves:
        correction=g.sub(heat,startup_ff)
        if cfg.get('negative_feedback_channel')=='G':
            pg=g.add(pg_ff,g.op('Min',correction,c(0)));pe=g.add(pe_ff,g.op('Max',correction,c(0)))
        else:
            pg=pg_ff;pe=g.add(pe_ff,correction)
    else:
        pg_limit=g.op('Min',c(cfg['PG_max']),g.add(c(28),g.scale(g.op('Max',g.sub(t,c(.5)),c(0)),2))) if cfg.get('startup_hold_s') is not None and not cfg.get('fast_PG',False) else c(cfg['PG_max'])
        pg=g.op('Min',g.op('Max',heat,c(0)),pg_limit)
        pe=g.clip(g.sub(heat,pg),0,20)
    pe_base=pe
    if cfg.get('ac_gain'):
        ac_init=g.op('LessOrEqual',t,c(2.))
        slow=g.op('Where',ac_init,p,g.add(g.scale(m[12],10*(1-cfg['ac_slow_alpha'])),g.scale(p,cfg['ac_slow_alpha'])))
        fast=g.op('Where',ac_init,p,g.add(g.scale(m[13],10*(1-cfg['ac_fast_alpha'])),g.scale(p,cfg['ac_fast_alpha'])))
        q_update=g.op('Less',t,c(7.5))
        qsum=g.op('Where',q_update,g.add(m[15],g.scale(g.sub(g.col(7),c(cfg['q_mid'])),100)),m[15])
        qcount=g.op('Where',q_update,g.add(m[16],c(1)),m[16])
        qgain=g.op('Sigmoid',g.scale(g.op('Div',qsum,g.op('Max',qcount,c(1))),-40.))
        ac_active=g.op('GreaterOrEqual',t,c(cfg['ac_start_s']))
        ac=g.op('Where',ac_active,g.sub(g.scale(g.mul(qgain,g.sub(fast,slow)),cfg['ac_gain']),g.scale(m[14],10/cfg['ac_balance_tau_s'])),c(0))
        pe=g.add(pe,ac)
    delta=cfg.get('slew_delta_MW',2.)
    pg=g.add(prev_g,g.clip(g.sub(pg,prev_g),-delta,delta));pe=g.add(prev_e,g.clip(g.sub(pe,prev_e),-delta,delta))
    pg=g.clip(pg,0,33);pe=g.clip(pe,0,20)
    memory=[ef,er,g.scale(p,.1),g.scale(target,.1),g.scale(delayed_target,.1),g.scale(epf,.1),g.scale(wf,.1),g.scale(integral,.1),g.scale(heat,.1),g.scale(w,.1),g.scale(deficit,.01),g.scale(wt,.1)]
    if use_curves:
        memory += [qsum,qcount,curve_gain]
        if 'feedback_hold_s' in cfg:memory += [g.scale(feedback_correction,.1)]
    if cfg.get('ac_gain'):
        balance=g.op('Where',ac_active,g.add(g.scale(m[14],10),g.mul(dt,g.sub(pe,pe_base))),c(0))
        memory += [g.scale(slow,.1),g.scale(fast,.1),g.scale(balance,.1),qsum,qcount]
    memory=[g.clip(x,-100,100) for x in memory]+[c(np.zeros((1,32-len(memory)),np.float32))]
    out=g.op('Concat',pg,pe,*memory,axis=1)
    g.nodes.append(h.make_node('Identity',[out],['output']))
    model=h.make_model(h.make_graph(g.nodes,'CausalEnergyPI',[h.make_tensor_value_info('input',T.FLOAT,[1,48])],[h.make_tensor_value_info('output',T.FLOAT,[1,34])],g.weights),opset_imports=[h.make_opsetid('',18)],ir_version=10)
    model=onnx.shape_inference.infer_shapes(model,strict_mode=True,data_prop=True);onnx.checker.check_model(model,full_check=True)
    tmp=Path(str(dest)+'.tmp');onnx.save(model,tmp);os.replace(tmp,dest)
    print(json.dumps({'path':str(dest),'nodes':len(model.graph.node),'bytes':Path(dest).stat().st_size,'parameters':{k:v for k,v in cfg.items() if k not in ['coarse_commands_MW','fine_commands_MW']}}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');a=p.parse_args();build(json.load(open(a.config)),a.output)
