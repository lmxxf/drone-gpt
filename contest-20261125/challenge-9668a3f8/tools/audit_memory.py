"""Replay actual delayed/noisy observations to audit memory and original commands."""
import sys,json,argparse
from pathlib import Path
import numpy as np
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'));import eval as ev
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--report',default=str(ROOT/'evaluation_report.json'));args=parser.parse_args();path=Path(args.report);report=json.load(open(path));case=ev.read_json(ROOT/'workspace/case.json');ref=ev.read_json(ROOT/'workspace/reference/baseline.json')['development'];opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1;s=ort.InferenceSession(report['artifact']['path'],sess_options=opts,providers=['CPUExecutionProvider']);name=s.get_inputs()[0].name;maximum=np.zeros(32);error=0.;delta=0.;eferr=0.;ererr=0.;n=0
 for pair in report['pairs']:
  seed=pair['seed'];noise=np.clip(np.random.Generator(np.random.PCG64(ev.scenario(seed,case)['sensor_seed'])).standard_normal((60,9)),-3,3)*np.array(case['observation']['noise_relative_sigma'])
  for level in ['coarse','fine']:
   d=Path(pair['run_directory'])/str(seed)/level;h=ev.read_json(d/'trace.json');a=ev.read_json(d/'actions.json');mem=np.zeros(32);applied=np.array([28.,15.]);command=applied.copy();ef=er=0.;prevp=prevdelayed=0.;previous_target=0.
   for k in range(60):
    target=float(np.interp(h[k]['time_s'],ref['reference_time_s'],ref['reference_fusion_W']))/1e6;x=ev.features(h,k,target,applied,command,mem,noise,case).astype(np.float32);y=s.run(None,{name:x.reshape(1,48)})[0].reshape(-1);delayed=float(x[11]) if k==0 else previous_target
    if float(x[0])>.75:ef+=float(x[1])*.5*(prevp+float(x[2]));er+=float(x[1])*.5*(prevdelayed+delayed)
    eferr=max(eferr,abs(float(y[2])*100-ef)*1e6);ererr=max(ererr,abs(float(y[3])*100-er)*1e6);prevp=float(x[2]);prevdelayed=delayed;previous_target=float(x[11]);error=max(error,float(np.max(abs(y[:2]-a[k]['command_MW']))));delta=max(delta,float(np.max(abs(y[:2]-command))));mem=y[2:].astype(float);maximum=np.maximum(maximum,abs(mem));assert np.isfinite(y).all() and max(abs(mem))<=100;command=np.array(a[k]['command_MW']);applied=np.array(a[k]['applied_end_MW']);n+=1
 assert error<=2e-6,(error,'original ONNX commands differ on replay');assert delta<=2+1e-6
 audit={'controller_calls':n,'maximum_command_replay_error_MW':error,'maximum_single_command_change_MW':delta,'maximum_absolute_memory_by_index':maximum.tolist(),'maximum_fusion_integral_float32_rounding_error_J':eferr,'maximum_target_integral_float32_rounding_error_J':ererr,'scope':'Exact original delayed/noisy observations and commands; memory is a delayed diagnostic estimator, not native-step ground truth'};report['memory_replay_audit']=audit;report['offline_driver_parity']={p:ev.read_json(ROOT/'runs'/p) for p in ['fast-baseline-coarse.json','fast-baseline-fine.json','fast-cold-parity.json','block-parity/parity.json']};path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(audit,indent=2))
