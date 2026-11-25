"""Plot completed original evaluator traces only."""
import os
os.environ['MPLBACKEND']='Agg'
import json,sys,argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--report',default=str(ROOT/'evaluation_report.json'));parser.add_argument('--output',default=str(ROOT/'evaluation_plot.png'));args=parser.parse_args();report=json.load(open(args.report));ref=json.load(open(ROOT/'workspace/reference/baseline.json'))['development'];pairs=report['pairs'];cold=next(p for p in pairs if p['physical_override'] and p['scenario']['event_start_s']==8.)
run=Path(cold['run_directory'])/str(cold['seed']);fig,axs=plt.subplots(3,2,figsize=(13,11),layout='constrained');colors=['#1b6ca8','#df7e14'];trace=[]
for level,color in zip(['coarse','fine'],colors):
 d=json.load(open(run/level/'trace.json'));native=json.load(open(run/level/'accepted-trace.json'));a=json.load(open(run/level/'actions.json'));trace.append(d);t=np.array([x['time_s'] for x in d]);tn=np.array([x['time_s'] for x in native]);pn=np.array([x['P_fusion_W'] for x in native]);q=np.r_[0,np.cumsum((pn[1:]+pn[:-1])*.5*np.diff(tn))]/1e9
 axs[0,0].plot(t,[x['P_fusion_W']/1e6 for x in d],label=level,color=color);axs[0,1].plot(tn,q,label=level,color=color);axs[1,0].plot(t,[x['W_thermal_total_J']/ref['reference_terminal_W_thermal_J'] for x in d],label=level,color=color)
 ta=[x['time_s'] for x in a];cmd=np.array([x['command_MW'] for x in a]);axs[1,1].step(ta,cmd[:,0],where='post',color=color,label=f'{level} G');axs[1,1].step(ta,cmd[:,1],where='post',color=color,linestyle='--',label=f'{level} E')
axs[0,0].plot(ref['reference_time_s'],np.array(ref['reference_fusion_W'])/1e6,'k:',label='target');axs[0,0].set(ylabel='Fusion power [MW]',title='Cold initial state, maximum impurity, event at 8 s')
axs[0,1].axhline(ref['required_fusion_energy_J']/1e9,color='k',linestyle=':',label='required final Q');axs[0,1].set(ylabel='Native fusion integral [GJ]',title='Accepted-step trapezoidal integration')
axs[1,0].axhline(.98,color='k',linestyle=':',label='terminal minimum');axs[1,0].set(ylabel='W / reference final W',title='Thermal-energy reserve');axs[1,1].set(ylabel='Issued command [MW]',title='Delayed actuators; independent slew limits')
t=np.array([x['time_s'] for x in trace[0]]);gap=np.abs(np.array([x['P_fusion_W'] for x in trace[0]])-np.array([x['P_fusion_W'] for x in trace[1]]))/max(ref['reference_fusion_W'])*100;axs[2,0].plot(t,gap,color='#6c3380',label='coarse/fine difference');axs[2,0].axhline(2,color='k',linestyle=':',label='2% limit');axs[2,0].set(ylabel='Power difference [% reference peak]',title='Official 61 common control samples')
base=report['development']['reference_supplied_heat_J']/1e9;sel=report['development']['worst_supplied_heat_J']/1e9;bars=axs[2,1].bar(['Reference','Selected'],[base,sel],color=['#888888','#23856b']);axs[2,1].bar_label(bars,fmt='%.4f GJ');axs[2,1].set(ylabel='Worst supplied heat [GJ]',title=f"Development score: {report['development']['score']:.6f}",ylim=(0,max(base,sel)*1.15))
for ax in axs.flat:
 ax.grid(alpha=.2)
 if ax is not axs[2,1]:ax.set_xlabel('Time [s]');ax.legend(fontsize=8)
fig.suptitle('Local native physics replay — trusted release gate unavailable on ARM',fontsize=13);fig.savefig(args.output,dpi=180);plt.close(fig)
print(args.output)
