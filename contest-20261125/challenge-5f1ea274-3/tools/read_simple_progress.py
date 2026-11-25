"""Read-only progress and finalized exit records of this challenge's SIMPLE process."""
import pathlib,subprocess,os,struct,json,time,sys,math
os.sched_setaffinity(0,{12,13,14,15})
root=pathlib.Path(__file__).resolve().parents[1]
s=json.loads((root/'workflow_status.json').read_text())
if not s.get('candidate'):raise RuntimeError('No active full-verification candidate')
cwd=(root/s['candidate']).parent/'work/simple';exe=(root/'tools/bin/simple.x').resolve()
pids=[]
for entry in pathlib.Path('/proc').iterdir():
 if not entry.name.isdigit():continue
 try:
  if (entry/'exe').resolve()==exe and (entry/'cwd').resolve()==cwd:pids.append(int(entry.name))
 except (OSError,RuntimeError):continue
if len(pids)!=1:raise RuntimeError(f'Expected exactly one SIMPLE process for this candidate, found {pids}')
pid=pids[0];maps=pathlib.Path(f'/proc/{pid}/maps').read_text().splitlines()
rows=[x.split() for x in maps if x.split()[-1]==str(exe) and x.split()[2]=='00000000']
if len(rows)!=1:raise RuntimeError('Could not establish ELF load address uniquely')
base=int(rows[0][0].split('-')[0],16)
symbols={}
for line in subprocess.check_output(['nm','-S',str(exe)],text=True).splitlines():
 parts=line.split()
 if len(parts)==4:symbols[parts[3]]=int(parts[0],16)
fd=os.open(f'/proc/{pid}/mem',os.O_RDONLY)
def read(name,fmt):
 addr=base+symbols['__progress_monitor_MOD_'+name];n=struct.calcsize(fmt);data=os.pread(fd,n,addr)
 if len(data)!=n:raise RuntimeError('Incomplete read')
 return struct.unpack(fmt,data)[0]
def array(name,fmt,width):
 desc=os.pread(fd,64,base+symbols['__params_MOD_'+name])
 if len(desc)!=64:raise RuntimeError('Incomplete array descriptor')
 ptr=struct.unpack_from('<Q',desc,0)[0];span=struct.unpack_from('<q',desc,32)[0]
 stride,lower,upper=struct.unpack_from('<qqq',desc,40)
 if span!=width or stride!=1 or upper-lower+1!=4096 or ptr==0:raise RuntimeError('Array layout did not match the compiled ABI')
 raw=os.pread(fd,width*4096,ptr)
 if len(raw)!=width*4096:raise RuntimeError('Incomplete terminal-state array read')
 return struct.unpack('<'+str(4096)+fmt,raw)
try:
 done=read('n_done_shared','<q');total=read('n_total','<i');start=read('t_start','<d')
 terminal_times=array('times_lost','d',8);codes=array('orbit_exit_code','i',4)
finally:os.close(fd)
if not 0<=done<=total==4096:raise RuntimeError(f'Invalid observed counters: {done}/{total}')
elapsed=time.monotonic()-start
out={'pid':pid,'completed':done,'total':total,'fraction':done/total,'observation':'read-only progress and finalized exit records; not a final acceptance test','elapsed_s':elapsed if elapsed>0 else None}
known=[i for i,t in enumerate(terminal_times) if t!=-1.0 or math.isnan(t)]
if abs(len(known)-done)>4:raise RuntimeError(f'Terminal records do not agree with the independent completion counter: {len(known)} versus {done}')
lost=sum(codes[i] in (1,2) and terminal_times[i]<=.2 for i in known)
unresolved=sum(not(codes[i]==0 or (codes[i] in (1,2) and terminal_times[i]<=.2)) for i in known)
out.update(recorded_terminal_states=len(known),observed_loss_count_lower_bound=lost,observed_unresolved_lower_bound=unresolved,whole_ensemble_loss_fraction_lower_bound=(lost+unresolved)/4096)
if done>0 and elapsed>0:out['linear_eta_s']=elapsed*(total-done)/done
print(json.dumps(out))
