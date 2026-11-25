import pathlib,re,subprocess,os
root=pathlib.Path(__file__).resolve().parent
src=next((root/'src').glob('STELLOPT-*'))
build=root/'build-neo';build.mkdir(exist_ok=True)
lib=src/'LIBSTELL/Sources';neo=src/'NEO/Sources'
files=list(neo.glob('*.f90'))+[lib/'Modules'/n for n in ['stel_kinds.f','stel_constants.f','vparams.f','safe_open_mod.f90','mpi_params.f','mpi_inc.f','read_boozer_mod.f90']]+[lib/'Miscel/parse_extension.f',lib/'Miscel/getcarg.f90']+list((lib/'Ezcdf').glob('*.f90'))
# Apply precisely the published signed-iota repair to the local source copy.
p=neo/'flint_bo.f90';s=p.read_text();s=s.replace('ONE / acc_req / iota_bar_fp','ONE / acc_req / ABS(iota_bar_fp)');p.write_text(s)
inc=root/'sysroot/usr/include';ld=root/'sysroot/usr/lib/aarch64-linux-gnu'
mods={};deps={}
for p in files:
 text=subprocess.check_output(['mpif90','-E','-cpp','-DLINUX','-DMPI_OPT','-DNETCDF','-ffree-line-length-none','-ffixed-line-length-none',str(p)],text=True)
 for m in re.findall(r'^\s*module\s+(\w+)',text,re.M|re.I):
  if m.lower()!='procedure':mods[m.lower()]=p
 deps[p]=set(m.lower() for m in re.findall(r'^\s*use\s+(\w+)',text,re.M|re.I))
done=set()
while files:
 ready=[p for p in files if all(m=='mpi' or m not in mods or mods[m] in done or mods[m]==p for m in deps[p])]
 if not ready:raise RuntimeError(str(files))
 for p in ready:
  cmd=['mpif90','-c','-O2','-cpp','-DLINUX','-DMPI_OPT','-DNETCDF','-fallow-argument-mismatch','-ffree-line-length-none','-ffixed-line-length-none','-I'+str(inc),str(p),'-o',p.stem+'.o']
  subprocess.run(cmd,cwd=build,check=True);done.add(p);files.remove(p)
subprocess.run(['mpif90','-o',str(root/'bin/xneo')]+[p.name for p in build.glob('*.o')]+['-L'+str(ld),'-Wl,-rpath,'+str(ld),'-lnetcdff','-lnetcdf'],cwd=build,check=True)
