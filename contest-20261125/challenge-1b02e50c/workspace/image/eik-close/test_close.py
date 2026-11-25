from pathlib import Path
from types import SimpleNamespace
import hashlib,gc,json
import numpy as np
from netCDF4 import Dataset
from gx_geometry.eik_files import write_eik_netcdf
r=Path('/opt/gx/build-evidence/eik-close')
with Dataset(r/'source-eik.nc') as d:data={k:np.array(v[:]).copy() for k,v in d.variables.items()}
fl=SimpleNamespace(nl=len(data['theta']),sigma_Bxy=data['sigma_Bxy'],kxfac=data['kxfac'],Rmajor_p=data['Rmaj'],iota=1/data['q'],shat=data['shat'],alpha=data['alpha'],phi_center=data['zeta_center'],nfp=data['nfp'],z=data['theta'])
for k in ['bmag','gradpar','grho','gds2','gds21','gds22','gbdrift','gbdrift0','cvdrift','cvdrift0']:setattr(fl,k,data[k])
gc.collect();gc.disable();out=r/'closed-eik.nc';write_eik_netcdf(fl,str(out));before=hashlib.sha256(out.read_bytes()).hexdigest();gc.collect();after=hashlib.sha256(out.read_bytes()).hexdigest();gc.enable();assert before==after
with Dataset(out) as d:
 assert all(not np.ma.getmaskarray(v[:]).any() for v in d.variables.values())
 assert np.all(d.variables['Zplot'][:]==0)
 for k in ['theta','bmag','gradpar','grho','gds2','gds21','gds22','gbdrift','gbdrift0','cvdrift','cvdrift0']:assert np.array_equal(d.variables[k][:],data[k]),k
native=hashlib.sha256(Path('/opt/gx/bin/gx').read_bytes()).hexdigest();assert native=='0f14d22252f39f1d3d79af81a180871689396a1c6aab58c41ec91e5557670ca0'
(r/'test-result.json').write_text(json.dumps({'status':'passed','before_gc_sha256':before,'after_gc_sha256':after,'all_variables_written':True,'physical_arrays_identical':True,'native_binary_sha256':native},indent=2)+'\n')
