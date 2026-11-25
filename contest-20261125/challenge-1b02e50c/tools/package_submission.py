"""Package only measured evidence and explicitly record missing verifications."""
from pathlib import Path
import os,sys,json,shutil,hashlib
os.sched_setaffinity(0,{16,17,18,19})
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'workspace'))
import magnetic_eval as m
from importlib.metadata import version
out=ROOT/'submission_outputs';out.mkdir(exist_ok=True)
def read(p):return json.loads((ROOT/p).read_text())
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
best=read('results/refined-best.json');baseline=read('results/proxy-baseline.json')
boundary=ROOT/'refined_boundary.json';b=m.read_boundary(boundary)
m.M_MAX=m.N_MAX=5;m.NFP_MAX=2;b=m.read_boundary(boundary)
g=m.geometry(b);g=m.geometry(b.scaled(1.7/g['minor_radius']))
metrics=dict(volume_relative_error=abs(g['volume']/581.8735601800271-1),minor_radius_m=g['minor_radius'],kappa_max=g['kappa_max'],iota_min=best['profile']['iota_min'],well=best['profile']['well'],lgradb_min=best['coil']['lgradb_min'],eps_eff_max=best['neo']['eps_eff_max'])
limits=dict(volume_relative_error=.001,kappa_max=8,iota_min=.3,well=0,lgradb_min=2.4,eps_eff_max=.01)
violations={k:max(0,limits[k]-v) if k in ('iota_min','well','lgradb_min') else max(0,v-limits[k]) for k,v in metrics.items() if k in limits}
report={
 'status':'proxy_optimized_candidate_with_incomplete_formal_physics_validation',
 'boundary_sha256':hashlib.sha256(boundary.read_bytes()).hexdigest(),
 'full_feasible':None,'official_reward':None,'official_nonlinear_J':None,
 'submission_decision':'User explicitly requested immediate submission with particle-verification gap disclosed.',
 'measured_geometry':g,'measured_screening_magnetic_metrics':metrics,'screening_constraint_excess':violations,
 'magnetic_measurement_scope':{
  'equilibrium_backend':'VMEC++ 0.9.4, fixed-boundary stellarator-symmetric vacuum, zero toroidal current',
  'ns':65,'mpol':8,'ntor':6,'force_residuals':best['residuals'],
  'geometry_grid':{'phi_per_period':256,'theta':512,'self_intersection_checked':True},
  'coil_gradient_grid':{'theta':128,'phi':128},
  'neo':'Actual STELLOPT NEO binary, published signed-iota correction, five surfaces and control from magnetic_eval.py',
  'boozer':'Actual booz_xform; adaptive transform tolerance checked by magnetic_eval.run_boozer',
  'field_strength':'B00 axis is rescalable to 5.70 T; full candidate alpha run at that field was not completed',
  'warning':'Zero excess here applies only to measured screening metrics; it is NOT seven-constraint formal feasibility.'},
 'proxy_results':{'baseline_J_proxy':baseline['J_proxy'],'candidate_J_proxy':best['J_proxy'],
  'relative_reduction':1-best['J_proxy']/baseline['J_proxy'],'worst_point':max(best['points'],key=best['points'].get),
  'baseline_points':baseline['points'],'candidate_points':best['points'],
  'model':'Uncalibrated fluid/interchange mixing-length geometry heuristic, NOT linear kinetic GX and NOT nonlinear heat flux.',
  'normalization':'Arbitrary proxy units. Reference evaluated with exactly the same model; official baseline unchanged.',
  'definition_file':'proxy-definition.json','gridcheck':read('results/proxy-gridcheck.json') if (ROOT/'results/proxy-gridcheck.json').exists() else None},
 'particle_verification':{
  'candidate_4096_particles':'NOT COMPLETED','candidate_loss_fraction':None,'candidate_wilson_upper':None,
  'baseline_probe':read('runs/alpha-runtime-benchmark/result.json'),
  'warning':'16-particle zero-loss result is for the PUBLIC BASELINE only and does not certify this candidate or the 4096-particle requirement.'},
 'verification_gaps':[
  '257-surface, full prescribed Fourier-resolution final equilibrium was started but interrupted for immediate submission; no convergence result claimed.',
  'Candidate 4096 alpha particles at s=0.25, 3.5 MeV, 0.2 s were not completed.',
  'Candidate full-resolution NEO, well, iota and L_grad_B constraints remain unverified.',
  'No GX nonlinear trajectory was run; no nonlinear heat-flux suppression or platform score improvement established.',
  'Installed simsopt 1.11.1 differs from locked 1.10.6; screen uses VMEC++ rather than prescribed VMEC2000.'
 ],
 'official_package_audit':{
  'runtime_lock_state':read('workspace/runtime-lock.json')['state'],
  'baseline_state':read('workspace/reference/baseline.json')['state'],
  'J_reference':read('workspace/reference/baseline.json')['J_reference'],
  'task_mapping':'0.75 - atan(max(candidate heat-flux upper bounds))/(2*pi)',
  'bundled_eval_mapping':'0.5 + 0.5*sigmoid(J_reference-max(candidate_upper/reference_lower))',
  'statistics_conflict':'Bundled eval uses reference means in tolerance checks; task specifies candidate means.',
  'original_eval_unchanged':True,'runtime_and_baseline_not_forged':True,
  'baseline_eval_result':read('results/baseline-full/reward.json'),
  'baseline_eval_reason':read('results/baseline-full/details.json')['reason'],
  'private_registry':'Specified fixed-digest image manifest request returned denied / unauthorized.',
  'scope':'Package audit concerns the supplied local evaluator; remote platform evaluation is reported separately after submission.'},
 'search':{'initial_sweep':read('results/search.json'),'refinement_steps':[dict(step=r['elongation_step'],J_proxy=r['J_proxy'],eps_eff_max=r.get('neo',{}).get('eps_eff_max')) for r in read('results/refinement.json')],
  'final_transform':'For all m>0, R Fourier amplitudes multiplied by exp(0.04), Z amplitudes by exp(-0.04); uniform minor-radius normalization and exact R00 volume correction.',
  'selected_by':'Lowest measured J_proxy among tested candidates passing available geometry/profile/coil and actual NEO screening checks; alpha constraint remains unknown.'},
 'reproduction':{'cpu_affinity':[16,17,18,19], 'packages':{k:version(k) for k in ['vmecpp','simsopt','booz-xform','numpy','scipy','netCDF4']},
  'commands':['python solve_turbulence.py --search 6','python solve_turbulence.py --refine'],
  'external_native_requirements':'Native NEO and SIMPLE executables plus shared libraries must be supplied separately. Tools/environment.sh records this machine configuration.',
  'caches':'Search caches keyed by boundary and v2 model directory; remove caches before changing model or numerical resolution.'},
 'sources':['https://arxiv.org/abs/2502.11657','https://arxiv.org/abs/2310.18842','https://arxiv.org/abs/2301.09356']}
copy(boundary,out/'boundary.json');copy(boundary,ROOT/'boundary.json')
for name in ['solve_turbulence.py','validate_native.py','tools/refine_proxy.py','tools/environment.sh']:
 copy(ROOT/name,out/name)
for name in ['magnetic_eval.py','eval.py','case.json','runtime-lock.json','reference/baseline.json','reference_configurations/landreman_paul_2022_QA_well.json']:
 copy(ROOT/'workspace'/name,out/'workspace'/name)
shutil.copytree(ROOT/'tools/src/gx_geometry/gx_geometry',out/'tools/src/gx_geometry/gx_geometry',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
copy(ROOT/'results/proxy-definition.json',out/'proxy-definition.json')
for name,folder in [('baseline',baseline['folder']),('candidate',best['folder'])]:
 for tube in ['T00','T10','T01','T11']:copy(ROOT/folder/(tube+'.nc'),out/'evidence'/name/(tube+'.nc'))
 copy(ROOT/folder/'input.eq',out/'evidence'/name/'input.eq')
 copy(ROOT/folder/'result.json',out/'evidence'/name/'screening-result.json')
for src in ['results/refined-best.json','results/proxy-baseline.json','runs/alpha-runtime-benchmark/result.json','runs/alpha-runtime-benchmark/simple/orbit_exit_code.dat']:
 copy(ROOT/src,out/'evidence'/Path(src).parent.name/Path(src).name)
put(out/'audit_report.json',report)
put(out/'artifact_manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'})
print(json.dumps({'geometry':g,'screen_metrics':metrics,'screen_excess':violations,'proxy_reduction':report['proxy_results']['relative_reduction'],'report':str(out/'audit_report.json')},indent=2))
