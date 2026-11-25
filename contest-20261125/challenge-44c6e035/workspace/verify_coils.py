#!/usr/bin/env python3
"""Run unmodified official measurement, then public-bank score, retaining evidence."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import eval as ev
p=argparse.ArgumentParser();p.add_argument('submission');p.add_argument('name');a=p.parse_args()
root=Path(__file__).resolve().parent
work=root/'runs'/a.name; reward=root/'results'/a.name
if work.exists() and any(work.iterdir()): raise RuntimeError('refuse to reuse nonempty run')
env=os.environ.copy();env.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',JAX_PLATFORMS='cpu')
cmd=[sys.executable,str(root/'eval.py'),a.submission,'--assets',str(root/'assets'),'--scenarios',str(root/'assets/search_scenarios.npz'),'--workdir',str(work),'--reward-dir',str(reward),'--measure']
result=subprocess.run(cmd,env=env)
details=json.loads((reward/'details.json').read_text())
if not details['valid'] or details.get('metrics') is None:
    print(json.dumps(details,indent=2));sys.exit(result.returncode or 2)
case=json.loads((root/'assets/case.json').read_text());calibration=json.loads((root/'assets/search-baseline.json').read_text())
score,feasible,violations=ev.score(details['metrics'],calibration['J_ref_search'],case['constraints'])
report=dict(scope='local official --measure, scored with public search-bank reference; not formal held-out score',feasible=feasible,score=score,J_ref_search=calibration['J_ref_search'],violations=violations,measurement=details)
(reward/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
print(json.dumps(dict(feasible=feasible,score=score,metrics=details['metrics'],violations={k:v for k,v in violations.items() if v}),indent=2))
