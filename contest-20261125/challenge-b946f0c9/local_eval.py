#!/usr/bin/env python3
"""Local path adapter; all transport, statistics, precision and scoring use workspace/eval.py unchanged."""
import importlib.util
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='4'
os.environ['LD_LIBRARY_PATH']=str(ROOT/'runtime/sysroot/usr/lib/aarch64-linux-gnu')+':'+os.environ.get('LD_LIBRARY_PATH','')
spec=importlib.util.spec_from_file_location('blanket_official_eval', ROOT/'workspace/eval.py')
eval=importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval)
def load_local(root, case_path):
    case=eval.read_json(case_path)
    baseline=eval.read_json(root/'reference/baseline.json')
    lock=eval.read_json(root/'runtime-lock.json')
    assert baseline['state']=='calibrated' and lock['state']=='verified'
    assert baseline['case_sha256']==eval.sha256(case_path)
    assert baseline['runtime_lock_sha256']==eval.sha256(root/'runtime-lock.json')
    assert baseline['reference_sha256']==eval.sha256(root/'reference_configurations/uniform.json')
    for record in baseline['evidence'].values():
        assert record['status']=='passed'
        for artifact in record['artifacts']:
            assert eval.sha256(root/artifact['path'])==artifact['sha256']
    for record in eval.read_json(root/'provenance/nuclear-data.json')['neutron_files']:
        assert eval.sha256(ROOT/'runtime/data'/Path(record['path']).name)==record['sha256']
    import openmc
    assert openmc.__version__==case['openmc']['version']
    runtime={'cross_sections_xml':str(ROOT/'runtime/data/cross_sections.xml'),
             'openmc_executable':str(ROOT/'runtime/native-env/bin/openmc')}
    return case,runtime,baseline
eval.load_trusted=load_local
if __name__=='__main__': eval.main()
