#!/usr/bin/env python3
"""Package a complete native formal certificate and its exact seven-value design."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if not (ROOT/'workspace/physics').exists():ROOT=ROOT.parent
sys.path.insert(0,str(ROOT/'workspace/physics'))
import eval as ev
from report_plasma import summarize

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def verify(work,design):
    p=read(ROOT/'workspace/physics/assets/protocol.json');r=read(work/'details.json')
    assert r['status']=='feasible' and r['mode']=='formal' and r['all_cases_evaluated']
    assert set(r['cases'])=={s['name'] for s in p['scenarios']}
    runs=integrations=0;pressures=[];jobs={}
    for scenario in p['scenarios']:
        levels=r['cases'][scenario['name']];assert set(levels)=={'coarse','fine'}
        for level,c in levels.items():
            job_path=work/f"{scenario['name']}-{level}"/'job.json';job=read(job_path)
            assert all(job[k]==p[k] for k in ('R0_m','B0_T','volume_m3','current_A'))
            assert job['shape']=={k:design[k] for k in p['shape_bounds']}
            assert job['axis_pressure_Pa']==design['axis_pressure_Pa']
            assert job['ffprime_hollow']==scenario['ffprime_hollow'] and job['resolution']==p[level]
            jobs[str(job_path.relative_to(ROOT))]=sha(job_path)
            ev.check_equilibrium(c['equilibrium'],p,design['axis_pressure_Pa'])
            assert not ev.check_stability(c['stability'],p)
            expected={str(v) for v in p['phases_'+level] if v!=0}
            assert set(c['phases'])==expected and not c['violations']
            native=[c['stability'],*c['phases'].values()]
            for phase in c['phases'].values():assert not ev.check_stability(phase,p,modes=[1])
            for block in native:
                for m in block['modes'].values():
                    assert m['normal_termination'] and m['exit_code']==0
                    runs+=1;integrations+=m['ballooning_integrations']
            pressures.append(c['equilibrium']['average_pressure_Pa'])
        c,f=levels['coarse'],levels['fine'];a=p['acceptance']
        assert abs(f['equilibrium']['average_pressure_Pa']/c['equilibrium']['average_pressure_Pa']-1)<=a['pressure_refinement_relative']
        for n in map(str,p['modes']):
            old,new=c['stability']['modes'][n],f['stability']['modes'][n]
            assert abs(old['globals']['qmin']-new['globals']['qmin'])<=a['q_refinement_absolute']
            assert abs(old['minimum_energy'][0]-new['minimum_energy'][0])<=a['energy_refinement_absolute']+a['energy_refinement_relative']*abs(new['minimum_energy'][0])
    assert runs==90
    pressure=min(pressures);score=pressure/(pressure+p['score_pressure_scale_Pa'])
    assert pressure==r['worst_average_pressure_Pa'] and abs(score-r['reward'])<1e-14
    return {'native_dcon_runs':runs,'completed_ballooning_integrations':integrations,'job_hashes':jobs}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--formal',type=Path,required=True);ap.add_argument('--baseline',type=Path,default=ROOT/'runs/baseline-formal');ap.add_argument('--outputs',type=Path,default=ROOT/'submission_outputs');a=ap.parse_args()
    p=read(ROOT/'workspace/physics/assets/protocol.json');d=ev.read_design(a.candidate,p)
    work=a.formal.resolve();assert d==read(work.parent/(work.name+'.json'))
    verification=verify(work,d);base_design=read(next((ROOT/'datasets').glob('*/1/resources/plasma.json')));verify(a.baseline.resolve(),base_design)
    final=summarize(work/'details.json');base=summarize(a.baseline/'details.json');base_search=summarize(ROOT/'runs/baseline-search/details.json')
    convergence=read(ROOT/'runs/climb_progress.json') if (ROOT/'runs/climb_progress.json').exists() else None
    final_search=None
    stem=work.name.removesuffix('-formal');search_work=ROOT/'runs'/(stem+'-search');search_input=ROOT/'runs'/(stem+'.json')
    if (search_work/'details.json').exists() and search_input.exists() and read(search_input)==d:
        final_search=summarize(search_work/'details.json')
    out=a.outputs.resolve();out.mkdir(exist_ok=False)
    shutil.copy2(a.candidate,out/'plasma.json')
    shutil.copy2(ROOT/'workspace/physics/assets/protocol.json',out/'protocol.json')
    for name in ['solve_plasma.py','report_plasma.py','package_submission.py','build_native.sh','env.sh']:shutil.copy2(ROOT/name,out/name)
    for name in ['build-notes.txt','requirements-lock.txt','evaluator_integrity.json']:shutil.copy2(ROOT/'native'/name,out/name)
    shutil.copy2(work/'details.json',out/'formal_details.json');shutil.copy2(work/'reward.json',out/'formal_reward.json');shutil.copy2(a.baseline/'details.json',out/'baseline_formal_details.json')
    shutil.copy2(next((ROOT/'datasets').glob('*/1/resources/upstream/SOURCES.md')),out/'UPSTREAM_SOURCES.md')
    save(out/'evaluation_report.json',{'design':d,'formal':final,'baseline_formal':base,'baseline_search':base_search,'final_candidate_search':final_search,'optimizer_convergence':None if convergence is None else {k:v for k,v in convergence.items() if k!='incumbent'},'independent_packaging_checks':verification})
    records=[]
    for path in sorted((ROOT/'runs/screen').glob('*/screen.json')):
        r=read(path);records.append({**{k:v for k,v in r.items() if k!='cases'},'completed_native_modes_by_scenario':{name:sorted(map(int,c['stability']['modes'])) for name,c in r.get('cases',{}).items()},'case_violations':{name:c['violations'] for name,c in r.get('cases',{}).items()},'source_report':str(path.relative_to(ROOT))})
    probes=[read(path) for path in sorted((ROOT/'runs/equilibrium-probes').glob('*/probe.json'))]
    save(out/'search_results.json',{'screens':records,'equilibrium_only_high_pressure_probes':probes,'note':'Screen-only results are not formal feasibility certificates. Library-experiment records are explicitly invalidated.'})
    lines=[f"正式本地验收：feasible=True，p_bar={final['worst_average_pressure_Pa']:.9f} Pa，S={final['reward']:.12f}。",f"起始设计正式基线：p_bar={base['worst_average_pressure_Pa']:.9f} Pa，S={base['reward']:.12f}。平均压力提高 {(final['worst_average_pressure_Pa']/base['worst_average_pressure_Pa']-1)*100:.4f}%。",f"独立打包复核了全部六个平衡、{verification['native_dcon_runs']} 次正常终止的 DCON 和 {verification['completed_ballooning_integrations']} 次完整气球模积分。所有 n=1–5、规定相位、Mercier、q 下限、平衡残差和粗细网格一致性均通过。",'','| 工况 | 平均压力 Pa | q_min | 最小总能量（含相位） | Mercier 最大值 | 气球模节点 |','|---|---:|---:|---:|---:|---:|']
    for name,c in final['cases'].items():lines.append(f"| {name} | {c['average_pressure_Pa']:.6f} | {c['q_min']:.8f} | {c['minimum_energy_all_phases']:.8f} | {c['mercier_max']:.8f} | {c['ballooning_nodes_max']} |")
    lines+=['','搜索使用原生粗网格筛选、七维空间填充、坐标爬山、形状/压力联合边界探测及高压力区原生平衡审计。最终仅从完整 formal 通过的设计中选择本次搜索得到的最高平均压力解；未证明连续设计空间的全局最优。','', '公开 evaluator、equilibrium、dcon、geometry 与 protocol 均逐字节匹配 background.zip。原生源码包的 SHA-256 已核对。构建环境及兼容性修改详见 build-notes.txt。OpenBLAS 对照失败的运行已隔离；最终报告使用参考 BLAS 重跑结果。','', '在原题目录运行 `source env.sh`，再执行 `taskset -c 4-7 .venv/bin/python -I workspace/physics/eval.py submission_outputs/plasma.json --binary native/GPEC/bin/dcon --workdir runs/recheck-final --mode formal`。结果目录必须尚不存在。','', '重建入口为 build_native.sh（Ubuntu Noble ARM64）；优化入口为 solve_plasma.py，支持 --latin、--audit、--climb、--pressure-boundary、--de、--local 和 --formal。formal 标签与候选绑定，筛选记录不能替代正式验收。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    manifest={'design_sha256':sha(out/'plasma.json'),'formal_details_sha256':sha(out/'formal_details.json'),'native_dcon_sha256':sha(ROOT/'native/GPEC/bin/dcon'),'native_oftpy_sha256':sha(ROOT/'native/oft-build/bin/liboftpy.so'),'source_archives':{path.name:sha(path) for path in (ROOT/'datasets').glob('*/1/resources/upstream/*source.tar.gz')},'files':{path.name:sha(path) for path in out.iterdir() if path.is_file()}}
    save(out/'manifest.json',manifest)
    print(json.dumps({'outputs':str(out),'reward':final['reward'],'pressure':final['worst_average_pressure_Pa'],'verification':verification},indent=2))
if __name__=='__main__':main()
