"""从封存预测复算完整比较；缺任意预登记任务时失败，不输出不完整胜负总表。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from harness.integrations.official_research_task import metric
from harness.runtime.storage import atomic_json


def verdict(project,paper,higher=False):
    difference=project-paper
    if abs(difference)<1e-8:return '平'
    return '胜' if (difference>0)==higher else '负'


def summarize(folder,bundle):
    rows=[]
    for task in ['gas','bearing','engine']:
        outcomes={};identities=[]
        alias_path=folder/task/'accepted-runs.json'
        aliases=json.loads(alias_path.read_bytes()) if alias_path.exists() else {}
        oracle=json.loads((bundle/task/'oracle.json').read_bytes())
        identity=json.loads((bundle/task/'task-identity.json').read_bytes())
        from engineering.cli.workflow import public_maintenance as public
        package=public.read_official_gas(Path(identity['archive'])) if task=='gas' else public.PACKAGES[task][2](Path(identity['archive']))
        answers={r.sample_id:r.label for r in package.validation}
        truth=np.asarray([answers[k] for k in oracle['service_ids']])
        with np.load(bundle/task/'outer-inputs.npz',allow_pickle=False) as z:
            future=z['future']
        future_groups=np.asarray(oracle['service_groups'])[future].tolist()
        for arm in ['project','comind']:
            work=folder/task/aliases.get(arm,arm)
            scores=json.loads((work/'scores.json').read_bytes())
            provenance=work/'driver-provenance.json'
            audit=json.loads(provenance.read_bytes()) if provenance.exists() else {}
            calls=audit.get('actual_model_invocations',scores['model_calls']+audit.get('extra_model_invocations',0))
            frozen=json.loads((work/'input-identity.json').read_bytes())
            identities.append((frozen['bundle_identity_sha256'],frozen['description_sha256'],frozen['max_model_calls']))
            decision=json.loads((work/'decision.json').read_bytes())
            assert sha256((work/'serving/model.pkl').read_bytes()).hexdigest()==decision['candidate']['weights_sha256']
            with np.load(work/'issued.npz',allow_pickle=False) as z:
                assert z['ids'].tolist()==oracle['service_ids']
                np.testing.assert_array_equal(z['actual'],z['direct'] if decision['adopted'] else z['baseline'])
                measured=metric(task,truth,z['actual'],oracle['service_groups'])['score']
                assert abs(measured-scores['scores']['actual']['score'])<1e-10
                after=metric(task,truth[future],z['actual'][future],future_groups)['score']
                parent=metric(task,truth,z['baseline'],oracle['service_groups'])['score']
                parent_after=metric(task,truth[future],z['baseline'][future],future_groups)['score']
            assert calls<=frozen['max_model_calls'], '实际模型调用超出共同预算'
            outcomes[arm]={'score':measured,'calls':calls,'adopted':decision['adopted'],
                           'after_maintenance':after,'baseline':parent,'baseline_after_maintenance':parent_after,
                           'issued':len(truth),'after_maintenance_rows':int(future.sum()),
                           'path':work.relative_to(ROOT).as_posix()}
        assert identities[0]==identities[1], '双方输入或调用上限不同'
        rows.append({'task':task,'scope':'官方开发集',**outcomes,
                     'verdict':verdict(outcomes['project']['score'],outcomes['comind']['score'],task=='gas')})
    from engineering.cli.workflow import london_field_driver as london
    london.configure('2026-07-01')
    service_ids=np.flatnonzero(london.CLOCK>=london.START)
    for site in ['BX2','CD1','CE2']:
        outcomes={};task_files=[];baselines=[]
        case=folder/'london-2026'/site
        alias_path=case/'accepted-runs.json'
        aliases=json.loads(alias_path.read_bytes()) if alias_path.exists() else {}
        truth=london.readings(ROOT/'data/public/london-air-2026/stations.zip',site).to_numpy()
        for arm in ['project','comind']:
            work=case/aliases.get(arm,arm)
            scores=json.loads((work/'scores.json').read_bytes())
            initial=json.loads((work/'initial/training.json').read_bytes())
            for component in initial['components']:
                fitted=Path(component['folder'])/'fit'
                assert sha256((fitted/'model.joblib').read_bytes()).hexdigest()==component['weights_sha256']
                # 组件身份记录的是源码文本；Windows 落盘换行需按读取文本还原。
                assert sha256((fitted/'source.py').read_text(encoding='utf-8').encode()).hexdigest()==component['source_sha256']
            with np.load(work/'issued.npz',allow_pickle=False) as z:
                ids=z['ids'];prediction=z[arm];use=np.isfinite(truth[ids])
                np.testing.assert_array_equal(ids,service_ids)
                baselines.append(z['baseline'].copy())
                measured=float(np.sqrt(np.mean((prediction[use]-truth[ids][use])**2)))
                assert len(ids)==1344 and abs(measured-scores['rmse'][arm])<1e-10
                assert np.isfinite(prediction).all()
                future=np.arange(len(ids))>=28*24
                np.testing.assert_array_equal(prediction[~future],z['baseline'][~future])
                after=float(np.sqrt(np.mean((prediction[use&future]-truth[ids][use&future])**2)))
                parent=float(np.sqrt(np.mean((z['baseline'][use]-truth[ids][use])**2)))
                parent_after=float(np.sqrt(np.mean((z['baseline'][use&future]-truth[ids][use&future])**2)))
                if not scores['adopted']:np.testing.assert_array_equal(prediction,z['baseline'])
            identity=work/'task-identity.json'
            if identity.exists():task_files.append(json.loads(identity.read_bytes())['task_files'])
            provenance=work/'driver-provenance.json'
            audit=json.loads(provenance.read_bytes()) if provenance.exists() else {}
            calls=audit.get('actual_model_invocations',scores.get('total_model_invocations',scores['model_calls'])+audit.get('extra_model_invocations',0))
            assert calls<=48,'实际模型调用超出共同预算'
            # 发布可补训，校验的是最终服务权重，而非确认前旧权重。
            model=work/'serving/input/model.pkl'
            if model.exists():
                checksum=sha256(model.read_bytes()).hexdigest()
                records=json.loads((work/'executions/index.json').read_bytes())
                assert any(c['weights_sha256']==checksum for r in records if r['phase']=='publication'
                           for c in r['candidates'])
            outcomes[arm]={'score':measured,'calls':calls,'adopted':scores['adopted'],
                           'after_maintenance':after,'baseline':parent,'baseline_after_maintenance':parent_after,
                           'issued':len(ids),'scored':int(use.sum()),'after_maintenance_scored':int((use&future).sum()),
                           'path':work.relative_to(ROOT).as_posix()}
        if task_files:assert len(task_files)==2 and task_files[0]==task_files[1]
        np.testing.assert_array_equal(baselines[0],baselines[1])
        rows.append({'task':site,'scope':'伦敦2026迁移期',**outcomes,
                     'verdict':verdict(outcomes['project']['score'],outcomes['comind']['score'])})
    summary={'paper':'CoMind ICLR2026作者完整框架适配','rows':rows,
             'official':{v:sum(r['verdict']==v for r in rows[:3]) for v in ['胜','平','负']},
             'transfer':{v:sum(r['verdict']==v for r in rows[3:]) for v in ['胜','平','负']},
             'model_calls':{arm:sum(r[arm]['calls'] for r in rows) for arm in ['project','comind']},
             'prediction_and_selection_verified':True}
    atomic_json(folder/'comparison.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--bundle',type=Path,required=True)
    args=parser.parse_args();summarize(args.folder.resolve(),args.bundle.resolve())
