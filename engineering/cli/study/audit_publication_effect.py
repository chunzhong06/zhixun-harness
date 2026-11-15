"""固定原选案与反馈时序，重放补训前后权重，诊断发布过程的实际影响。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.workflow import london_field_driver as london
from harness.models.sparse_reference import correct,setting_of
from harness.integrations.candidate_selection import choose_candidate, match_publication
from harness.runtime.storage import atomic_json


def selected_models(work,arm):
    """按原决策绑定候选；不按本次诊断得到的未来分数重新选案。"""
    decision=json.loads((work/'decision.json').read_bytes())
    records=json.loads((work/'executions/index.json').read_bytes())
    selected=next(r for r in records if r['sequence']==decision['selected_execution'])
    if arm=='comind':
        before=json.loads((work/'author-workspace/author-selection.json').read_bytes())['candidate']
    else:
        before=choose_candidate(selected,decision.get('candidate_id'))
    assert before['weights_sha256']==decision['weights_sha256']
    publication=next(r for r in reversed(records) if r['phase']=='publication')
    after=match_publication(before,publication)
    assert sha256((work/'serving/input/model.pkl').read_bytes()).hexdigest()==after['weights_sha256']
    assert before['feedback']==after['feedback'],'该项包含反馈规则变化，不能只归因于补训权重'
    for candidate in [before,after]:
        path=Path(candidate['path']).resolve()
        assert path.is_relative_to(ROOT)
        assert sha256(path.with_suffix('.pkl').read_bytes()).hexdigest()==candidate['weights_sha256']
    return decision,before,after


def cases(baseline):
    for site in ['BX2','CE2']:
        folder=baseline/'london-2026'/site
        alias=folder/'accepted-runs.json'
        names=json.loads(alias.read_bytes()) if alias.exists() else {}
        for arm in ['project','comind']:
            work=folder/names.get(arm,arm)
            decision,before,after=selected_models(work,arm)
            assert decision['adopted'],'本次四个预先列出的案例均采用了新模型'
            yield site,arm,work,before,after


def input_files(baseline):
    """供运行前检查点直接收集真正使用的权重、预测与原选择依据。"""
    paths={baseline/'comparison.json'}
    for site,arm,work,before,after in cases(baseline):
        paths.update(work/name for name in ['decision.json','scores.json','issued.npz','oracle.npz',
            'executions/index.json','serving/input/service.csv','serving/input/model.pkl','serving/prediction.npy'])
        if arm=='comind':paths.add(work/'author-workspace/author-selection.json')
        alias=baseline/'london-2026'/site/'accepted-runs.json'
        if alias.exists():paths.add(alias)
        for candidate in [before,after]:
            p=Path(candidate['path']);paths.update([p,p.with_suffix('.pkl'),p.with_suffix('.npz')])
    return sorted(paths)


def replay(weights,service,destination,*,timeout=90):
    destination.mkdir(parents=True)
    inputs=destination/'input';inputs.mkdir()
    shutil.copy2(weights,inputs/'model.pkl');shutil.copy2(service,inputs/'service.csv')
    script=destination/'predict.py'
    script.write_text('''from pathlib import Path
import cloudpickle
import pandas as pd
import numpy as np
model=cloudpickle.loads(Path('input/model.pkl').read_bytes())
rows=pd.read_csv('input/service.csv')
values=[]
for i in range(len(rows)):
    p=np.asarray(model.predict(rows.iloc[[i]]),dtype=float)
    if p.shape!=(1,) or not np.isfinite(p).all():raise ValueError('单条服务预测无效')
    values.append(max(0.,float(p[0])))
np.save('prediction.npy',np.asarray(values))
''',encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/study/execute_author_program.py'),
        '--script',str(script)],cwd=destination,env=env,capture_output=True,text=True,encoding='utf-8',timeout=timeout)
    (destination/'stdout.log').write_text(result.stdout,encoding='utf-8')
    (destination/'stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('原冻结权重重放失败：'+result.stderr[-1000:])
    return np.load(destination/'prediction.npy',allow_pickle=False)


def run(args):
    baseline=args.baseline.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('诊断输出必须是新目录')
    output.mkdir(parents=True);london.configure('2026-07-01');results={}
    for site,arm,work,before,after in cases(baseline):
        destination=output/site/arm
        with np.load(work/'issued.npz',allow_pickle=False) as z:
            ids=z['ids'];parent=z['baseline'];returned=z['returned'];original=z[arm]
        with np.load(work/'oracle.npz',allow_pickle=False) as z:
            past_hours=np.r_[z['validation_hours'],z['confirmation_hours']]
            past_reference=np.r_[z['validation_y'],z['confirmation_y']]
        future=np.arange(len(ids))>=28*24
        clock=np.r_[past_hours,ids[future]]
        reference=np.r_[past_reference,returned[future]]
        issued={};fingerprints={}
        for name,candidate in [('accepted_weights',before),('refitted_weights',after)]:
            source=Path(candidate['path'])
            prediction=replay(source.with_suffix('.pkl'),work/'serving/input/service.csv',destination/name)
            with np.load(source.with_suffix('.npz'),allow_pickle=False) as z:
                past=np.r_[z['validation'],z['confirmation']]
            assert len(prediction)==int(future.sum()) and len(past)==len(past_hours)
            values=correct(np.r_[past,prediction][:,None],reference,clock,setting_of(candidate['feedback']))[:,0]
            issued[name]=np.r_[parent[~future],values[len(past):]]
            fingerprints[name]=candidate['weights_sha256']
            if name=='refitted_weights':
                np.testing.assert_allclose(prediction,np.load(work/'serving/prediction.npy'),rtol=0,atol=1e-9)
        # 阳性对照必须复现原真实交付，才解释补训前权重的反事实。
        np.testing.assert_allclose(issued['refitted_weights'],original,rtol=0,atol=1e-9)
        np.savez_compressed(destination/'issued.npz',ids=ids,baseline=parent,**issued)
        truth=london.readings(args.archive,site).to_numpy()[ids];valid=np.isfinite(truth)
        metrics={}
        for name,values in [('baseline',parent),*issued.items()]:
            metrics[name]={'complete_rmse':float(np.sqrt(np.mean((values[valid]-truth[valid])**2))),
                'after_maintenance_rmse':float(np.sqrt(np.mean((values[valid&future]-truth[valid&future])**2)))}
        result={'metrics':metrics,'weights_sha256':fingerprints,'issued':len(ids),
            'after_maintenance_rows':int((valid&future).sum()),'original_delivery_reproduced':True,
            'refit_change_after_maintenance':metrics['refitted_weights']['after_maintenance_rmse']-metrics['accepted_weights']['after_maintenance_rmse']}
        atomic_json(destination/'scores.json',result);results[site+'/'+arm]=result
    report={'scope':'已见结果上的补训因果诊断，不作为新设计独立验证','original_selection_frozen':True,
            'new_training':0,'new_model_calls':0,'results':results}
    atomic_json(output/'scores.json',report);print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,default=ROOT/'data/public/london-air-2026/stations.zip')
    run(p.parse_args())
