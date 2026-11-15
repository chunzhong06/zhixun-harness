"""固定上下文成本适配与跨任务原配置验证；不按服务评分修改参数。"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
import numpy as np
import pandas as pd
from harness.runtime.storage import atomic_json
from harness.integrations.broad_transfer_task import metric as broad_metric,accepted as broad_accepted
from harness.integrations.official_research_task import metric as official_metric
from engineering.cli.workflow.official_components import accepted as official_accepted


def run_case(task,args):
    is_official=task=='bearing';classification=task=='sgp40'
    source=args.base/('frontier-2026-inputs' if is_official else 'broad-transfer-inputs')/task
    output=args.output/task;output.mkdir()
    identity=json.loads((source/'task-identity.json').read_bytes())
    for name,checksum in identity['task_files'].items():
        if sha256((source/'task'/name).read_bytes()).hexdigest()!=checksum:raise ValueError('冻结任务改变')
    meta=json.loads((source/'task/task.json').read_bytes())
    train=pd.read_csv(source/'task/train.csv');target=train.pop('target').to_numpy();groups=np.asarray(meta['train_groups'])
    if train.columns[-1]!='parent_value':raise ValueError('须明确原始特征和父预测列')
    columns=list(train.columns[:-1]);variants={'full_source':np.arange(len(train))}
    if is_official:
        balanced=np.sort(np.concatenate([rows[np.linspace(0,len(rows)-1,min(128,len(rows))).astype(int)]
            for g in np.unique(groups) for rows in [np.flatnonzero(groups==g)]]))
        variants={'balanced_context':balanced,'uniform_same_size':np.linspace(0,len(train)-1,len(balanced)).astype(int)}
    frames={phase:pd.read_csv(source/'task'/name) for phase,name in [('explore','validation.csv'),('confirm','test.csv')]}
    with np.load(source/'outer-inputs.npz',allow_pickle=False) as z:
        baseline,future=z['baseline'],z['future'];frames['future']=pd.DataFrame(z['service_x'][future],columns=meta['feature_columns'])
    oracle=json.loads((source/'oracle.json').read_bytes())
    metric=official_metric if is_official else broad_metric;accepted=official_accepted if is_official else broad_accepted
    parent={phase:metric(task,oracle[phase+'_y'],frames[phase].iloc[:,-1],oracle[phase+'_groups']) for phase in ['explore','confirm']}
    weights_folder=ROOT/'references/models/tabicl-v2';manifest=json.loads((weights_folder/'source-manifest.json').read_bytes())
    filename='tabicl-'+('classifier' if classification else 'regressor')+'-v2-20260212.ckpt'
    checksum=next(v['lfs']['oid'] for v in manifest['files'] if v['path']==filename)
    records,predictions={},{}
    for variant,rows in variants.items():
        folder=output/variant;folder.mkdir()
        fitting=train.iloc[rows][columns].copy();fitting['target']=target[rows];fitting.to_csv(folder/'train.csv',index=False)
        for phase,frame in frames.items():frame[columns].to_csv(folder/(phase+'.csv'),index=False)
        atomic_json(folder/'spec.json',{'classification':classification,'checkpoint':str(weights_folder/filename),'checkpoint_sha256':checksum,
                    'source_row_indices':rows.tolist(),'source_group_counts':{str(g):int(np.sum(groups[rows]==g)) for g in np.unique(groups)},
                    'no_target_labels_fit':True,'n_estimators':8})
        started=time.monotonic()
        try:
            proc=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/study/tabicl_candidate_worker.py'),'--folder',str(folder.resolve())],
                                cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=900)
            (folder/'stdout.log').write_text(proc.stdout,encoding='utf-8');(folder/'stderr.log').write_text(proc.stderr,encoding='utf-8')
            if proc.returncode:raise RuntimeError(proc.stderr[-2000:])
            record=json.loads((folder/'complete.json').read_bytes())
            with np.load(folder/'predictions.npz',allow_pickle=False) as z:pp={phase:z[phase] for phase in frames}
            check={phase:metric(task,oracle[phase+'_y'],pp[phase],oracle[phase+'_groups']) for phase in ['explore','confirm']}
            record.update(check,adopted=all(accepted(task,parent[p],check[p]) for p in check))
            direct=baseline.copy();direct[future]=pp['future'];predictions[variant+'/direct']=direct
            predictions[variant+'/actual']=direct if record['adopted'] else baseline.copy()
        except (subprocess.TimeoutExpired,RuntimeError) as exc:
            record={'success':False,'reason':str(exc),'elapsed_seconds':time.monotonic()-started,'adopted':False}
        records[variant]=record;atomic_json(folder/'decision.json',record)
    success=[k for k,v in records.items() if v['success']]
    selected=min(success,key=lambda k:(-1 if classification else 1)*records[k]['explore']['score']) if success else None
    atomic_json(output/'decisions-sealed.json',{'parent':parent,'variants':records,'selected_by_explore':selected})
    np.savez_compressed(output/'issued.npz',ids=np.asarray(oracle['service_ids']),**predictions)
    if is_official:
        from harness.data.official_bearing import read_official_bearing
        archive=Path(identity['archive'])
        if sha256(archive.read_bytes()).hexdigest()!=identity['archive_sha256']:raise ValueError('原官方包改变')
        answers={r.sample_id:r.label for r in read_official_bearing(archive).validation};truth=np.asarray([answers[k] for k in oracle['service_ids']])
    else:
        with np.load(source/'private-answers.npz',allow_pickle=False) as z:
            truth=z['y']
            if not np.array_equal(z['ids'],oracle['service_ids']):raise ValueError('服务答案ID改变')
    service_groups=np.asarray(oracle['service_groups'])
    scores={name:{'complete':metric(task,truth,values,service_groups),'future':metric(task,truth[future],values[future],service_groups[future])}
            for name,values in predictions.items()}
    for values in predictions.values():
        if not np.array_equal(values[~future],baseline[~future]):raise ValueError('旧交付改变')
    result={'task':task,'variants':records,'selected_by_explore':selected,'scores':scores,'baseline':metric(task,truth,baseline,service_groups),
            'issued_sha256':sha256((output/'issued.npz').read_bytes()).hexdigest(),'new_research_model_calls':0,
            'scope':'ICML2026原预测器默认8成员跨任务组件验证；轴承仅改变源上下文构成；非新完整研究驱动比较'}
    atomic_json(output/'scores.json',result);print(json.dumps(result,ensure_ascii=False),flush=True);return result


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    result={task:run_case(task,args) for task in ['bearing','ncmapss_ds06','sgp40']};atomic_json(args.output/'scores.json',result)
