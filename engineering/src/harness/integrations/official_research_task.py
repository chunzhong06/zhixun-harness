"""只从官方三类原包和真实交付提取研究任务；搜索看不到确认及服务答案。"""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

from engineering.cli.workflow import public_maintenance as public
from harness.integrations.published_parent import bind,augment,read
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json

ROOT=Path(__file__).resolve().parents[4]
PARENTS=ROOT/'results/experiments/local-defaults/whole-maintenance/gas-feedback-coverage/integrated'


def metric(task,labels,prediction,groups):
    y=np.asarray(labels);p=np.asarray(prediction);groups=np.asarray(groups)
    if not len(y) or p.shape!=y.shape or not np.isfinite(p).all():raise ValueError('无效的实际预测')
    if task=='gas':
        if np.any(p!=p.astype(int)) or np.any((p<1)|(p>6)):raise ValueError('气体类别必须为1至6')
        matrix=np.zeros((6,6));np.add.at(matrix,(y.astype(int)-1,p.astype(int)-1),1)
        denominator=matrix.sum(0)+matrix.sum(1)
        f1=np.divide(2*matrix.diagonal(),denominator,out=np.zeros(6),where=denominator>0)
        recall=np.divide(matrix.diagonal(),matrix.sum(1),out=np.zeros(6),where=matrix.sum(1)>0)
        # 官方Macro-F1按答案与预测出现过的类别并集计分；完整服务恰好覆盖六类。
        return {'score':float(f1[denominator>0].mean()),'worst':float(recall[matrix.sum(1)>0].min()),'confusion':matrix.astype(int).tolist()}
    errors=(p-y)**2
    by_group={str(g):float(np.sqrt(errors[groups==g].mean())) for g in np.unique(groups)}
    return {'score':float(np.sqrt(errors.mean())),'worst':max(by_group.values()),'by_group':by_group,
            'signed_bias':float(np.mean(p-y))}


def prepare(task,output,*,stage=5,data_root=ROOT/'data/official'):
    """维护在第stage+1轮开始，之前发出的预测保持；其余服务输入只归外层持有。"""
    if task not in ('gas','bearing','engine') or not 0<=stage<10:raise ValueError('只接受官方三任务和已有十轮')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'task-identity.json').exists():raise ValueError('研究任务不可覆盖')
    run=PARENTS/task;report=read(run/'run.json')
    state_path=run/('regular/state.json' if task!='bearing' else 'state.json')
    if not state_path.exists():
        run=ROOT/'results/experiments/local-defaults/fixed-tree-references/setting-reference/automatic/integrated/bearing'
        report=read(run/'run.json');state_path=run/'state.json'
    state=read(state_path);samples=state['samples']
    if task=='gas':
        from harness.data.official_gas import read_official_gas
        filename='cpu-a-guide-20261110-v1.zip';checksum=public.PACKAGE_SHA;reader=read_official_gas
    else:filename,checksum,reader=public.PACKAGES[task]
    archive=Path(data_root)/task/filename
    if sha256(archive.read_bytes()).hexdigest()!=checksum:raise ValueError('官方原包身份改变')
    package=reader(archive)
    # 原包训练集是唯一可拟合标签来源，探索/确认设备和服务标签不进入任务文件。
    train=list(package.train)
    fit_ids=[r.sample_id for r in train];fit_x=np.asarray([r.values for r in train]);fit_y=np.asarray([r.label for r in train])
    explore=sorted((r for r in samples.values() if r['role']=='explore' and r['available_at']==stage),key=lambda r:r['sample_id'])
    confirm=sorted((r for r in samples.values() if r['role']=='adopt' and r['available_at']==stage),key=lambda r:r['sample_id'])
    reference_ordinal=stage+1;feedback_receipt=None
    if task=='gas':
        if stage<1:raise ValueError('气体首轮服务前没有现场回标可用于这项维护')
        # 第stage轮已交付并收到查询结果，第stage+1轮尚未开始。
        # 用查询前的父版评价这批新反馈，避免确认标签已被父版学习。
        feedback_receipt=report['accepted'][stage-1]
        returned=set(feedback_receipt['returned']);reference_ordinal=stage
        if len(returned)>report['budget']:raise ValueError('真实气体回标超过原预算')
        rows=[samples[k] for k in returned]
        if any(r['available_at']!=stage-1 or r['role'] not in ('explore','adopt') for r in rows):raise ValueError('气体回标时点或用途不符')
        explore=sorted((r for r in rows if r['role']=='explore'),key=lambda r:r['sample_id'])
        confirm=sorted((r for r in rows if r['role']=='adopt'),key=lambda r:r['sample_id'])
    serve=sorted((r for r in samples.values() if r['role']=='serve'),key=lambda r:(r['available_at'],r['sample_id']))
    if not explore or not confirm or any(r['label'] is not None for r in serve):raise ValueError('官方反馈角色不完整或服务泄漏标签')
    def device(row):return row.get('unit_id') or row['group']
    # 发动机group是FD子集，unit_id才是独立设备；不能把同子集误判为同设备。
    if task!='gas' and set(map(device,explore))&set(map(device,confirm)):raise ValueError('寿命探索与确认设备重叠')
    entries=report['accepted'] if task!='bearing' else report['rounds']
    baseline={};bindings=[];service_features=[]
    def raw(rows):return np.asarray([r['values'] for r in rows])
    with TemporaryDirectory(dir=output,prefix='temporary-') as temporary:
        temporary=Path(temporary)
        def bearing_augment(ordinal,x):
            event=entries[ordinal-1];header,blob=ModelArtifacts(run/'models').read(event['model_artifact'])
            weight=temporary/'parent.npz';weight.write_bytes(blob)
            with ZipFile(archive) as z,_trainer(z,temporary,'B') as trainer:
                model=trainer.Model.load(weight);prediction=np.asarray([float(model.predict(row[None,:])[0]) for row in x])
            return np.column_stack((x,prediction)),{'model_sha256':header['sha256'],'artifact':event['model_artifact']}
        def transformed(ordinal,x):
            if task=='bearing':return bearing_augment(ordinal,x)
            binding=bind(run,task,ordinal);return augment(binding,x,data_root),binding
        train_x,binding=transformed(reference_ordinal,fit_x)
        valid_x,_=transformed(reference_ordinal,raw(explore));confirm_x,_=transformed(reference_ordinal,raw(confirm))
        for ordinal,event in enumerate(entries,1):
            receipt=run/('accepted/'+event['request']+'.json') if task!='bearing' else run/event['submitted']
            payload=read(receipt)['payload'];rows=[r for r in serve if r['available_at']==ordinal-1]
            values,identity=transformed(ordinal,raw(rows));bindings.append(identity)
            expected=np.asarray([payload['predictions'][r['sample_id']] for r in rows])
            if not np.allclose(values[:,-1],expected,rtol=0,atol=1e-8):raise ValueError('官方父版未复现实际交付')
            for r,v in zip(rows,expected,strict=True):baseline[r['sample_id']]=float(v)
            service_features.extend(values)
    columns=[f'input_{i}' for i in range(fit_x.shape[1])]
    columns+=([f'parent_probability_{i}' for i in range(1,7)]+['parent_label'] if task=='gas' else ['parent_value'])
    folder=output/'task';folder.mkdir()
    frame=pd.DataFrame(train_x,columns=columns);frame['target']=fit_y;frame.to_csv(folder/'train.csv',index=False)
    pd.DataFrame(valid_x,columns=columns).to_csv(folder/'validation.csv',index=False)
    pd.DataFrame(confirm_x,columns=columns).to_csv(folder/'test.csv',index=False)
    atomic_json(folder/'task.json',{'task':task,'classification':task=='gas','metric':'macro_f1' if task=='gas' else 'rmse',
        'feature_columns':columns,'train_groups':[r.unit_id if getattr(r,'unit_id',None) else r.group for r in train]})
    service_ids=[r['sample_id'] for r in serve];future=np.asarray([r['available_at']>=stage for r in serve]);service_x=np.asarray(service_features)
    oracle={'task':task,'stage':stage,'train_ids':fit_ids,'explore_ids':[r['sample_id'] for r in explore],
            'confirm_ids':[r['sample_id'] for r in confirm],'service_ids':service_ids,
            'explore_y':[r['label'] for r in explore],'confirm_y':[r['label'] for r in confirm],
            'explore_groups':[device(r) for r in explore],'confirm_groups':[device(r) for r in confirm],
            'service_groups':[device(r) for r in serve]}
    atomic_json(output/'oracle.json',oracle)
    np.savez_compressed(output/'outer-inputs.npz',service_x=service_x,future=future,baseline=np.asarray([baseline[k] for k in service_ids]))
    identity={'task':task,'stage':stage,'reference_parent_ordinal':reference_ordinal,'feedback_receipt':feedback_receipt,
        'archive':str(archive),'archive_sha256':checksum,'parent_run':str(run),
        'parent_run_sha256':sha256((run/'run.json').read_bytes()).hexdigest(),'state_sha256':sha256(state_path.read_bytes()).hexdigest(),
        'parent_bindings':bindings,'train_rows':len(train),'explore_rows':len(explore),'confirm_rows':len(confirm),'service_rows':len(serve),
        'task_files':{p.name:sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()},'future_inputs_in_task':False}
    atomic_json(output/'task-identity.json',identity)
    return identity
