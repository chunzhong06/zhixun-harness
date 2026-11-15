"""恢复已完成作者决策链的最后源码执行；不重新生成、改选或重放前面研究。"""
from argparse import ArgumentParser
import ast
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.workflow.mlestar_author import AUTHOR,finish_delivery,verify_source
from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.workflow.london_complete import masks
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.author_mlestar import ACTOR
from harness.integrations.recipe_regressor import code_inputs
from harness.models.stream_plan import StreamPlanBank
from harness.runtime.storage import atomic_json


def read(path):return json.loads(Path(path).read_bytes())
def tree(source):return ast.dump(ast.parse(source),include_attributes=False)


def main():
    parser=ArgumentParser();parser.add_argument('--interrupted',required=True,type=Path);parser.add_argument('--archive',required=True,type=Path)
    parser.add_argument('--library',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();old=args.interrupted.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'recovery.json').exists():raise ValueError('不覆盖末段恢复')
    verify_source();meta=read(old/'task-identity.json');config=read(args.library)
    if meta['archive_sha256']!=sha256(args.archive.read_bytes()).hexdigest() or meta['library_sha256']!=sha256(args.library.read_bytes()).hexdigest():raise ValueError('恢复的数据或方案库身份改变')
    args.site=meta['site'];args.start=meta['start'];scene.configure(args.start)
    features,names=code_inputs(scene.prepare(args.archive,True,True));truth=scene.readings(args.archive,args.site).to_numpy()
    prior,initial_fit,initial_windows,service,returned=scene.roles(truth);start=int(service[0]);at=start+28*24
    labels=np.r_[prior[:start],returned];fit,windows,history=masks(labels,at)
    if not np.array_equal(np.flatnonzero(fit),meta['train_ids']):raise ValueError('试训标签行改变')
    initial_history=np.arange(start-21*24-6,start);bank=StreamPlanBank(output/'unused',features,names)
    baseline=bank.issue(config['initial_plan'],read(old/'initial/training.json'),initial_history,service,labels,np.maximum(0.,features[service,4]))
    with np.load(old/'baseline-issued.npz',allow_pickle=False) as z:
        if not np.array_equal(z['ids'],service) or not np.allclose(z['prediction'],baseline,rtol=0,atol=1e-8):raise ValueError('恢复改变了原已发预测')
    np.savez_compressed(output/'baseline-issued.npz',ids=service,prediction=baseline)
    validation=history[history<at-7*24];confirmation=history[history>=at-7*24]
    if not np.array_equal(validation,meta['validation_ids']) or not np.array_equal(confirmation,meta['confirmation_ids']):raise ValueError('恢复改变了探索或确认时间')
    oracle={'validation_y':labels[validation],'confirmation_y':labels[confirmation],'validation_hours':validation,
            'confirmation_hours':confirmation,'exploration_masks':[w[validation] for w in windows[:2]]}
    task_name='London_'+args.site;task=output/'tasks'/task_name;task.mkdir(parents=True)
    for name,checksum in meta['task_files'].items():
        p=old/'tasks'/task_name/name
        if sha256(p.read_bytes()).hexdigest()!=checksum:raise ValueError('原任务文件改变：'+name)
        shutil.copy2(p,task/name)
    for name,ids in [('train.csv',np.flatnonzero(fit)),('validation.csv',validation),('test.csv',confirmation)]:
        data=pd.read_csv(task/name)
        if name=='train.csv' and not np.allclose(data.pop('target').to_numpy(),labels[ids],rtol=0,atol=1e-10):raise ValueError('训练标签不同')
        if list(data.columns)!=names or not np.allclose(data.to_numpy(),features[ids],rtol=0,atol=1e-10):raise ValueError('原任务输入不同')
    queue=old/'queue';requests=[]
    for p in sorted((queue/'requests/mle-model').glob('*.json')):
        q=read(p);response=queue/'responses/mle-model'/p.name
        if q['context']['actor']=='submission_agent' and response.is_file():requests.append((p,q,read(response)))
    if not requests:raise ValueError('原作者没有提供最终提交源码')
    request_path,request,response=requests[-1]
    if response['request_id']!=request['request_id']:raise ValueError('原最终响应身份不同')
    raw=response['plan']['text'];source=raw.replace('```python','').replace('```','')
    if 'submit_model' not in source:raise ValueError('原最终源码没有交付模型的接口')
    instruction=request['context']['payload']['request']['config']['system_instruction']
    parent=re.search(r'# Python solution\s+```python\n(.*?)\n```',instruction,re.S)
    records=read(old/'executions/index.json')
    if not parent:raise ValueError('无法核对作者实际选中的方案')
    matching=[r for r in records if r['returncode']==0 and tree(Path(r['source']).read_text(encoding='utf-8'))==tree(parent.group(1))]
    if not matching:raise ValueError('作者选中的源码没有原始真实执行依据')
    adapted=source+'\n# ./final/submission.csv 由原 submit_model 调用生成。\n'
    if tree(source)!=tree(adapted):raise ValueError('最终源码执行语义改变')
    spec=importlib.util.spec_from_file_location('author_submission_predicate',AUTHOR/'machine_learning_engineering/shared_libraries/code_util.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not module.get_run_code_condition('submission_agent',adapted):raise ValueError('仍不符合原作者提交检查')
    provenance=read(old/'driver-provenance.json');replays=read(queue/'replays.json')
    model_responses=len(list((queue/'responses/mle-model').glob('*.json')))
    broker=SimpleNamespace(calls=model_responses,search_calls=len(list((queue/'responses/mle-search').glob('*.json'))),
        reused_model_calls=sum(r['kind']=='model' for r in replays),reused_search_calls=sum(r['kind']=='search' for r in replays),
        prior_model_invocations=provenance['prior_model_invocations'])
    recovery={'original_run':str(old),'original_final_request':str(request_path),'original_response_sha256':sha256((queue/'responses/mle-model'/request_path.name).read_bytes()).hexdigest(),
        'source_sha256':sha256(source.encode()).hexdigest(),'adapted_source_sha256':sha256(adapted.encode()).hexdigest(),
        'python_ast_unchanged':True,'original_selected_executions':[r['sequence'] for r in matching],
        'prior_actual_model_invocations':broker.prior_model_invocations+broker.calls-broker.reused_model_calls,
        'new_model_invocations':0,'reason':'原最终源码调用submit_model，作者词面提交检查未识别；增加非执行注释并执行原实际交付接口'}
    atomic_json(output/'recovery.json',recovery);atomic_json(output/'task-identity.json',meta)
    (output/'original-submission.py').write_text(source,encoding='utf-8');(output/'adapted-submission.py').write_text(adapted,encoding='utf-8')
    workspace=output/'author-workspace';work=workspace/task_name/'ensemble';shutil.copytree(task,work/'input')
    executor=AuthorExecutor(output/'executions',workspace,oracle);ACTOR.set('submission_agent_recovery')
    result=executor(adapted,work,'final_solution.py',90)
    if result['returncode'] or not (Path(executor.records[-1]['directory'])/'final/submission.csv').is_file():raise ValueError('原最终程序仍未实际交付模型与文件')
    finish_delivery(args,output,features,names,labels,history,service,windows,at,baseline,task_name,task,executor,workspace,broker,executor.records[-1],truth,
        {'scope':'已见开发资料上的完整作者决策链及最终交付恢复','completion_mode':'原最终源码AST不变的提交接口恢复','recovery_from':str(old),'new_model_invocations':0})


if __name__=='__main__':main()
