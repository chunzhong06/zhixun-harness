"""官方三任务共用的真实部件研究入口：搜索封存后才确认和交付。"""
from argparse import ArgumentParser
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from harness.integrations.author_mlestar import ACTOR,AuthorBroker
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.official_research_task import metric
from harness.integrations.official_component_driver import search,POLICIES
from harness.runtime.storage import atomic_json
from engineering.cli.study.prepare_official_research import SOURCE as PARENT


def fixed_sources(task):
    prefix='from harness_task import load_data,evaluate_model\nX,y,V,T=load_data()\n'
    if task=='gas':
        models=[('标准化线性分类','from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\nmodel=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=200))'),
                ('原输入非线性分类','from sklearn.ensemble import RandomForestClassifier\nmodel=RandomForestClassifier(n_estimators=32,max_depth=8,random_state=42,n_jobs=1)')]
    else:
        models=[('线性全特征','from sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import Ridge\nmodel=make_pipeline(StandardScaler(),Ridge(alpha=10))'),
                ('非线性全特征','from sklearn.ensemble import RandomForestRegressor\nmodel=RandomForestRegressor(n_estimators=24,max_depth=6,min_samples_leaf=8,random_state=42,n_jobs=1)')]
    sources={'保持完整父版':PARENT}
    for name,body in models:sources[name]=prefix+body+'\nmodel.fit(X,y)\nevaluate_model(model,label='+repr(name)+')\n'
    if task!='gas':
        sources['源设备等距锚点中位数']=prefix+'''import json
from pathlib import Path
import numpy as np
from sklearn.dummy import DummyRegressor
groups=np.asarray(json.loads(Path('input/task.json').read_text(encoding='utf-8'))['train_groups'])
ids=[]
for g in np.unique(groups):
    rows=np.flatnonzero(groups==g)
    ids.extend(rows[np.linspace(0,len(rows)-1,min(8,len(rows))).astype(int)])
model=DummyRegressor(strategy='median').fit(X.iloc[ids],y[ids])
evaluate_model(model,label='源设备等距锚点中位数')
'''
    return sources


def read(path):return json.loads(Path(path).read_bytes())


def accepted(task,old,new):
    if task=='gas':return new['score']>old['score']+1e-12
    return new['score']<old['score'] and (task!='engine' or new['worst']<=1.01*old['worst'])


async def run(args):
    task=args.task;source=args.bundle.resolve()/task;out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'input-identity.json').exists():raise ValueError('不覆盖真实部件比较')
    identity=read(source/'task-identity.json');oracle=read(source/'oracle.json')
    for name,checksum in identity['task_files'].items():
        if sha256((source/'task'/name).read_bytes()).hexdigest()!=checksum:raise ValueError('冻结合法任务文件改变')
    atomic_json(out/'input-identity.json',{'bundle':str(source),'task_identity_sha256':sha256((source/'task-identity.json').read_bytes()).hexdigest(),
        'protocol':'同合法任务、固定源程序、每组四次Luna调用；第六轮确认后交付','controllers':list(POLICIES)})
    def measured(values):
        result=metric(task,oracle['explore_y'],values,oracle['explore_groups'])
        return {**result,'loss':1-result['score'] if task=='gas' else result['score']}
    basic_oracle={'validation_y':np.asarray(oracle['explore_y']),'confirmation_y':np.asarray(oracle['confirm_y'])}
    description={'task':task,'unit':'类别1至6' if task=='gas' else ('分钟' if task=='bearing' else '剩余循环'),
        'maintenance_round':identity['stage']+1,'training_rows':identity['train_rows'],'explore_rows':identity['explore_rows'],
        'columns':read(source/'task/task.json')['feature_columns'],'training':'全部官方源训练记录；不能用探索或确认输入伪标签回训',
        'parent_columns':'末列是完整父版预测；气体此前六列是按1至6排列的父版概率',
        'scope':'只基于本请求资料改进今后交付。可替换学习器、特征、损失和训练选择。前五轮不能改变。',
        'score':'官方Macro-F1越高越好' if task=='gas' else 'RMSE越低越好；保留既有保护条件'}
    states={}
    # 两个控制器在独立目录执行；相同预置工具分别计入本组实际成本。
    for arm in POLICIES:
        folder=out/arm;workspace=folder/'workspace';work=workspace/'task';shutil.copytree(source/'task',work/'input')
        executor=AuthorExecutor(folder/'executions',workspace,basic_oracle,scorer=measured)
        previous=args.resume_from/arm if args.resume_from else None
        if previous:
            old_identity=read(args.resume_from/'input-identity.json')
            if old_identity['task_identity_sha256']!=sha256((source/'task-identity.json').read_bytes()).hexdigest():raise ValueError('恢复时合法任务改变')
            records=read(previous/'executions/index.json')
            shared_count=len(fixed_sources(task))
            if len(records)<shared_count or any(r['actor']!=arm+'_shared_components' for r in records[:shared_count]):raise ValueError('原共享工具执行不完整')
            for record in records:
                for name,checksum in identity['task_files'].items():
                    if sha256((Path(record['directory'])/'input'/name).read_bytes()).hexdigest()!=checksum:raise ValueError('原已执行训练输入改变')
                if sha256(Path(record['source']).read_text(encoding='utf-8').encode()).hexdigest()!=record['source_sha256']:raise ValueError('原执行源码改变')
                for candidate in record['candidates']:
                    if sha256(Path(candidate['path']).with_suffix('.pkl').read_bytes()).hexdigest()!=candidate['weights_sha256']:raise ValueError('原实际权重改变')
            executor.records=records[:shared_count];executor.pending_replay=records[shared_count:]
            atomic_json(folder/'executions/index.json',executor.records)
        else:
            ACTOR.set(arm+'_shared_components')
            for name,code in fixed_sources(task).items():executor(code,work,'shared.py',90)
        recorded=len(list((previous/'queue/responses/mle-model').glob('*.json'))) if previous else 0
        provenance=read(previous/'driver-provenance.json') if previous and (previous/'driver-provenance.json').exists() else {}
        extra=provenance.get('extra_model_invocations',0)
        broker=AuthorBroker(folder/'queue',max_calls=args.calls,timeout=200,replay_from=previous,prior_model_invocations=recorded+extra)
        broker.recorded_calls=recorded
        states[arm]=(folder,workspace,work,executor,broker)
    async def arm_run(arm):
        folder,workspace,work,executor,broker=states[arm]
        selected=await search(arm,description,work,executor,broker,args.calls,broker.recorded_calls)
        candidate=selected['candidates'][-1]
        with np.load(Path(candidate['path']).with_suffix('.npz'),allow_pickle=False) as z:
            checks=[metric(task,oracle['explore_y'],z['validation'],oracle['explore_groups']),metric(task,oracle['confirm_y'],z['confirmation'],oracle['confirm_groups'])]
        parent_checks=[metric(task,oracle['explore_y'],pd.read_csv(source/'task/validation.csv').iloc[:,-1].to_numpy(),oracle['explore_groups']),
                       metric(task,oracle['confirm_y'],pd.read_csv(source/'task/test.csv').iloc[:,-1].to_numpy(),oracle['confirm_groups'])]
        adopted=all(accepted(task,a,b) for a,b in zip(parent_checks,checks,strict=True))
        decision={'selected_execution':selected['sequence'],'candidate':candidate,'adopted':adopted,'parent_metrics':parent_checks,'candidate_metrics':checks,
                  'model_calls':broker.prior_model_invocations+broker.calls-broker.reused_model_calls,
                  'reused_calls':broker.reused_model_calls,'source_sha256':selected['source_sha256']}
        atomic_json(folder/'decision.json',decision)
        return decision
    decisions=dict(zip(POLICIES,await asyncio.gather(*(arm_run(a) for a in POLICIES)),strict=True))
    # 两组决定都已冻结，才将未来无答案输入交给权重重载进程；控制器不再接收任何结果。
    with np.load(source/'outer-inputs.npz',allow_pickle=False) as z:future=z['future'];baseline=z['baseline'];features=z['service_x'][future]
    serving=out/'serving';(serving/'models').mkdir(parents=True)
    pd.DataFrame(features,columns=description['columns']).to_csv(serving/'service.csv',index=False)
    cases={};owners={}
    for arm,(_,_,_,executor,_) in states.items():
        for execution in executor.records:
            if execution['returncode'] or not execution['candidates']:continue
            candidate=execution['candidates'][-1];p=Path(candidate['path']).with_suffix('.pkl');key=arm+'-'+str(execution['sequence'])
            if sha256(p.read_bytes()).hexdigest()!=candidate['weights_sha256']:raise ValueError('发布前权重发生变化')
            shutil.copy2(p,serving/'models'/(key+'.pkl'));cases[key]=candidate['weights_sha256'];owners[key]=(arm,execution,candidate)
    atomic_json(serving/'cases.json',cases)
    script=serving/'predict.py';script.write_text('''from pathlib import Path
import json
import cloudpickle
import numpy as np
import pandas as pd
rows=pd.read_csv('service.csv')
for key in json.loads(Path('cases.json').read_text(encoding='utf-8')):
    model=cloudpickle.loads((Path('models')/(key+'.pkl')).read_bytes())
    values=[]
    for i in range(len(rows)):
        p=np.asarray(model.predict(rows.iloc[[i]]),dtype=float)
        if p.shape!=(1,) or not np.isfinite(p).all():raise ValueError('实际单行预测无效')
        values.append(max(0.,float(p[0])))
    np.save(key+'.npy',np.asarray(values))
''',encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/study/execute_author_program.py'),'--script',str(script)],
        cwd=serving,env=env,capture_output=True,text=True,encoding='utf-8',timeout=240)
    (serving/'stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise ValueError('实际发布预测失败：'+result.stderr[-1000:])
    predictions={'baseline':baseline};fixed=[]
    for key,(arm,execution,candidate) in owners.items():
        values=baseline.copy();values[future]=np.load(serving/(key+'.npy'),allow_pickle=False)
        predictions['fixed-'+key]=values;fixed.append('fixed-'+key)
        with np.load(Path(candidate['path']).with_suffix('.npz'),allow_pickle=False) as z:
            checks=[metric(task,oracle['explore_y'],z['validation'],oracle['explore_groups']),metric(task,oracle['confirm_y'],z['confirmation'],oracle['confirm_groups'])]
        gate=all(accepted(task,a,b) for a,b in zip(decisions[arm]['parent_metrics'],checks,strict=True))
        predictions['gated-'+key]=values if gate else baseline.copy();fixed.append('gated-'+key)
    for arm,d in decisions.items():
        key=arm+'-'+str(d['selected_execution']);predictions[arm]=predictions['fixed-'+key] if d['adopted'] else baseline.copy()
        # 实际交付入口保留模型、输入契约、每条预测和采用决定；原父版仍可作为未采用分支恢复。
        atomic_json(out/arm/'delivery.json',{'decision_sha256':sha256((out/arm/'decision.json').read_bytes()).hexdigest(),
            'model_key':key,'used_new_model':d['adopted'],'input_identity':str(out/'input-identity.json'),
            'predictions':dict(zip(oracle['service_ids'],map(float,predictions[arm]),strict=True))})
    np.savez_compressed(out/'issued.npz',ids=np.asarray(oracle['service_ids']),**predictions)
    atomic_json(out/'predictions-complete.json',{'done':True,'selection_closed_before_service':True})
    from engineering.cli.workflow import public_maintenance as public
    package=public.read_official_gas(Path(identity['archive'])) if task=='gas' else public.PACKAGES[task][2](Path(identity['archive']))
    answers={r.sample_id:r.label for r in package.validation};y=[answers[k] for k in oracle['service_ids']]
    scores={k:metric(task,y,v,oracle['service_groups']) for k,v in predictions.items()}
    best=(max if task=='gas' else min)(['baseline',*fixed],key=lambda k:scores[k]['score'])
    final={'task':task,'scope':'官方已见开发资料，第六轮维护后完成原十轮实际交付','scores':scores,'best_fixed':best,
        'arms':{a:{'model_calls':d['model_calls'],'adopted':d['adopted'],'execution':d['selected_execution']} for a,d in decisions.items()},
        'shared_tools':'原父版、预置全特征学习器、源端锚点常量及两组实际新源码；逐条发放','independent_acquisition':False}
    atomic_json(out/'scores.json',final)
    print(json.dumps({task:{'baseline':scores['baseline']['score'],'mle_star':scores['mle_star']['score'],'iris':scores['iris']['score'],
        'best_fixed':scores[best]['score'],'model_calls':sum(d['model_calls'] for d in decisions.values())}},ensure_ascii=False))


def main(argv=None):
    parser=ArgumentParser();parser.add_argument('--task',choices=('gas','bearing','engine'),required=True)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--calls',type=int,default=4)
    parser.add_argument('--resume-from',type=Path,help='按严格相同的原观察恢复真实提案与已完成执行，保留原失败及调用费用')
    args=parser.parse_args(argv)
    asyncio.run(run(args))


if __name__=='__main__':main()
