"""完整执行固定作者 MLE-STAR，并将最终模型接入实际逐小时交付。"""
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
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from engineering.cli.workflow import london_field_driver as scene
from engineering.cli.workflow.london_complete import masks,maintenance_ready
from harness.integrations.author_mlestar import AuthorBroker,load_author_pipeline
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.candidate_selection import candidate_id, match_publication
from harness.integrations.recipe_regressor import code_inputs
from harness.models.sparse_reference import correct,setting_of
from harness.models.station_residual import error,gate
from harness.models.stream_plan import StreamPlanBank
from harness.runtime.storage import atomic_json

AUTHOR=ROOT/'references/implementations/google/adk-recipes/8a745552288f541f863bf4e50ca198522383d51d/machine-learning-engineering'


def verify_source():
    manifest=json.loads((AUTHOR/'source-manifest.json').read_bytes())
    for item in manifest['files']:
        if sha256((AUTHOR/item['local']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('固定作者源码改变：'+item['local'])
    return manifest['commit']


def frame(x,names,path,y=None):
    data=pd.DataFrame(x,columns=names)
    if y is not None:data['target']=y
    data.to_csv(path,index=False)


def task_description(names,baseline_source):
    return '''任务：维护已连续发放28日的伦敦PM2.5回归模型。仅使用input中的合法资料；原始未来服务输入和所有评分标签均不在任务目录中。
输入train.csv的target为绝对浓度，其余列为特征；validation.csv与test.csv只有相同特征。所有训练标签严格早于维护21日前，不能将探索或确认预测当作标签回训。使用已提供的时间划分，不重新随机划分外层验证；可在训练集内部做自己的交叉验证。
validation为两个探索周及6小时反馈热身，test为最后七日确认输入，没有标签。这里的submission.csv是合法过去确认输入的预测，并非未来服务资料。外层在全部搜索结束后才确认是否采用。模型随后须保存并在未来逐小时输入上服务。
可用numpy、pandas、scipy、scikit-learn、torch CPU；模型单线程，程序每次90秒，不能联网、启动进程或读取任务目录外文件。模型必须在predict的一行输入上独立工作，转换统计量只能在fit中学习。
评价由外层给出：稀疏参考只在0/6/12/18时于预测后到达，默认增益0.5、6小时衰减；探索两周RMSE平方等权平均后开方，越低越好。评分标签不会提供给生成程序。不要自行编造或打印一个评价分数。
程序使用固定接口（可自行定义sklearn兼容模型、特征、损失及组合）：
from harness_task import load_data, evaluate_model, submit_model
X_train,y_train,X_valid,X_test=load_data()
# 自己构建model并model.fit(X_train,y_train)
evaluate_model(model, label="候选或消融名称")
# Final Validation Performance 由外层真实预测评价器补写，保留这一行注释。
消融可以逐个构建并训练不同模型，再分别调用evaluate_model，外层会在stdout返回每项真实探索分数。该函数返回产物标识，不返回可在同一程序中调参的分数；如需内部调参，只使用train.csv内部资料。
最终提交阶段调用submit_model(model)，它保存./final/submission.csv和实际模型。不要只保存预测而遗漏模型；最终程序仍需包含上面的Final Validation Performance注释。所有模型都经过保存重载，并对每条输入单独预测。
可选反馈必须来自已提供设置；优先保持默认，以免代码之外的条件改变。若选择其他设置，evaluate_model/submit_model的feedback_gain与feedback_hours必须相同。
特征名：'''+json.dumps(names,ensure_ascii=False)+'''
下面是此前开发得到的一个强固定训练实现，供检索和初解参考。此源码只说明已有能力，不是本批试验结果；fit参数X、y对应上述全部特征与绝对target。可将build_model返回的对象用于上述接口：
```python
'''+baseline_source+'\n```\n'


async def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'task-identity.json').exists():raise ValueError('完整作者运行不可覆盖；失败及原响应必须保留')
    commit=verify_source();config=json.loads(args.library.read_bytes());scene.configure(args.start)
    tools=scene.prepare(args.archive,True,True);features,names=code_inputs(tools)
    truth=scene.readings(args.archive,args.site).to_numpy()
    if np.isfinite(truth[scene.CLOCK<scene.START]).mean()<.8:raise ValueError('该站未满足事前历史覆盖资格')
    old,initial_fit,initial_windows,service,returned=scene.roles(truth);start=int(service[0]);at=start+28*24
    labels=np.r_[old[:start],returned];fit,windows,history=masks(labels,at)
    initial_history=np.arange(start-21*24-6,start);bank=StreamPlanBank(output/'initial-components',features,names)
    initial=bank.train(config['initial_plan'],output/'initial',features,names,initial_fit,old,initial_history,old[initial_history],initial_windows)
    baseline=bank.issue(config['initial_plan'],initial,initial_history,service,labels,np.maximum(0.,features[service,4]))
    np.savez_compressed(output/'baseline-issued.npz',ids=service,prediction=baseline)
    if not maintenance_ready(windows):
        # 入场合格、维护反馈不足的站仍需完整服务，双方按同一既定条件保持原版。
        np.savez_compressed(output/'issued.npz',ids=service,baseline=baseline,**{args.arm:baseline})
        atomic_json(output/'decision.json',{'adopted':False,'reason':'维护窗口反馈不足8条','counts':[int(w.sum()) for w in windows]})
        atomic_json(output/'predictions-complete.json',{'done':True})
        valid=np.isfinite(truth[service]);score=error(truth[service][valid],baseline[valid])
        summary={'scope':'新采集期完整交付；维护条件不满足','controller':args.arm,'site':args.site,'start':args.start,
                 'rmse':{'baseline':score,args.arm:score},'issued':len(service),'scored':int(valid.sum()),
                 'model_calls':0,'search_calls':0,'executions':0,'adopted':False}
        atomic_json(output/'scores.json',summary);print(json.dumps(summary,ensure_ascii=False));return
    validation=history[history<at-7*24];confirmation=history[history>=at-7*24]
    oracle={'validation_y':labels[validation],'confirmation_y':labels[confirmation],'validation_hours':validation,
            'confirmation_hours':confirmation,'exploration_masks':[w[validation] for w in windows[:2]]}
    np.savez_compressed(output/'oracle.npz',**{k:v for k,v in oracle.items() if k!='exploration_masks'})
    task_name='London_'+args.site;task=output/'tasks'/task_name;task.mkdir(parents=True)
    frame(features[fit],names,task/'train.csv',labels[fit]);frame(features[validation],names,task/'validation.csv');frame(features[confirmation],names,task/'test.csv')
    reference=config['plans'][config['default_plan_id']]['components'][0]['python_source']
    description=task_description(names,reference)
    if args.frontier_task:
        from harness.integrations.frontier_task import DELIVERY
        description+='\n'+DELIVERY
    (task/'task_description.txt').write_text(description,encoding='utf-8')
    if args.comparison_task:
        comparison=json.loads((args.comparison_task/'task-identity.json').read_bytes())
        current={p.name:sha256(p.read_bytes()).hexdigest() for p in task.iterdir()}
        if current!=comparison['task_files']:raise ValueError('对照双方的任务描述或合法输入不相同')
    atomic_json(output/'task-identity.json',{'author_commit':commit,'site':args.site,'start':args.start,
        'archive_sha256':sha256(args.archive.read_bytes()).hexdigest(),'library_sha256':sha256(args.library.read_bytes()).hexdigest(),
        'train_ids':np.flatnonzero(fit).tolist(),'validation_ids':validation.tolist(),'confirmation_ids':confirmation.tolist(),
        'task_files':{p.name:sha256(p.read_bytes()).hexdigest() for p in task.iterdir()},'future_inputs_exposed_to_driver':False})
    broker=AuthorBroker(output/'queue',max_calls=args.max_calls,timeout=360,replay_from=args.replay_from,
        prior_model_invocations=args.prior_model_invocations,prior_searches=args.prior_searches);workspace=output/'author-workspace'
    executor=AuthorExecutor(output/'executions',workspace,oracle)
    if args.arm=='project':
        from harness.integrations.free_research_driver import run_free_research
        try:
            selected=await run_free_research((task/'task_description.txt').read_text(encoding='utf-8'),task,workspace,executor,broker)
        finally:
            atomic_json(output/'driver-complete.json',{'done':True,'model_calls':broker.calls,'search_calls':broker.search_calls,
                'reused_model_calls':broker.reused_model_calls,'reused_search_calls':broker.reused_search_calls,
                'total_model_invocations':broker.prior_model_invocations+broker.calls-broker.reused_model_calls})
    elif args.arm=='comind':
        from harness.integrations.author_comind import run_comind
        workspace.mkdir(parents=True,exist_ok=True)
        seed=reference+'\nfrom harness_task import load_data,evaluate_model\nX,y,V,T=load_data()\nmodel=build_model().fit(X,y)\nevaluate_model(model)\n'
        selected=await run_comind(description,task,workspace,executor,broker,seed)
    else:
        pipeline,nodes=load_author_pipeline(AUTHOR,{'data_dir':str(task.parent),'task_name':task_name,'workspace_dir':str(workspace),
            'agent_model':'gpt-6-luna','exec_timeout':90},broker)
        from machine_learning_engineering.shared_libraries import code_util
        code_util.run_python_code=executor
        atomic_json(output/'author-graph.json',{'nodes':nodes,'stages':[n.name for n in pipeline.sub_agents],
            'adaptations':['模型供应方传输','内建搜索传输','目录与进程执行边界','外层真实评分','跳过云认证和聊天转接',
                '提交接口语义识别及真实文件核验','定向改进选择包含实际执行的最后一项']})
        state={}
        if args.resume_refinement_from:
            from harness.integrations.author_recovery import restore_refined_state
            from machine_learning_engineering.shared_libraries.config import CONFIG
            from machine_learning_engineering.sub_agents.refinement.agent import update_outer_loop_states
            previous=json.loads((args.resume_refinement_from/'task-identity.json').read_bytes())
            if previous['task_files']!={p.name:sha256(p.read_bytes()).hexdigest() for p in task.iterdir()}:
                raise ValueError('改进恢复与原任务资料不同')
            state,audit=restore_refined_state(args.resume_refinement_from,workspace,task_name,
                (task/'task_description.txt').read_text(encoding='utf-8'),CONFIG,update_outer_loop_states)
            atomic_json(output/'refinement-recovery.json',{'original_run':str(args.resume_refinement_from.resolve()),
                'branches':audit,'new_stages':['ensemble','submission'],'prior_model_invocations':args.prior_model_invocations,
                'selection_uses':'仅原实际探索分数；未使用确认或服务期答案'})
            pipeline.sub_agents=pipeline.sub_agents[2:]
        sessions=InMemorySessionService();await sessions.create_session(app_name='machine_learning_engineering',user_id='local',session_id='author',state=state)
        runner=Runner(agent=pipeline,app_name='machine_learning_engineering',session_service=sessions)
        try:
            with (output/'events.jsonl').open('w',encoding='utf-8') as stream:
                async for event in runner.run_async(user_id='local',session_id='author',new_message=types.Content(role='user',parts=[types.Part(text='完整执行已配置的维护任务。')])):
                    stream.write(json.dumps({'id':event.id,'author':event.author,'timestamp':event.timestamp,
                        'state_keys':list(event.actions.state_delta),'has_content':bool(event.content)},ensure_ascii=False)+'\n');stream.flush()
        finally:
            session=await sessions.get_session(app_name='machine_learning_engineering',user_id='local',session_id='author')
            atomic_json(output/'session-state.json',dict(session.state))
            atomic_json(output/'driver-complete.json',{'done':True,'model_calls':broker.calls,'search_calls':broker.search_calls,
                'reused_model_calls':broker.reused_model_calls,'reused_search_calls':broker.reused_search_calls,
                'total_model_invocations':broker.prior_model_invocations+broker.calls-broker.reused_model_calls})
        submissions=[r for r in executor.records if r['actor'].startswith('submission') and r['returncode']==0 and r['candidates']]
        if not submissions:raise ValueError('作者完整流程未交付最终模型')
        selected=submissions[-1]
    finish_delivery(args,output,features,names,labels,history,service,windows,at,baseline,task_name,task,executor,workspace,broker,selected,truth)


def finish_delivery(args,output,features,names,labels,history,service,windows,at,baseline,task_name,task,executor,workspace,broker,selected,truth,summary_extra=None):
    """保留各系统的发布语义；CoMind交付其选优时保存的模型。"""
    arm=getattr(args,'arm','paper')
    start=int(service[0]);returned=labels[service];candidate=selected['candidates'][-1]
    selected_path=Path(candidate['path']);raw=np.load(selected_path.with_suffix('.npz'),allow_pickle=False)
    past=np.r_[raw['validation'],raw['confirmation']];feedback=setting_of(candidate['feedback'])
    issued_past=correct(past[:,None],labels[history],history,feedback)[:,0]
    candidate_errors=[error(labels[history][w[history]],issued_past[w[history]]) for w in windows]
    parent_errors=[error(labels[w],baseline[np.flatnonzero(w)-start]) for w in windows]
    adopted=gate(np.asarray(parent_errors),np.asarray(candidate_errors))
    atomic_json(output/'decision.json',{'adopted':adopted,'parent_errors':parent_errors,'candidate_errors':candidate_errors,'selected_execution':selected['sequence'],
        'source_sha256':selected['source_sha256'],'weights_sha256':candidate['weights_sha256'],
        'candidate_id':candidate_id(candidate),'candidate_label':candidate['label']})
    if arm=='comind':
        # 作者框架返回原best_submission；额外重训会改变对照方法及其实际预测。
        published=candidate
        publication_rule='retain_author_selected_weights'
    else:
        # 项目驱动已经明确告知发布补训；确认封存后运行，训练内分数不再反馈给驱动。
        publication=workspace/task_name/'publication';(publication/'input').mkdir(parents=True)
        for p in task.iterdir():shutil.copy2(p,publication/'input'/p.name)
        pubfit=(np.arange(len(labels))<at)&np.isfinite(labels)
        frame(features[pubfit],names,publication/'input/train.csv',labels[pubfit])
        executor.phase='publication';code=Path(selected['source']).read_text(encoding='utf-8')
        pub=executor(code,publication,'final_solution.py',90)
        if pub['returncode']!=0:raise ValueError('最终程序的合法发布补训失败，原失败及旧版保留')
        published=match_publication(candidate,executor.records[-1])
        publication_rule='refit_frozen_program_with_legal_history'
    pub_path=Path(published['path'])
    if sha256(pub_path.with_suffix('.pkl').read_bytes()).hexdigest()!=published['weights_sha256']:
        raise ValueError('发布权重与候选记录不一致')
    atomic_json(output/'publication.json',{'rule':publication_rule,'selected_weights_sha256':candidate['weights_sha256'],
        'published_weights_sha256':published['weights_sha256'],'candidate_path':str(pub_path)})
    serving=output/'serving';(serving/'input').mkdir(parents=True)
    shutil.copy2(pub_path.with_suffix('.pkl'),serving/'input/model.pkl');future=service[service>=at]
    frame(features[future],names,serving/'input/service.csv')
    script=serving/'predict.py';script.write_text('''from pathlib import Path
import cloudpickle
import numpy as np
import pandas as pd
model=cloudpickle.loads(Path('input/model.pkl').read_bytes())
rows=pd.read_csv('input/service.csv')
values=[]
for i in range(len(rows)):
    p=np.asarray(model.predict(rows.iloc[[i]]),dtype=float)
    if p.shape!=(1,) or not np.isfinite(p).all():raise ValueError('单条服务预测无效')
    values.append(max(0.,float(p[0])))
np.save('prediction.npy',np.asarray(values))
''',encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/study/execute_author_program.py'),'--script',str(script)],cwd=serving,env=env,capture_output=True,text=True,encoding='utf-8',timeout=90)
    (serving/'stdout.log').write_text(result.stdout,encoding='utf-8');(serving/'stderr.log').write_text(result.stderr,encoding='utf-8')
    if result.returncode:raise ValueError('发布模型逐条服务失败')
    with np.load(pub_path.with_suffix('.npz'),allow_pickle=False) as z:past=np.r_[z['validation'],z['confirmation']]
    base=np.r_[past,np.load(serving/'prediction.npy',allow_pickle=False)];clock=np.r_[history,future]
    updated=correct(base[:,None],labels[clock],clock,setting_of(published['feedback']))[:,0][len(history):]
    direct=np.r_[baseline[:len(service)-len(future)],updated];actual=direct if adopted else baseline.copy()
    np.savez_compressed(output/'issued.npz',ids=service,baseline=baseline,returned=returned,**{arm:actual,arm+'_direct':direct})
    atomic_json(output/'predictions-complete.json',{'done':True})
    valid=np.isfinite(truth[service]);scores={k:error(truth[service][valid],v[valid]) for k,v in {'baseline':baseline,arm:actual,arm+'_direct':direct}.items()}
    summary={'scope':'已见开发资料的完整源码研究与逐小时交付','controller':arm,'site':args.site,'start':args.start,'rmse':scores,
        'issued':len(service),'scored':int(valid.sum()),'model_calls':broker.calls,'search_calls':broker.search_calls,'executions':len(executor.records),
        'reused_model_calls':broker.reused_model_calls,'reused_search_calls':broker.reused_search_calls,
        'total_model_invocations':broker.prior_model_invocations+broker.calls-broker.reused_model_calls,
        'adopted':adopted,'author_complete':arm=='paper','independent_verification':False,'same_budget_project_comparison':False}
    if getattr(args,'frontier_task',False):
        summary.update(scope='既有站点2026新采集期，完整研究与逐小时交付',
                       author_complete=arm in ('paper','comind'),
                       benchmark='CoMind ICLR2026同条件完整框架适配比较',
                       paired_budget_verification='由双方完成后的共同核验执行')
    if summary_extra:summary.update(summary_extra)
    atomic_json(output/'scores.json',summary);print(json.dumps(summary,ensure_ascii=False))


def main(argv=None):
    parser=ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--start',required=True)
    parser.add_argument('--site',required=True);parser.add_argument('--library',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-calls',type=int,default=48);parser.add_argument('--replay-from',type=Path)
    parser.add_argument('--prior-model-invocations',type=int,default=0);parser.add_argument('--prior-searches',type=int,default=0)
    parser.add_argument('--arm',choices=('paper','project','comind'),default='paper')
    parser.add_argument('--frontier-task',action='store_true',help='为双方加入共同可重载模型及双CSV任务契约')
    parser.add_argument('--comparison-task',type=Path,help='要求任务描述和全部合法任务文件与既有对手逐字相同')
    parser.add_argument('--resume-refinement-from',type=Path,help='在原实际初解与改进基础上修正选优，重新执行作者集成和提交')
    args=parser.parse_args(argv);asyncio.run(run(args))


if __name__=='__main__':main()
