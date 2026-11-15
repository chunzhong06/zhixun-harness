"""在第1及第6轮合法反馈上真实提案，交付后续发动机预测并保留同工具强对手。"""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from engineering.cli.study.engine_parent_capacity import sources, labeled
from engineering.cli.workflow.london_code_driver import fit_source
from engineering.cli.workflow import public_maintenance as public
from harness.contracts import digest
from harness.integrations import engine_parent_view
from harness.integrations.generated_regressor import IMPORTS, NUMPY, load_module
from harness.integrations.proposal_inbox import ProposalInbox
from harness.integrations.published_parent import augment, read, reference_source
from harness.integrations.pytrio_planner import ChatRequest
from harness.methods.training_inquiry import LabeledBatch
from harness.runtime.storage import LocalStore, atomic_json

ROOT = Path(__file__).resolve().parents[3]


def predict(folder, x):
    load_module(folder / "source.py"); model = joblib.load(folder / "model.joblib")
    output=[]
    for row in x:
        value=np.asarray(model.predict(row[None,:]),dtype=float)
        if value.shape!=(1,) or not np.isfinite(value).all(): raise ValueError("生成预测不是单个有限值")
        output.append(max(0.,float(value[0])))
    return np.asarray(output)


def gate(old, new):
    return new["error"] < old["error"] and new["worst"] <= 1.01*old["worst"]


def observation(rows, x, prediction):
    score=labeled(rows,x).score(prediction); y=np.asarray([r["label"] for r in rows]); error=prediction-y
    score["prediction_minus_answer"]=float(error.mean())
    score["by_parent_prediction"]=[]
    for lo,hi in ((0,50),(50,100),(100,150),(150,10000)):
        use=(x[:,-1]>=lo)&(x[:,-1]<hi)
        if use.any():score["by_parent_prediction"].append({"interval":[lo,hi],"rows":int(use.sum()),
            "rmse":float(np.sqrt(np.mean(error[use]**2))),"prediction_minus_answer":float(error[use].mean())})
    score["largest_errors"]=[{"input":x[i].tolist(),"answer":float(y[i]),"prediction":float(prediction[i])}
                              for i in np.argsort(-np.abs(error))[:6]]
    return score


def main(argv):
    ap=ArgumentParser(description=__doc__)
    ap.add_argument("--prior",type=Path,required=True)
    ap.add_argument("--capacity",type=Path,required=True)
    ap.add_argument("--data-root",type=Path,default=ROOT/'data/official')
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--resume-from",type=Path,help="只复用逐请求身份相同的原提案和已完成训练；原超时仍占一个名额")
    args=ap.parse_args(argv)
    with LocalStore(args.output).exclusive(), TemporaryDirectory(dir=args.output,prefix='temporary-') as temporary:
        if (args.output/'scores.json').exists():raise ValueError('不覆盖已有现场比较')
        os.environ['TEMP']=temporary;os.environ['TMP']=temporary
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
        import tempfile
        tempfile.tempdir=temporary
        binding=read(args.capacity/'binding.json')
        samples=read(ROOT/binding['run']/'regular/state.json')['samples']
        serve=sorted((r for r in samples.values() if r['role']=='serve'),key=lambda r:(r['available_at'],r['sample_id']))
        train={}
        for view in ('raw','native'):
            for scope in ('anchors','full'):
                with np.load(args.prior/f'{view}-{scope}.npz',allow_pickle=False) as z:
                    train[view,scope]=(z['train_x'],z['train_y'],z['train_ids'])
        def views(rows):
            raw=np.asarray([r['values'] for r in rows]); native,parent=engine_parent_view.transform(binding,raw,args.data_root)
            return {'raw':np.column_stack((raw,parent)),'native':np.column_stack((native,parent))}
        pool={}; canonical_sources=sources()
        for view in ('raw','native'):
            for scope in ('anchors','full'):
                for learner,source in canonical_sources.items():
                    name=f'{view}-{scope}-{learner}'
                    folder=(args.capacity/'models'/learner if (view,scope)==('raw','anchors') else args.prior/'models'/name)
                    published=args.output/'shared-programs'/name;published.mkdir(parents=True,exist_ok=True)
                    for filename in ('source.py','model.joblib'):shutil.copy2(folder/filename,published/filename)
                    atomic_json(published/'program.json',{'binding':binding,'feature_view':view,
                        'view_sha256':sha256(Path(engine_parent_view.__file__).read_bytes()).hexdigest(),
                        'source_sha256':sha256((published/'source.py').read_bytes()).hexdigest(),
                        'weights_sha256':sha256((published/'model.joblib').read_bytes()).hexdigest()})
                    pool[name]={'folder':published,'view':view,'scope':scope,'source':source,'available_at':0,'name':name}
        stages={}
        for stage in (0,5):
            explore=sorted((r for r in samples.values() if r['role']=='explore' and r['available_at']==stage),key=lambda r:r['sample_id'])
            confirm=sorted((r for r in samples.values() if r['role']=='adopt' and r['available_at']==stage),key=lambda r:r['sample_id'])
            ex,cx=views(explore),views(confirm)
            # 只有探索分数进入两边观察；确认原件由发布器单独持有。
            baseline={name:labeled(explore,ex[p['view']]).score(predict(p['folder'],ex[p['view']])) for name,p in pool.items()}
            stages[stage]={'explore':explore,'confirm':confirm,'ex':ex,'cx':cx,'baseline':baseline}
        arms={arm:{'current':None,'trials':[],'events':[],'issued':[]} for arm in ('project','paper')}
        def values(record,x):return x['raw'][:,-1] if record is None else predict(Path(record['folder']),x[record['view']])
        def run_arm(arm,stage):
            state=arms[arm]; task=stages[stage]; ex=task['ex']; rows=task['explore']; current=state['current']
            reference={'source':reference_source('engine'),'view':'raw','scope':'full'} if current is None else {k:current[k] for k in ('source','view','scope')}
            reference_identity=digest(reference); trials=[]
            # 已共享实训的对照也是可交付版本；停止新增试验不应丢弃已知更好候选。
            contenders=[current]+[{**p,'folder':str(p['folder'])} for p in pool.values()]
            for attempt in (1,2):
                context={'task_id':arm,'batch_id':f'engine-r{stage+1}-trial{attempt}',
                    'reference_identity':reference_identity,'reference':reference,
                    'task':'官方发动机剩余循环预测；只用已到期探索反馈提案。当前为已见开发资料，但禁止读取确认和服务答案。',
                    'input_contract':{'columns':'前24列为21传感器和3工况，末列为完整冻结父版预测；raw原值/native为父版冻结的工况刻度，两者宽度25。',
                        'fit_labels':'合法源设备3968锚点或113089全量；fit(X,y)接收绝对剩余循环。只能在fit内使用这份标签。',
                        'scope_options':['anchors','full'],'view_options':['raw','native'],'prediction':'每次仅一行；必须有限、无状态。'},
                    'current_exploration':observation(rows,ex['raw'],values(current,ex)),
                    'same_tools_component_comparison':task['baseline'],'fixed_sources':canonical_sources,
                    'source_target_summary':{'rows':len(train['raw','full'][1]),'minimum':float(train['raw','full'][1].min()),
                        'median':float(np.median(train['raw','full'][1])),'maximum':float(train['raw','full'][1].max())},
                    'trials_this_stage':trials,'previous_publication':[{'stage':e['stage'],'adopted':e['adopted']} for e in state['events']],
                    'allowed_imports':{k:sorted(v) if v is not None else None for k,v in IMPORTS.items()},'numpy_operations':sorted(NUMPY),
                    'proposal_format':{'action':'experiment','reference_identity':reference_identity,'hypothesis':'中文可检验假设','view':'raw或native',
                        'scope':'anchors或full','python_source':'完整build_model()代码，20KB内。也可以action:stop并写reason。'},
                    'budget':{'requests_remaining':3-attempt,'worker_seconds':90,'max_trees':128,'max_iterations':256,'threads':1}}
                policy=('自主根据实际误差、组件比较和已做试验，决定最值得试的完整改法；可停止。' if arm=='project' else
                    '采用MLE-STAR定向改进的本地机制：先从同资料的组件比较定位输入刻度、训练量或学习器中最有依据的一项；集中修改该项，第二案依据真实结果修订，并保留较好解。这里不包含联网模型检索及完整多代理搜索。')
                instruction='你是隔离的gpt-6-luna现场驱动。'+policy+'两组共享同样工具与信息。仅返回规定JSON方案；禁止读取队友目录、项目其他实验及最终评分。代码禁止I/O、动态执行和额外导入。'
                request=ChatRequest('local-inbox',{'model':'gpt-6-luna','messages':[{'role':'system','content':instruction},
                    {'role':'user','content':json.dumps(context,ensure_ascii=False)}]},75)
                folder=args.output/arm/f'round-{stage+1}'/f'trial-{attempt}';folder.mkdir(parents=True,exist_ok=True)
                key=digest(request.body);request_name=f"{context['batch_id']}-{key}.json"
                old_folder=None
                try:
                    if args.resume_from and arm=='paper' and stage==5 and attempt==1:
                        # 318项这一请求确实超时，后到的响应不能倒改原失败。
                        original=args.resume_from/'queue/requests'/arm/request_name
                        if not original.is_file():raise ValueError('原超时请求身份变化')
                        raise TimeoutError('复用318项已发生的75秒提案超时；迟到回复保留但不采用')
                    old_request=(args.resume_from/'queue/requests'/arm/request_name) if args.resume_from else None
                    old_response=(args.resume_from/'queue/responses'/arm/request_name) if args.resume_from else None
                    if old_request and old_request.is_file() and old_response.is_file():
                        original=read(old_request)
                        if original['context']!=context or original['instruction']!=instruction:raise ValueError('原观察不是本次相同请求')
                        for kind,old in (('requests',old_request),('responses',old_response)):
                            dest=args.output/'queue'/kind/arm/request_name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dest)
                        old_folder=args.resume_from/arm/f'round-{stage+1}'/f'trial-{attempt}'
                    response=ProposalInbox(args.output/'queue').send(request)
                    raw=json.loads(response.body['choices'][0]['message']['content'])
                except TimeoutError as exc:
                    trials.append({'success':False,'error':str(exc),'request_timeout':True})
                    atomic_json(folder/'request-error.json',{'request_id':key,'error':str(exc),'counts_toward_budget':True})
                    continue
                atomic_json(folder/'proposal.json',raw)
                if raw.get('action')=='stop':
                    trials.append({'stop':True,'reason':raw.get('reason','')});break
                try:
                    if raw.get('action')!='experiment' or raw.get('reference_identity')!=reference_identity:raise ValueError('动作或父版身份不符')
                    view,scope=raw['view'],raw['scope'];tx,ty,tids=train[view,scope]
                    if not str(raw.get('hypothesis','')).strip():raise ValueError('缺少可检验假设')
                    source=raw['python_source'];data=folder/'training-input.npz'
                    np.savez_compressed(data,train_x=tx,train_y=ty,train_ids=tids,predict_x=ex[view],predict_ids=np.arange(len(rows)))
                    program=folder/'program'
                    if old_folder and (old_folder/'program/base.npz').is_file():
                        if read(old_folder/'proposal.json')!=raw:raise ValueError('重放提案与原件不同')
                        with np.load(old_folder/'training-input.npz',allow_pickle=False) as z:
                            expected={'train_x':tx,'train_y':ty,'train_ids':tids,'predict_x':ex[view],'predict_ids':np.arange(len(rows))}
                            if set(z.files)!=set(expected) or any(not np.array_equal(z[k],v) for k,v in expected.items()):raise ValueError('原训练或探索输入改变')
                        shutil.copytree(old_folder/'program',program)
                        with np.load(program/'base.npz',allow_pickle=False) as z:index,pred=z['ids'],z['prediction']
                        atomic_json(folder/'replay.json',{'original':str(old_folder),'request_id':key,'same_proposal_and_arrays':True})
                    else:index,pred=fit_source(source,data,program)
                    if not np.array_equal(index,np.arange(len(rows))):raise ValueError('试训行顺序不符')
                    if not np.allclose(pred,predict(program,ex[view]),rtol=0,atol=1e-9):raise ValueError('重载无法复现探索预测')
                    manifest={'binding':binding,'feature_view':view,'view_sha256':sha256(Path(engine_parent_view.__file__).read_bytes()).hexdigest(),
                        'source_sha256':sha256((program/'source.py').read_bytes()).hexdigest(),'weights_sha256':sha256((program/'model.joblib').read_bytes()).hexdigest()}
                    atomic_json(program/'program.json',manifest)
                    record={'folder':str(program),'view':view,'scope':scope,'source':source,'available_at':stage,'name':f'{arm}-r{stage+1}-t{attempt}'}
                    contenders.append(record);state['trials'].append(record)
                    trials.append({'success':True,'hypothesis':raw['hypothesis'],'view':view,'scope':scope,'python_source':source,
                        'exploration':observation(rows,ex[view],pred)})
                except Exception as exc:trials.append({'success':False,'error':str(exc),'hypothesis':raw.get('hypothesis')})
            nominated=min(contenders,key=lambda p:labeled(rows,ex['raw']).score(values(p,ex))['error'])
            confirmed=labeled(task['confirm'],task['cx']['raw'])
            old,new=confirmed.score(values(current,task['cx'])),confirmed.score(values(nominated,task['cx']))
            adopted=gate(old,new);state['current']=nominated if adopted else current
            event={'stage':stage,'trials':trials,'nomination':None if nominated is None else nominated['name'],
                'adopted':adopted,'confirmation':{'incumbent':old,'candidate':new},'published':state['current']}
            state['events'].append(event);atomic_json(args.output/arm/f'round-{stage+1}'/'decision.json',event)
        for stage in (0,5):
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(lambda arm:run_arm(arm,stage),('project','paper')))
            current_rows=[r for r in serve if stage<=r['available_at']<(5 if stage==0 else 10)]
            sx=views(current_rows)
            for arm,state in arms.items():
                prediction=values(state['current'],sx)
                folder=args.output/arm/f'round-{stage+1}';np.savez_compressed(folder/'issued.npz',ids=[r['sample_id'] for r in current_rows],prediction=prediction)
                state['issued'].extend(prediction.tolist())
        atomic_json(args.output/'drivers-complete.json',{'complete':True})
        # 交付已冻结，原有固定模型和现场实际创造的所有源码均作为同工具固定对手。
        sx=views(serve);predictions={'parent':sx['raw'][:,-1],**{arm:np.asarray(state['issued']) for arm,state in arms.items()}}
        bank=dict(pool)
        for state in arms.values():
            for record in state['trials']:bank[record['name']]={**record,'folder':Path(record['folder'])}
        for name,record in bank.items():
            full=predict(record['folder'],sx[record['view']])
            # 第六轮才提出的源码也不能作为固定对手倒改前五轮。
            eligible=np.asarray([r['available_at']>=record['available_at'] for r in serve])
            for weight in (1.,.25,.5):
                suffix=name if weight==1 else name+f'-mix-{weight}'
                mixed=weight*full+(1-weight)*predictions['parent']
                predictions['fixed-direct-'+suffix]=np.where(eligible,mixed,predictions['parent'])
                accepted=[];active=False
                for stage in (0,5):
                    task=stages[stage];confirm=labeled(task['confirm'],task['cx']['raw']);parent_values=task['cx']['raw'][:,-1]
                    new=weight*predict(record['folder'],task['cx'][record['view']])+(1-weight)*parent_values
                    old=new if active else parent_values
                    if stage>=record['available_at'] and gate(confirm.score(old),confirm.score(new)):active=True
                    mask=np.asarray([stage<=r['available_at']<(5 if stage==0 else 10) for r in serve])
                    accepted.extend((mixed[mask] if active else predictions['parent'][mask]).tolist())
                predictions['fixed-gated-'+suffix]=np.asarray(accepted)
        np.savez_compressed(args.output/'issued.npz',ids=[r['sample_id'] for r in serve],names=list(predictions),predictions=np.column_stack(list(predictions.values())))
        # 独立进程仅从实际发布程序重放完整发放，确认当时记录可由正式单条入口复现。
        checks=[]
        for arm,state in arms.items():
            for event in state['events']:
                current=event['published'];stage=event['stage']
                selected=[r for r in serve if stage<=r['available_at']<(5 if stage==0 else 10)]
                if current is None:
                    replay=augment(binding,[r['values'] for r in selected],args.data_root)[:,-1]
                else:
                    lines='\n'.join(json.dumps({'task':'regression','inputs':{'sensors':r['values'][:21],'settings':r['values'][21:]}}) for r in selected)+'\n'
                    response=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/workflow/run_workflow.py'),'published-program-predict',
                        '--program',str(Path(current['folder']).resolve()),'--data-root',str(args.data_root.resolve())],input=lines,capture_output=True,text=True,encoding='utf-8',timeout=60)
                    if response.returncode:raise ValueError(response.stderr[-1200:])
                    replay=np.asarray([json.loads(line)['outputs']['value'] for line in response.stdout.splitlines()])
                with np.load(args.output/arm/f'round-{stage+1}'/'issued.npz',allow_pickle=False) as z:
                    if not np.allclose(replay,z['prediction'],rtol=0,atol=1e-9):raise ValueError('正式预测入口没有复现实际发布')
                checks.append({'arm':arm,'stage':stage,'rows':len(selected),'matched':True,'separate_process':current is not None})
        package=public.PACKAGES['engine'][2](args.data_root/'engine'/public.PACKAGES['engine'][0]);targets={r.sample_id:r.label for r in (*package.train,*package.validation)}
        scoring=LabeledBatch(sx['raw'],np.asarray([targets[r['sample_id']] for r in serve]),tuple(r['unit_id'] for r in serve))
        scores={name:scoring.score(pred) for name,pred in predictions.items()}
        fixed=min((name for name in scores if name.startswith('fixed-') or name=='parent'),key=lambda k:scores[k]['error'])
        result={'scope':'已见官方发动机，两次真实维护提案、十轮完整交付；MLE-STAR定向改进本地适配，不是完整作者复现',
            'scores':scores,'best_fixed':fixed,'driver_calls':{arm:sum(len(e['trials']) for e in state['events']) for arm,state in arms.items()},
            'issued_rows':len(serve),'entry_checks':checks}
        atomic_json(args.output/'scores.json',result)
        print(json.dumps({'project':scores['project'],'paper':scores['paper'],'best_fixed':fixed,'fixed_score':scores[fixed],
            'driver_calls':result['driver_calls'],'issued_rows':len(serve)},ensure_ascii=False),flush=True)
