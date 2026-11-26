"""平台当轮输入转换；研究、训练、确认和发布调用共同内核。"""
from argparse import ArgumentParser
from contextlib import redirect_stdout
from hashlib import sha256
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
from harness.adapters.storage.local import atomic_json, LocalStore
from harness.adapters.datasets.baseline_sources import DirectorySources
from harness.learning.inference.complete_checkpoint import load_checkpoint
from harness.engine.research_recovery import file_identity, verify_files


def _probe_indices(length, limit=32):
    """确定性选取接口探针；完整服务数组仍来自一次批量预测。"""
    if type(length) is not int or length < 0 or type(limit) is not int or limit < 1:
        raise ValueError('探针长度和上限必须是非负整数与正整数')
    if length == 0:
        return np.asarray([], dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, min(length, limit), dtype=np.int64))


def _probe_limit(default=32):
    raw = os.environ.get('HARNESS_A_PROBE_ROWS', str(default))
    try:
        limit = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError('HARNESS_A_PROBE_ROWS 必须是 1 至 128 的整数') from exc
    if not 1 <= limit <= 128:
        raise ValueError('HARNESS_A_PROBE_ROWS 必须是 1 至 128 的整数')
    return limit


def prepare_round(profile, train_path, validation_path, parent, work, *, baseline_directory=None,
                  selection_policy='public-validation', tabular_resource=None, training_hint=None):
    """比赛用全公开验证选案；confirmation仅供显式冻结的研究协议。

    两种模式的验证标签均只留在宿主评价端，不进入训练输入或尺度拟合。
    public-validation保留旧数组接口，但explore/confirm是同一集合，不是独立确认。
    """
    from harness.adapters.platforms.baselines.core_model import Model
    from harness.adapters.platforms.baselines.data import read_dataset
    from harness.adapters.platforms import baselines
    if selection_policy not in ('public-validation', 'confirmation'):
        raise ValueError('未知的验证选案协议')
    work=Path(work).resolve(); work.mkdir(parents=True,exist_ok=False)
    taskdir=work/'task'; taskdir.mkdir()
    parent=Path(parent).resolve(); model=load_checkpoint(parent,Model)
    if profile not in {'A','B','C'} or model.meta['profile']!=profile:
        raise ValueError('平台任务与实际父版不匹配')
    train=read_dataset(train_path,model.meta['task'],model.meta['classes'],model.meta['paths'])
    valid=read_dataset(validation_path,model.meta['task'],model.meta['classes'],model.meta['paths'])
    if set(train['ids']) & set(valid['ids']): raise ValueError('训练与验证身份交叉')
    if tabular_resource is not None:
        rebase = 'context_train_sha256' in tabular_resource
        fields = {'checkpoint','sha256','context_train_sha256'} if rebase else {'checkpoint','sha256'}
        if profile not in ('A','B') or set(tabular_resource) != fields:
            raise ValueError('共享TabICL资源仅用于气体或轴承且须提供精确身份')
        resource_path=Path(tabular_resource['checkpoint']).resolve()
        if sha256(resource_path.read_bytes()).hexdigest()!=tabular_resource['sha256']:
            raise ValueError('共享TabICL资源身份不符')
        if rebase and profile == 'B':
            from harness.learning.training.teacher_resources import rebuild_bearing_resource
            tabular_resource=rebuild_bearing_resource(tabular_resource,train,work/'current-bearing-resource.npz')
        elif rebase:
            from harness.learning.training.teacher_resources import rebuild_gas_resource
            tabular_resource=rebuild_gas_resource(tabular_resource,train,work/'current-gas-resource.npz',model.meta['classes'])
        else:
            tabular_resource={'checkpoint':str(resource_path),'sha256':tabular_resource['sha256']}
    def groups(path, *, training=False):
        rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
        result = []
        for row in rows:
            meta = row.get('meta') or {}
            device = meta.get('unit')
            if device is None:
                device = meta.get('bearing')
            if training and profile == 'C' and device is None:
                result.append('source')
            else:
                result.append(str((meta.get('source_subset','source'),device if device is not None else 'source')))
        return result
    units=groups(validation_path)
    grouped=len(set(units))>=2
    if selection_policy == 'public-validation':
        if not len(valid['ids']): raise ValueError('公开验证不能为空')
        masks={role:np.ones(len(valid['ids']),dtype=bool) for role in ('explore','adopt')}
        split='全公开验证用于选案；两个数组别名不构成独立确认'
    else:
        # 独立确认研究有多台身份时整台分离，否则按ID固定切分。
        keys=units if grouped else valid['ids']
        ranked=sorted(set(keys),key=lambda value:sha256(('platform-confirm-v1:'+value).encode()).hexdigest())
        if len(ranked)<2: raise ValueError('独立探索与确认至少需要两个可分离单位')
        roles={key:index%2 for index,key in enumerate(ranked)}
        masks={name:np.asarray([roles[key]==role for key in keys]) for name,role in [('explore',0),('adopt',1)]}
        split='整台身份固定切分' if grouped else '样本身份固定切分'
    task={'A':'gas','B':'bearing','C':'engine'}[profile]
    labels=valid['y']+1 if task=='gas' else valid['y']
    oracle={role:{'ids':list(np.asarray(valid['ids'])[mask]),'groups':list(np.asarray(units)[mask]),
                  'y':labels[mask].tolist()} for role,mask in masks.items()}
    # 这些输出仅用作预测接口探针，平台服务答案不可见，故不填服务分数。
    oracle['serve']={'ids':valid['ids'],'groups':units,'y':None}
    (taskdir/'initial.npz').write_bytes(parent.read_bytes())
    np.savez_compressed(taskdir/'train.npz',x=train['x'],y=train['y'],groups=np.asarray(groups(train_path,training=True)))
    np.savez_compressed(taskdir/'evaluation-inputs.npz',explore=valid['x'][masks['explore']],confirm=valid['x'][masks['adopt']])
    np.savez_compressed(work/'service-inputs.npz',x=valid['x'])
    if hasattr(model, 'bind_round_cache'):
        # 只缓存冻结教师对特征的输出；公开验证标签仍仅在宿主评分端使用。
        cache = work / '.frozen-teacher' / sha256(parent.read_bytes()).hexdigest()
        hint_name = model.training_hint_name(train['x']) if hasattr(model, 'training_hint_name') else None
        if profile != 'A' and training_hint is not None and hint_name is not None:
            from harness.learning.inference.teacher_cache import copy_training_hint
            copy_training_hint(training_hint, cache, name=hint_name)
        model.bind_round_cache(cache)
        if profile != 'A':
            # 已有回归路径预取合法训练行。气体只复用本轮内部评价的教师概率。
            from time import monotonic, process_time
            start_wall, start_cpu = monotonic(), process_time()
            model.prepare_training(train['x'])
            atomic_json(work/'training-cache-start.json', {'training_hit': bool(model.round_cache_hits['training']),
                                                          'training_hint_name': hint_name,
                                                          'training_rows': len(train['x']),
                                                          'teacher_wall_s': round(monotonic()-start_wall, 3),
                                                          'teacher_cpu_s': round(process_time()-start_cpu, 3)})
    pred=getattr(model,'predict_batch',model.predict)(valid['x'])
    if profile == 'A' and selection_policy == 'public-validation':
        # 完整服务数组复用批量结果，只做有限单行调用确认接口一致性。
        probe_indices = _probe_indices(len(valid['x']), _probe_limit())
        probe_pred = np.asarray([model.predict(row[None,:])[0]
                                 for row in valid['x'][probe_indices]])
        served_pred = pred.copy()
        probe_difference = float(np.max(np.abs(probe_pred-pred[probe_indices]))) if len(probe_indices) else 0.
        probe_record = {'rows':len(valid['x']),'probe_rows':len(probe_indices),
                        'probe_indices':probe_indices.tolist(),'max_absolute_difference':probe_difference,
                        'policy':'batch_full_with_small_interface_probe',
                        'probe_limit_env':'HARNESS_A_PROBE_ROWS'}
        atomic_json(work/'service-probe.json', probe_record)
    else:
        served_pred = np.asarray([model.predict(row[None,:])[0] for row in valid['x']])
        probe_indices = np.asarray([], dtype=np.int64)
        probe_pred = pred[:0].copy()
        probe_record = None
    initial_predictions = {'explore':pred[masks['explore']],'adopt':pred[masks['adopt']],'serve':served_pred}
    if probe_record is not None:
        initial_predictions.update(serve_probe_indices=probe_indices,serve_probe=probe_pred)
    np.savez_compressed(work/'initial-predictions.npz',**initial_predictions)
    source_dir=Path(baseline_directory or Path(baselines.__file__).parent).resolve()
    sources=DirectorySources(source_dir)
    identities={name:sha256(sources.read(name)).hexdigest() for name in
                ['baselines/__init__.py','baselines/model.py','baselines/data.py','baselines/predict.py']}
    (taskdir/'baseline-model.py').write_bytes(sources.read('baselines/model.py'))
    task_record={'task':task,'profile':profile,'stage_id':'platform-round',
        'baseline_directory':str(source_dir),'baseline_sha256':identities,'selection_policy':selection_policy,
        'data_identity':{'train_sha256':train['sha256'],'validation_sha256':valid['sha256']},
        'validation_rows':len(valid['y']),'tabular_resource':tabular_resource,
        'service_role':'public_validation_probe',
        'archive_sha256':sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),
        'feature_paths':model.meta['paths'],'train_rows':len(train['y']),
        'inputs_sha256':{name:sha256((taskdir/name).read_bytes()).hexdigest() for name in ['train.npz','evaluation-inputs.npz']}}
    if probe_record is not None:
        task_record['service_probe'] = probe_record
    atomic_json(taskdir/'task.json',task_record)
    atomic_json(work/'oracle.json',oracle)
    atomic_json(work/'identity.json',{'task':task,'parent_sha256':sha256(parent.read_bytes()).hexdigest(),
        'train_sha256':train['sha256'],'validation_sha256':valid['sha256'],
        'selection_policy':selection_policy,'validation_split':split,
        'scope':'当前公开验证；服务输出是接口探针，无平台服务评分',
        'service_probe':probe_record})
    return taskdir,oracle


async def run_platform(profile, train, validation, parent, output, *, stage_id='platform-round',
                       state_dir=None, calls=3, max_epochs=60, broker_factory=None, resume_work=False,
                       recording='local', recording_settings=None, driver=None,
                       selection_policy='public-validation', tabular_resource=None, program_seconds=90,
                       model_name=None, final_choice_at_host=False):
    """平台传入 parent 始终权威；历史仅作经验，不替换实际父版。"""
    from harness.bootstrap import run_prepared
    output=Path(output).resolve(); parent=Path(parent).resolve()
    state=Path(state_dir or output.parent/'.harness-state').resolve()
    run_key=sha256((str(output)+stage_id).encode()).hexdigest()[:20]
    work=state/'runs'/run_key
    with LocalStore(state/'locks'/run_key).exclusive():
        if output.exists() and not resume_work: raise ValueError('交付目录已存在；不能覆盖原版本')
        from .round_budget import current
        prepare_args = (profile,train,validation,parent,work)
        prepare_options = dict(selection_policy=selection_policy,tabular_resource=tabular_resource,
                               training_hint=state/'prior-training-hint')
        prepared = None
        if not resume_work:
            budget = current.get()
            prepared = (budget.prepare(prepare_args,prepare_options) if budget else
                        prepare_round(*prepare_args,**prepare_options))
        parent_sha=sha256(parent.read_bytes()).hexdigest()
        history_path=state/('history-'+profile+'.json')
        history=json.loads(history_path.read_bytes()).get('events',[]) if history_path.exists() else []
        if not history:
            sidecar=parent.parent/'harness-history.json'
            if sidecar.is_file():
                previous=json.loads(sidecar.read_bytes())
                if previous.get('published_sha256')==parent_sha: history=previous['events']
        invocation=work/'platform-invocation.json'
        if resume_work:
            saved=json.loads(invocation.read_bytes());verify_files(saved['inputs'])
            if saved['parent_sha256']!=parent_sha: raise ValueError('恢复父版变化')
            if saved.get('selection_policy','confirmation')!=selection_policy:
                raise ValueError('恢复时不能改变公开验证选案协议')
            if saved.get('final_choice_at_host',False) != final_choice_at_host:
                raise ValueError('恢复不能改变本轮最终选案位置')
            if saved.get('program_seconds',90)!=program_seconds or saved.get('tabular_resource')!=tabular_resource:
                raise ValueError('恢复时不能改变程序预算或共享模型资源')
            history=saved['history']
            prepared=work/'task', json.loads((work/'oracle.json').read_bytes())
        else:
            atomic_json(invocation,{'parent_sha256':parent_sha,'stage_id':stage_id,'history':history,
                'selection_policy':selection_policy, 'final_choice_at_host':final_choice_at_host,
                'tabular_resource':tabular_resource,'program_seconds':program_seconds,
                'inputs':file_identity([parent,train,validation])})
        from harness.contest.lineage import parent_handoff
        handoff = parent_handoff(parent_sha, history)
        if not resume_work:
            specification_path = work/'task/task.json'
            specification = json.loads(specification_path.read_bytes())
            specification['parent_handoff'] = handoff
            if final_choice_at_host:
                specification['final_choice_at_host'] = True
                if profile == 'A' and budget:
                    specification['extra_program_seconds'] = budget.program_limit
                if profile == 'A' and budget and budget.defer_gas_reference:
                    specification['defer_gas_reference'] = True
            atomic_json(specification_path, specification)
        if driver not in (None, 'service', 'queue', 'fixture'):
            raise ValueError('平台驱动须为service、queue或显式注入的fixture')
        if driver == 'fixture' and broker_factory is None:
            raise ValueError('工程驱动须显式注入')
        if broker_factory is None and driver == 'service':
            # 比赛训练只能使用平台注入的rnd.model网关；禁止镜像自行读取凭据或直连外部服务。
            raise ValueError('service驱动必须由平台宿主注入broker_factory')
        else: driver=driver or 'fixture'
        result=await run_prepared({'A':'gas','B':'bearing','C':'engine'}[profile],work,driver=driver,calls=calls,stages=1,
            model=model_name or ('gpt-6-luna' if driver == 'queue' else 'Qwen/Qwen3.8-27B'),
            parent_checkpoint=parent,prepared=prepared,history=history,broker_factory=broker_factory,
            max_epochs=max_epochs,resume=resume_work,candidate_cache=True,program_seconds=program_seconds)
        output.mkdir(parents=True,exist_ok=resume_work)
        for file in (work/'release').rglob('*'):
            if file.is_file():
                relative=file.relative_to(work/'release')
                # 权重的原训练记录与本轮试验分开，保持父版时也不得复用旧轮报告冒充新训练。
                if relative.as_posix() in ('training.json','history.jsonl'):
                    relative=relative.with_name('selected-'+relative.name)
                dest=output/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                if relative.as_posix()=='manifest.json' and dest.exists():
                    original=json.loads(file.read_bytes());existing=json.loads(dest.read_bytes())
                    original.pop('artifact_bytes');existing.pop('artifact_bytes')
                    if original!=existing:raise ValueError('已有平台发布清单身份改变')
                    continue
                if dest.exists() and dest.read_bytes()!=file.read_bytes():raise ValueError('已有交付内容不同')
                # 交付身份取决于文件字节；非root挂载不一定允许复制时间戳等元数据。
                if not dest.exists():shutil.copyfile(file,dest)
        event=result['stages'][0]
        trace=json.loads((work/'stage-1/trace.json').read_bytes())
        publication={'format':'harness.platform-round.v2','stage_id':stage_id,'profile':profile,
            'actual_parent_sha256':parent_sha,'published_sha256':event['published_sha256'],
            'parent_handoff':handoff,
            'decision':'adopt' if event['accepted'] else 'retain','platform_adoption':'等待下一轮实际父版确认',
            'recording_directory':str(work),'event':event,'history_available':bool(history)}
        from harness.contest.round_report import write_round_report, finalize_manifest
        publication['round_report']=write_round_report(output,work,parent,validation,stage_id,profile,event,trace,history)
        atomic_json(output/'platform.json',publication)
        atomic_json(output/'trace.json',trace)
        history_event={**event,'round_id':run_key,'platform_stage_id':stage_id,'actual_parent_sha256':parent_sha,
                       'parent_handoff':handoff}
        atomic_json(output/'harness-history.json',{'published_sha256':event['published_sha256'],'events':history+[history_event]})
        # 补传旧轮次不能覆盖更晚的研究历史；不同运行在同一任务锁下追加。
        with LocalStore(state/'history-locks'/profile).exclusive():
            latest=json.loads(history_path.read_bytes()).get('events',[]) if history_path.exists() else list(history)
            existing=[row for row in latest if row.get('round_id')==run_key]
            if existing and existing!=[history_event]: raise ValueError('同一运行的历史事件已变化')
            if not existing: latest.append(history_event)
            atomic_json(history_path,{'events':latest})
        receipt=(json.loads((output/'recording-status.json').read_bytes()) if (output/'recording-status.json').exists()
                 else {'status':'pending','recording_directory':str(work)})
        if recording=='online':
            from harness.adapters.tracking.native_recording import publish_completed_run,DEFAULT_SETTINGS
            try: receipt=publish_completed_run(work,settings_path=recording_settings or DEFAULT_SETTINGS,allow_network=True)
            except Exception as exc: receipt={'status':'pending','error_type':type(exc).__name__,'new_training_on_retry':False}
        atomic_json(output/'recording-status.json',receipt)
        finalize_manifest(output)
        return publication


def main(argv=None):
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True,choices=['A','B','C'])
    parser.add_argument('--train',required=True,type=Path);parser.add_argument('--validation',required=True,type=Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--resume',type=Path);group.add_argument('--from-baseline',type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--stage-id',default='platform-round')
    parser.add_argument('--state-dir',type=Path,default=os.environ.get('HARNESS_STATE_DIR'))
    parser.add_argument('--resume-work',action='store_true')
    parser.add_argument('--calls',type=int,default=3);parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--recording',choices=['local','online'],default=os.environ.get('HARNESS_RECORDING','local'))
    parser.add_argument('--recording-settings',type=Path);parser.add_argument('--label-provenance')
    parser.add_argument('--driver',choices=['service','queue'],default='service',
                        help='比赛保留指定服务；研发可显式使用真实Luna隔离队列')
    parser.add_argument('--selection-policy',choices=['public-validation','confirmation'],default='public-validation',
                        help='比赛默认全公开验证选案；confirmation仅用于冻结的独立确认研究')
    parser.add_argument('--tabular-resource',type=Path,help='宿主提供的冻结教师资源JSON，含checkpoint及sha256')
    parser.add_argument('--program-seconds',type=int,help='单程序墙钟上限；普通90秒、TabICL 900秒、TabPFN 1800秒，受宿主时限约束')
    for name in ['learning-rate','batch-size','seed']:parser.add_argument('--'+name)
    args=parser.parse_args(argv)
    if any(getattr(args,k) is not None for k in ['learning_rate','batch_size','seed']):
        parser.error('这些参数由父状态及研究程序管理，不接受未生效的覆盖参数')
    if not 2<=args.calls<=16 or not 1<=args.epochs<=10000 or (args.program_seconds is not None and not 1<=args.program_seconds<=3600):
        parser.error('本地调用预算为2至16；训练预算沿参考执行器支持1至10000轮，另受执行时限约束')
    from harness.learning.training.teacher_resources import registered_tabular_resource
    parent=args.resume or args.from_baseline
    resource=json.loads(args.tabular_resource.read_bytes()) if args.tabular_resource else (
        registered_tabular_resource(args.profile,parent,args.train) if args.profile=='A' or args.epochs>=25 else None)
    from harness.learning.training.teacher_resources import default_program_seconds
    program_seconds=args.program_seconds if args.program_seconds is not None else default_program_seconds(parent, resource)
    # 平台把stdout作为一次JSON回执读取；逐轮诊断仍保留在stderr及正式记录中。
    with redirect_stdout(sys.stderr):
        result=asyncio.run(run_platform(args.profile,args.train,args.validation,args.resume or args.from_baseline,args.output,
            stage_id=args.stage_id,state_dir=args.state_dir,calls=args.calls,max_epochs=args.epochs,
            resume_work=args.resume_work,recording=args.recording,recording_settings=args.recording_settings,driver=args.driver,
            selection_policy=args.selection_policy,program_seconds=program_seconds,tabular_resource=resource))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
