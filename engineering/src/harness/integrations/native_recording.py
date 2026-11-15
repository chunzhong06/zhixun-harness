"""独立比赛运行的云端记录与补传；不依赖科研账本或个人技能脚本。"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

from harness.integrations.research_recovery import verify_files
from harness.integrations.research_tracking import ResearchSwanLab
from harness.runtime.storage import LocalStore, atomic_json

ROOT=Path(__file__).resolve().parents[4]
DEFAULT_SETTINGS=ROOT/'engineering/configs/tracking/swanlab.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()


def completed_payload(folder, settings):
    """计算产物先通过原完成清单核验，再建立独立于传输代码的记录身份。"""
    folder=Path(folder).resolve()
    completion=read(folder/'completion.json')
    verify_files(completion['artifacts'])
    if str((folder/'run.json').resolve()) not in completion['artifacts']:
        raise ValueError('完成清单没有绑定当前运行目录')
    report=read(folder/'run.json')
    if report['status']!='completed':raise ValueError('只能发布已完成且可核验的原生运行')
    traces=[]
    for index,event in enumerate(report['stages'],1):
        trace_path=folder/f'stage-{index}/trace.json'
        if str(trace_path.resolve()) not in completion['artifacts']:
            raise ValueError('逐轮证据不属于当前完成清单')
        trace=read(trace_path)
        if event['stage']!=index or trace['轮次']!=index or trace['任务']!=report['task'] or trace['采用决定']!=event:
            raise ValueError('逐轮证据与实际完成运行不对应')
        traces.append(trace)
    if len({t['trace_id'] for t in traces})!=len(traces) or not traces:
        raise ValueError('逐轮证据缺失或身份重复')
    binding={'schema_version':1,'run_directory':str(folder),'task':report['task'],
             'completion_sha256':sha256((folder/'completion.json').read_bytes()).hexdigest(),
             'traces_sha256':digest(traces),'stage_count':len(traces),
             'destination':{k:settings[k] for k in ['workspace','project','mode','sdk_version']}}
    return report,traces,binding


def publish_completed_run(folder, *, settings_path=DEFAULT_SETTINGS, allow_network=False, evidence_hook=None):
    """按固定身份创建或恢复单个云端作业，只上传缺失且未被改写的轮次。

    训练完成清单和记录完成回执分开。中断后重试本函数只补传；完成后的重复
    调用读取已核验回执，不重新训练或联网。evidence_hook仅供中断工程验证。
    """
    if not allow_network:raise PermissionError('在线记录需要显式选择online或--online')
    folder=Path(folder).resolve()
    settings=read(settings_path)
    report,traces,binding=completed_payload(folder,settings)
    output=folder/'recording'
    with LocalStore(output).exclusive():
        identity=digest(binding)
        session_path=output/'session.json'
        session={'binding':binding,'binding_sha256':identity,'cloud_run_id':'n'+identity[:15]}
        if session_path.exists():
            if read(session_path)!=session:raise ValueError('待恢复记录的模型、证据或云端目标已经改变')
        else:atomic_json(session_path,session)
        completed=output/'completed.json'
        if completed.exists():
            saved=read(completed)
            if saved['binding_sha256']!=identity or not saved['cloud_verified']:
                raise ValueError('记录完成回执不匹配')
            return {**saved,'reused_completed_recording':True,'uploaded_now':0}
        attempt_no=len(list(output.glob('attempt-*.json')))+1
        attempt_path=output/f'attempt-{attempt_no:03}.json'
        attempt={'attempt':attempt_no,'started_at':datetime.now(timezone.utc).isoformat(),'status':'started',
                 'binding_sha256':identity,'implementation_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
        atomic_json(attempt_path,attempt)
        tracker=None
        try:
            config={'记录类型':'独立原生运行','任务':report['task'],'轮次':len(traces),
                    '记录绑定SHA256':identity,'训练完成清单SHA256':binding['completion_sha256'],
                    '运行目录':str(folder),'证据范围':report['scope'],
                    '驱动模型':[r['model'] for r in report['stages']],
                    '外部模型实际调用':sum(r['calls'] for r in report['stages'] if r['driver']!='fixture')}
            tracker=ResearchSwanLab(settings,output,name=f"原生续训｜{report['task']}｜{len(traces)}轮",
                    config=config,kind='native-runtime',description='独立入口的真实原生运行过程；模型与确认结果按绑定证据记录。',
                    run_id=session['cloud_run_id'],resume_mode='allow')
            remote=tracker.sdk.Api().run(f"{settings['workspace']}/{settings['project']}/{tracker.id}")
            existing=ResearchSwanLab.read_remote_evidence(remote)
            expected={t['trace_id']:t for t in traces}
            existing_ids=set()
            for prior in existing:
                key=prior.get('trace_id')
                if key in existing_ids or expected.get(key)!=prior:
                    raise ValueError('云端已有证据与本次实际运行不一致，不能覆盖')
                existing_ids.add(key)
            uploaded=0
            for index,trace in enumerate(traces,1):
                fresh=trace['trace_id'] not in existing_ids
                tracker.record_evidence(trace,step=index,upload=fresh)
                event=trace['采用决定']
                metrics={'round':index,'accepted':event['accepted'],'research_requests':event['calls'],
                         'new_training_workers':sum(r['本次启动训练'] for r in trace['训练和复用']),
                         'reused_candidates':sum(r['复用已完成候选'] for r in trace['训练和复用']),
                         'service_score':trace['实际交付']['本轮分数']['score']}
                if fresh:
                    tracker.log(metrics,index)
                    uploaded+=1
                else:tracker.latest.update({k:float(v) for k,v in metrics.items()})
                if evidence_hook is not None:evidence_hook(index)
            runtime_success=report['driver_completed_stages']==len(traces)
            tracker.log({'complete/baseline_score':report['scores']['baseline']['score'],
                         'complete/actual_score':report['scores']['actual']['score'],
                         'complete/workflow_succeeded':int(runtime_success)},len(traces)+1)
            cloud=tracker.finish(runtime_success)
            result={'binding_sha256':identity,'cloud_run_id':tracker.id,'url':tracker.url,
                    'cloud_verified':cloud['cloud_verified'],'evidence_cloud_verified':cloud['evidence_cloud_verified'],
                    'evidence_count':len(traces),'workflow_succeeded':runtime_success,
                    'uploaded_now':uploaded,'reused_cloud_stages':len(existing_ids),
                    'reused_completed_recording':False,'new_training_calls':0,'attempts':attempt_no,
                    'verified_at':datetime.now(timezone.utc).isoformat()}
            attempt.update(status='verified' if cloud['cloud_verified'] else 'verification_pending',result=result)
            atomic_json(attempt_path,attempt)
            if cloud['cloud_verified']:atomic_json(completed,result)
            return result
        except BaseException as exc:
            attempt.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error_type=type(exc).__name__)
            # 已发送的部分也按原身份封存，重试时下载已存在正文并仅补缺项。
            if tracker is not None:
                try:
                    partial=tracker.finish(False)
                    attempt['partial_cloud_verified']=partial['cloud_verified']
                    attempt['partial_evidence_count']=len(tracker.evidence)
                except BaseException as finishing:
                    attempt['finish_error_type']=type(finishing).__name__
            atomic_json(attempt_path,attempt)
            raise
