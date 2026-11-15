"""生成较长的人工批次，核对连续读取、历史重发、输出确认与恢复；不训练模型。"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"), str(ROOT)]
from harness.contracts import TaskSpec,Sample,BatchContext,EngineConfig,digest
from harness.runtime.engine import Harness
from harness.testing.fixtures import FixtureModelTools,FixturePlanner
from harness.runtime.handoff import run_stream,LocalPackageEndpoint
from harness.runtime.storage import LocalStore,atomic_json
from harness.evaluation.reference_evaluation import evaluate_references


def make_scenario(count=32):
    """每四批切换预写优势模型；后续标签、修订和全量历史重发均显式构造。"""
    if type(count) is not int or not 8<=count<=128:
        raise ValueError('工程批次数须在8至128之间')
    task=TaskSpec('continuity-fixture',('x','window'),(0,1))
    batches=[];history={};predictions={k:{} for k in ('base','even','odd')};build_outputs={}
    for ordinal in range(1,count+1):
        bid=f'b{ordinal:03d}';kind='even' if ((ordinal-1)//4)%2==0 else 'odd'
        # 上批服务输入在本批才获得答案；每七批另修订一个更早答案。
        for sid,sample in list(history.items()):
            if sample.role=='serve' and sample.available_at==ordinal-1:
                history[sid]=replace(sample,role='fit',revision=2,label=int(sample.values[0]),available_at=ordinal)
        if ordinal>=7 and ordinal%7==0:
            sid=f'b{ordinal-3:03d}-serve-0';prior=history[sid]
            history[sid]=replace(prior,revision=prior.revision+1,label=1-prior.label,available_at=ordinal)
        for role,size in (('fit',4),('explore',2),('adopt',2),('serve',2)):
            for i in range(size):
                sid=f'{bid}-{role}-{i}';label=i%2
                history[sid]=Sample(sid,1,(float(label),float(ordinal)),None if role=='serve' else label,
                    role,'condition-'+kind,ordinal,unit_id=f'{bid}-{role}-device-{i//2}')
                predictions['base'][sid]=0
                predictions[kind][sid]=label
                predictions['odd' if kind=='even' else 'even'][sid]=1-label
        batches.append(BatchContext(task.task_id,bid,ordinal,tuple(history.values())))
        build_outputs[bid]={recipe:kind for recipe in ('recent','uniform','coverage')}
    reference={'schema_version':1,'task_id':task.task_id,'sets':[{'set_id':'original-condition','anchor_batch':'b001',
        'source':'工程预写旧场景，始终由评价侧持有','samples':[{'sample_id':f'ref-{i}','values':[float(i),0.],
            'label':i,'group':'old-condition','unit_id':'reference-device'} for i in (0,1)]}]}
    for i in (0,1):
        predictions['base'][f'ref-{i}']=0;predictions['even'][f'ref-{i}']=i;predictions['odd'][f'ref-{i}']=1-i
    return task,tuple(batches),{'evidence_kind':'interface_fixture','predictions':predictions,'build_outputs':build_outputs},reference


def write_package(task,batches,folder):
    """生成同一公共读取器可用的JSONL批次；清单仅包含运行资料。"""
    folder.mkdir(parents=True,exist_ok=True)
    columns={k:k for k in ('sample_id','revision','label','role','group','available_at','unit_id')}
    columns['features']=['x','window']
    entries=[]
    for batch in batches:
        lines=[]
        for row in batch.samples:
            record=asdict(row);record.pop('time_start');record.pop('time_end');values=record.pop('values')
            record.update(x=values[0],window=values[1]);lines.append(json.dumps(record,ensure_ascii=False))
        name=batch.batch_id+'.jsonl';(folder/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')
        entries.append({'batch_id':batch.batch_id,'ordinal':batch.ordinal,'path':name,'format':'jsonl'})
    manifest={'schema_version':1,'kind':'harness_table_manifest','origin':'interface_fixture','task':asdict(task),'columns':columns,
        'role_values':{k:k for k in ('fit','explore','adopt','serve')},'label_values':{'0':0,'1':1},'batches':entries}
    atomic_json(folder/'manifest.json',manifest)
    return folder/'manifest.json'


def semantic_state(state):
    """比较业务状态；墙钟、耗时等每次执行不同的测量留在原始记录中。"""
    keys=('samples','models','current','archive','completed','decisions','predictions','feedback','ledger',
          'used_adoption_ids','methods','last_ordinal','pending')
    return {k:state.get(k) for k in keys}


def check_continuity(folder,count=32,policies=('plan_once','plan_feedback','plan_feedback_reuse')):
    folder=Path(folder);task,batches,fixture,reference=make_scenario(count)
    manifest=write_package(task,batches,folder/'input')
    summaries=[]
    for policy in policies:
        config=EngineConfig(policy=policy,component_actions=True,active_inspection=True,method_memory=True,
                            adaptive_loop=True,demo_budget_per_batch=80)
        normal=folder/policy/'continuous';resumed=folder/policy/'resumed'
        start=time.perf_counter()
        direct=Harness(task,config,FixtureModelTools(fixture),FixturePlanner(),LocalStore(normal))
        expected,_=run_stream(direct,LocalPackageEndpoint(manifest,normal/'accepted'),normal/'handoff')
        faults={'committed':False,'acknowledgement':False};tools=FixtureModelTools(fixture)
        target_commit=f'b{count//3:03d}';target_ack=f'b{2*count//3:03d}'
        def after_event(event):
            if not faults['committed'] and event['kind']=='batch_committed' and event['batch_id']==target_commit:
                faults['committed']=True;raise InterruptedError('批次提交后工程中断')
        def after_submit(receipt):
            accepted=json.loads((resumed/'accepted'/(receipt['request_id']+'.json')).read_text(encoding='utf-8'))
            if not faults['acknowledgement'] and accepted['payload']['batch_id']==target_ack:
                faults['acknowledgement']=True;raise InterruptedError('平台接收后丢失确认')
        # 每次恢复重新建立控制器与读取器，从落盘游标继续。
        for attempt in range(3):
            engine=Harness(task,config,tools,FixturePlanner(),LocalStore(resumed),after_event=after_event)
            try:
                actual,handoff=run_stream(engine,LocalPackageEndpoint(manifest,resumed/'accepted'),resumed/'handoff',after_submit=after_submit)
                break
            except InterruptedError:
                if attempt==2:raise
        if not all(faults.values()) or semantic_state(actual)!=semantic_state(expected):
            raise AssertionError('中断恢复与连续运行的业务状态不一致')
        builds,predicts=len(tools.build_calls),len(tools.predict_calls)
        run_stream(engine,LocalPackageEndpoint(manifest,resumed/'accepted'),resumed/'handoff')
        if (builds,predicts)!=(len(tools.build_calls),len(tools.predict_calls)):
            raise AssertionError('已完成运行重放产生了额外模型步骤')
        report=evaluate_references(task,actual,reference,FixtureModelTools(fixture),resumed/'reference-evaluation')
        accepted=list((resumed/'accepted').glob('*.json'))
        if len(accepted)!=count or len(handoff['accepted'])!=count:raise AssertionError('输出确认重复或丢失')
        summaries.append({'policy':policy,'batches':count,'semantic_state_equal':True,'faults_exercised':faults,
            'accepted_outputs':len(accepted),'resume_builds':builds,'uninterrupted_builds':len(direct.tools.build_calls),
            'resume_predicts':predicts,'uninterrupted_predicts':len(direct.tools.predict_calls),
            'no_extra_steps_on_completed_replay':True,'logical_samples':len(actual['samples']),
            'input_rows_with_replay':sum(len(b.samples) for b in batches),'state_bytes':(resumed/'state.json').stat().st_size,
            'seconds_for_pair':time.perf_counter()-start,'reference_status':report['sets'][0]['status']})
    result={'scope':'synthetic_continuity_engineering','model_training':False,'real_model_called':False,
            'input_digest':digest([asdict(b) for b in batches]),'policies':summaries}
    atomic_json(folder/'continuity.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--batches',type=int,default=32)
    args=parser.parse_args();result=check_continuity(args.output,args.batches)
    print(f"三种运行模式各完成{args.batches}批；连续运行与两处中断恢复的业务状态一致。")
