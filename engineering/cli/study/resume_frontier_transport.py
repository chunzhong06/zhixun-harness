"""恢复传输中断：重放原请求和原执行，只补封装，不重新抽样已有研究。"""
from argparse import ArgumentParser
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.workflow import frontier_comparison as official_workflow
from harness.integrations.author_mlestar import ACTOR
from harness.integrations.response_format import normalize_action_json
from harness.runtime.storage import atomic_json


async def run(args):
    previous=args.previous.resolve();output=args.output.resolve()
    official=(previous/'input-identity.json').exists()
    if official:
        workflow=official_workflow
        identity=json.loads((previous/'input-identity.json').read_bytes())
    else:
        from engineering.cli.workflow import mlestar_author as workflow
        identity=json.loads((previous/'task-identity.json').read_bytes())
        # 原站点、采集期与数据身份均从已冻结记录恢复，不接受新研究上下文。
        if sha256(args.archive.read_bytes()).hexdigest()!=identity['archive_sha256']:
            raise ValueError('恢复所用伦敦数据与原任务不同')
        if sha256(args.library.read_bytes()).hexdigest()!=identity['library_sha256']:
            raise ValueError('恢复所用既有源码与原任务不同')
    arm=identity['arm'] if official else 'project'
    counts=json.loads((previous/'driver-complete.json').read_bytes())
    repairs=[];replays=[]
    for request in sorted((previous/'queue/requests/mle-model').glob('*.json')):
        origin=previous/'queue/responses/mle-model'/request.name
        envelope=json.loads(request.read_bytes());response=json.loads(origin.read_bytes());plan=response['plan']
        if response['request_id']!=envelope['request_id']:
            # 文件名唯一绑定原请求；只修正外层抄写编号，模型文本逐字保留。
            repairs.append({'file':str(origin),'sha256':sha256(origin.read_bytes()).hexdigest(),'repair':'从同名原请求恢复外层编号','text_changed':False})
        if not official and envelope['context']['actor']!='harness_research_controller':
            raise ValueError('伦敦入口只恢复原项目组请求')
        if arm=='project':
            repaired,reason=normalize_action_json(plan['text'])
            if reason:
                repairs.append({'file':str(origin),'sha256':sha256(origin.read_bytes()).hexdigest(),'repair':'仅补动作JSON最末结束括号','text_changed':True})
                plan={**plan,'text':repaired}
        replays.append((envelope['context'],request,plan))
    records=json.loads((previous/'executions/index.json').read_bytes()) if (previous/'executions/index.json').exists() else []
    for record in records:
        assert sha256(Path(record['source']).read_text(encoding='utf-8').encode()).hexdigest()==record['source_sha256']
        for candidate in record['candidates']:
            assert sha256(Path(candidate['path']).with_suffix('.pkl').read_bytes()).hexdigest()==candidate['weights_sha256']
    OriginalBroker=workflow.AuthorBroker;OriginalExecutor=workflow.AuthorExecutor

    class ResumeBroker(OriginalBroker):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            for envelope,request,plan in replays:
                self.previous[self.semantic_key(envelope)].append((request,plan))
            atomic_json(output/'transport-recovery.json',{'origin':str(previous),'repairs':repairs,
                'prior_calls':counts['model_calls'],'prior_executions':len(records),'original_failure_preserved':True})

        async def exchange(self,kind,payload):
            if self.calls<counts['model_calls']:
                envelope={'kind':kind,'actor':ACTOR.get(),'payload':payload}
                if not self.previous[self.semantic_key(envelope)]:
                    raise ValueError('恢复请求与原请求不一致，未发出新模型请求')
            answer=await super().exchange(kind,payload)
            # 已重放的调用确实在原作业发生过，恢复后仍占用预算。
            # 逐步恢复计数，使每个重放请求的 remaining 字段与原现场完全相同。
            self.prior_model_invocations=self.reused_model_calls
            return answer

    class ResumeExecutor(OriginalExecutor):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.pending_replay=list(records)

    workflow.AuthorBroker=ResumeBroker;workflow.AuthorExecutor=ResumeExecutor
    args.arm=arm
    if official:
        if args.bundle is None:raise ValueError('官方恢复需要原任务包')
        args.task=identity['task'];args.calls=identity['max_model_calls']
    else:
        args.site=identity['site'];args.start=identity['start'];args.max_calls=48
        args.frontier_task=True;args.comparison_task=previous;args.replay_from=None
        args.prior_model_invocations=0;args.prior_searches=0;args.resume_refinement_from=None
    await workflow.run(args)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path)
    p.add_argument('--archive',type=Path,default=ROOT/'data/public/london-air-2026/stations.zip')
    p.add_argument('--library',type=Path,default=ROOT/'engineering/configs/experiments/london-complete-library.json')
    asyncio.run(run(p.parse_args()))
