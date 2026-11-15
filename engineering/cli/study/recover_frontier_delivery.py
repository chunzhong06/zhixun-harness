"""恢复已完成研究的最终交付：只接受原模型响应的机械 JSON 封装修复。"""
from argparse import ArgumentParser
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.workflow import frontier_comparison as workflow
from harness.runtime.storage import atomic_json


async def run(args):
    previous=args.previous.resolve();output=args.output.resolve()
    identity=json.loads((previous/'input-identity.json').read_bytes())
    if identity['arm']!='project':raise ValueError('只恢复已有自主控制器的封存finish动作')
    paths=sorted((previous/'queue/responses/mle-model').glob('*.json'))
    final=paths[-1];original=final.read_bytes()
    # 只补一个缺失的最外层结束括号，不允许改动内层生成内容。
    repaired=original.rstrip()+b'}\n'
    response=json.loads(repaired)
    request=json.loads((previous/'queue/requests/mle-model'/final.name).read_bytes())
    if response['request_id']!=request['request_id']:raise ValueError('原响应身份不同')
    action=json.loads(response['plan']['text'])
    if action.get('action')!='finish':raise ValueError('原模型尚未作出结束及提名决定，不能机械恢复')
    records=json.loads((previous/'executions/index.json').read_bytes())
    for record in records:
        if sha256(Path(record['source']).read_text(encoding='utf-8').encode()).hexdigest()!=record['source_sha256']:
            raise ValueError('原执行源码变化')
        for name,checksum in record['input_sha256'].items():
            if sha256((Path(record['directory'])/'input'/name).read_bytes()).hexdigest()!=checksum:
                raise ValueError('原执行输入变化')
        for candidate in record['candidates']:
            if sha256(Path(candidate['path']).with_suffix('.pkl').read_bytes()).hexdigest()!=candidate['weights_sha256']:
                raise ValueError('原模型权重变化')
    selected=next(r for r in records if r['sequence']==action['execution'])
    if selected['returncode'] or not selected['candidates']:raise ValueError('原提名不是有效模型')
    counts=json.loads((previous/'driver-complete.json').read_bytes())

    async def frozen_finish(description,task,workspace,executor,broker,**kwargs):
        current=json.loads((output/'input-identity.json').read_bytes())
        if current!=identity:raise ValueError('恢复改变了原共同任务或预算')
        executor.records=records
        atomic_json(output/'executions/index.json',records)
        broker.calls=counts['model_calls'];broker.search_calls=counts['search_calls']
        events=json.loads((previous/'workspace/project-decisions.json').read_bytes())
        atomic_json(Path(workspace)/'project-decisions.json',events+[action])
        (output/'repaired-final-response.json').write_bytes(repaired)
        atomic_json(output/'recovery.json',{'origin':str(previous),'original_response':str(final),
            'original_sha256':sha256(original).hexdigest(),'repair':'仅补最外层JSON结束括号',
            'selected_execution':action['execution'],'new_model_calls':0,'reused_model_calls':broker.calls,
            'new_training_executions':0,'original_failure_preserved':True})
        return selected
    workflow.run_free_research=frozen_finish
    args.task=identity['task'];args.arm='project';args.calls=identity['max_model_calls']
    await workflow.run(args)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);asyncio.run(run(p.parse_args()))
