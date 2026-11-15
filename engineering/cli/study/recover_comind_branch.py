"""恢复 CoMind 最后一轮被接入错误中断的分支，沿原额度完成作者选择与交付。"""
from argparse import ArgumentParser
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'engineering/src')]
from engineering.cli.workflow import frontier_comparison as workflow
from harness.integrations.author_comind import run_comind
from harness.runtime.storage import atomic_json


def fields(text,names):
    return {n:re.findall(f'<{n}>([^<]*(?:<(?!/{n}>)[^<]*)*)</{n}>',text) for n in names}


async def run(args):
    previous=args.previous.resolve();output=args.output.resolve()
    identity=json.loads((previous/'input-identity.json').read_bytes())
    capture=json.loads((previous/'branch-recovery-capture.json').read_bytes())
    if sha256(Path(capture['namespace']).read_bytes()).hexdigest()!=capture['sha256']:
        raise ValueError('中断状态指纹变化')
    records=json.loads((previous/'executions/index.json').read_bytes())
    for record in records:
        if sha256(Path(record['source']).read_text(encoding='utf-8').encode()).hexdigest()!=record['source_sha256']:
            raise ValueError('旧源码身份变化')
        for candidate in record['candidates']:
            if sha256(Path(candidate['path']).with_suffix('.pkl').read_bytes()).hexdigest()!=candidate['weights_sha256']:
                raise ValueError('旧权重身份变化')
    request=next((previous/'queue/requests/mle-model').glob('0029-*.json'))
    response=previous/'queue/responses/mle-model'/request.name
    first=next((previous/'queue/requests/mle-model').glob('0024-*.json'))
    coordinator=next((previous/'queue/responses/mle-model').glob('0023-*.json'))
    plans=fields(json.loads(coordinator.read_bytes())['plan']['text'],['pipeline'])['pipeline']
    drafts=[fields(x,['title','description','code']) for x in plans]
    draft=next(d for d in drafts if d['title'][0].strip().lower().replace(' ','-')==capture['branch'])
    counts=json.loads((previous/'driver-complete.json').read_bytes())
    spec={**capture,'previous':str(previous),'prior_requests':[24,26],
          'blocked_messages':json.loads(request.read_bytes())['context']['payload']['request']['messages'],
          'blocked_response':json.loads(response.read_bytes()),'is_lower_better':False,
          'remaining_cells':3,'title':draft['title'][0].strip(),'description':draft['description'][0].strip(),
          'draft_code':draft['code'][0].strip(),'prior_active_seconds':max(0.,response.stat().st_mtime-first.stat().st_mtime)}

    async def resume(description,task,workspace,executor,broker,seed):
        if json.loads((output/'input-identity.json').read_bytes())!=identity:
            raise ValueError('恢复改变了原共同输入或预算')
        executor.records=records
        atomic_json(executor.folder/'index.json',records)
        broker.calls=counts['model_calls'];broker.search_calls=counts['search_calls']
        atomic_json(output/'recovery.json',{'origin':str(previous),'branch':capture['branch'],
            'original_model_calls':broker.calls,'maximum_total_model_calls':broker.max_calls,
            'original_failure_preserved':True,'old_training_replayed':False,
            'last_original_generated_cell_reused':29,'remaining_original_cells':3,
            'note':'其他分支及原提案保持；不将接入错误导致的提前终止作为论文性能结果'})
        selected=await run_comind(description,task,workspace,executor,broker,seed,branch_resume=spec)
        atomic_json(output/'recovery-usage.json',{'prior_model_calls':counts['model_calls'],
            'new_model_calls':broker.calls-counts['model_calls'],'total_model_calls':broker.calls,
            'prior_executions':len(records)-3,'total_executions':len(executor.records)})
        return selected
    workflow.run_comind=resume
    args.task=identity['task'];args.arm='comind';args.calls=identity['max_model_calls']
    await workflow.run(args)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bundle',type=Path,required=True);asyncio.run(run(p.parse_args()))
