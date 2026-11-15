"""把实际续训、采用和交付编成可云端读回的证据；复用不计作新训练。"""
from hashlib import sha256
import json
from pathlib import Path

from harness.runtime.storage import atomic_json


def save_stage_trace(folder, *, task, parent_sha256, event, executions, sample_ids, prediction, baseline, service_score):
    folder = Path(folder)
    trials = []
    for record in executions:
        if record['sequence'] == 0:
            continue
        trial = {'执行号': record['sequence'], '源码': Path(record['source']).read_text(encoding='utf-8'),
                 '源码SHA256': record['source_sha256'], '父模型SHA256': record['parent_sha256'],
                 '复用已完成候选': record.get('execution_reused', False),
                 '本次启动训练': record.get('launched_training_worker', True),
                 '来源回执': record.get('reused_from'), '本次执行秒数': record['execution_time'],
                 '返回码': record['returncode'], '错误': record['stderr'], '候选': record['candidates']}
        trial['实际训练记录'] = [json.loads(Path(c['report']).read_bytes()) for c in record['candidates']]
        trials.append(trial)
    exchanges = [json.loads(p.read_bytes()) for p in sorted((folder / 'exchanges').glob('*.json'))]
    trace = {'trace_id': sha256(str(folder.resolve()).encode()).hexdigest(), 'schema_version': 1,
             '运行目录': str(folder.parent.resolve()), '任务': task, '轮次': event['stage'],
             '驱动类型': event['driver'], '驱动模型': event['model'],
             '进入本轮父模型SHA256': parent_sha256, '研究调用': exchanges,
             '训练和复用': trials, '采用决定': event,
             '实际交付': {'样本ID': list(sample_ids), '预测': prediction.tolist(), '原始基线预测': baseline.tolist(),
                          '本轮分数': service_score, '模型SHA256': event['published_sha256'],
                          '预测文件SHA256': sha256((folder / 'issued.npz').read_bytes()).hexdigest()}}
    atomic_json(folder / 'trace.json', trace)
    return trace


def emit_trace(trace):
    trials, event = trace['训练和复用'], trace['采用决定']
    metrics = {'round': trace['轮次'], 'accepted': event['accepted'], 'research_requests': event['calls'],
               'new_training_workers': sum(t['本次启动训练'] for t in trials),
               'reused_candidates': sum(t['复用已完成候选'] for t in trials),
               'execution_seconds': sum(t['本次执行秒数'] for t in trials),
               'service_score': trace['实际交付']['本轮分数']['score']}
    print(json.dumps({'native_trace': trace, 'native_metrics': {trace['任务']: metrics}}, ensure_ascii=False), flush=True)
