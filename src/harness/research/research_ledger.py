"""保留全部已执行假设与结果，避免只看前三份源码而遗忘其他尝试。"""
from hashlib import sha256
from collections import deque
from harness.research.candidate_selection import candidate_id


def history_entries(records, actions):
    """仅整理驱动已经见过的信息；不读取确认预测、服务答案或其他组记录。

    源码完整正文仍交给调用方按相关性挑选。这里保留简短假设、身份和真实
    结果；相同源码只是标注，不禁止有理由的重复实验。
    """
    hypotheses = {}
    for action in actions:
        source = action.get('python_source')
        if action.get('action') in ('experiment', 'reference') and isinstance(source, str):
            key = sha256(source.encode()).hexdigest()
            hypotheses.setdefault(key, deque()).append(action['hypothesis'])
    seen, history = {}, []
    for record in records:
        key = record['source_sha256']
        candidates = record.get('candidates', [])
        reasons = hypotheses.get(key)
        # 同一源码可被有意重跑；每次绑定其当时的理由，不能混入后续重复的解释。
        entry = {'execution': record['sequence'], 'hypotheses': [reasons.popleft()] if reasons else [],
                 'source_sha256': key, 'returncode': record['returncode'],
                 'seconds': record['execution_time'],
                 'candidates': [{'candidate_id': candidate_id(candidate), 'label': candidate['label'], 'exploration_errors': candidate['exploration_rmse'],
                                 'score': candidate['score']} for candidate in candidates]}
        if key in seen:
            entry['same_source_as'] = seen[key]
        else:
            seen[key] = record['sequence']
        history.append(entry)
    return history


def round_summary(records, actions):
    """跨轮传递已观察的假设、实际试验和提名理由，不携带确认或服务答案。"""
    entries = history_entries(records, actions)
    by_sequence = {record['sequence']: record for record in records}
    for entry in entries:
        record = by_sequence[entry['execution']]
        entry['error'] = record['stderr'][-2000:] if record['returncode'] else None
    nomination = next((action for action in reversed(actions) if action.get('action') == 'finish'), None)
    return {'experiments': [entry for entry in entries if entry['execution'] != 0],
            'invalid_actions': [{key: action.get(key) for key in
                                 ('error_code', 'error', 'finish_reason', 'model_calls')}
                                for action in actions if action.get('action') == 'invalid_response'],
            'nomination': ({key: nomination.get(key) for key in ('execution', 'candidate', 'reason')}
                           if nomination else None)}
