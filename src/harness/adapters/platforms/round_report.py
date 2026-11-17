"""发布本轮真实实验记录，区分尝试训练、训练成功和最终采用。"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from harness.adapters.storage.local import atomic_json
from harness.evaluation.native import measure
from harness.adapters.platforms.lineage import checkpoint_state, parent_handoff, previous_score_context


def _metadata(path):
    with np.load(path, allow_pickle=False) as arrays:
        return json.loads(str(arrays['metadata']))


def _metrics(task, labels, predictions):
    result = measure(task, labels, predictions)
    if task == 'gas':
        return {'macro_f1': result['score'],
                'accuracy': float(np.mean(predictions.argmax(1)+1 == labels))}
    return {'rmse': result['score'], 'mae': float(np.mean(np.abs(predictions-labels)))}


def write_round_report(output, work, parent, validation_path, stage_id, profile, event, trace, history):
    """计数来自候选审计与最终NPZ；退步候选的训练仍保留，父版计数不被抬高。

    比赛公开验证主指标可供本轮研究选择；confirmation研究的确认分数仅结束后汇总。
    原产物的训练依据由调用方保存为selected-training.json，不充当本轮记录。
    """
    output, work, parent = Path(output), Path(work), Path(parent)
    initial, final = _metadata(parent), _metadata(output/'checkpoint.npz')
    identity = json.loads((work/'identity.json').read_bytes())
    specification = json.loads((work/'task/task.json').read_bytes())
    expected_parent = sha256(parent.read_bytes()).hexdigest()
    published_sha = sha256((output/'checkpoint.npz').read_bytes()).hexdigest()
    if expected_parent != identity['parent_sha256'] or published_sha != event['published_sha256']:
        raise ValueError('本轮报告的实际父子模型身份不一致')
    # 调用方冻结的验证路径只在外层读取，保留官方完整公开验证口径。
    from harness.adapters.platforms.baselines.data import read_dataset
    valid = read_dataset(validation_path, final['task'], final['classes'], final['paths'])
    if valid['sha256'] != identity['validation_sha256']:
        raise ValueError('本轮报告的验证数据语义摘要改变')
    labels = valid['y']+1 if profile == 'A' else valid['y']
    task = specification['task']
    with np.load(work/'issued.npz', allow_pickle=False) as arrays:
        before, after = _metrics(task, labels, arrays['baseline']), _metrics(task, labels, arrays['actual'])
    research = event['research']
    by_execution = {row['execution']: row for row in research['experiments']}
    experiments, epochs = [], []
    for trial in trace['训练和复用']:
        sequence = trial['执行号']
        entry = {'execution': sequence, 'hypotheses': by_execution[sequence]['hypotheses'],
                 'source_sha256': trial['源码SHA256'], 'parent_checkpoint_sha256': trial['父模型SHA256'],
                 'launched_training_worker': trial['本次启动训练'], 'reused': trial['复用已完成候选'],
                 'returncode': trial['返回码'], 'error': trial['错误'], 'seconds': trial['本次执行秒数'],
                 'candidates': []}
        for candidate, audit in zip(trial['候选'], trial['实际训练记录'], strict=True):
            selected = event['accepted'] and candidate['weights_sha256'] == published_sha
            entry['candidates'].append({'candidate_id': Path(candidate['path']).stem,
                'checkpoint_sha256': candidate['weights_sha256'], 'adopted': bool(selected),
                'exploration': candidate['task_metrics'], 'training_audit': audit})
            if not trial['复用已完成候选']:
                epochs.extend({**row, 'stage_id': stage_id, 'execution': sequence,
                               'candidate_id': Path(candidate['path']).stem, 'adopted': bool(selected)}
                              for row in audit['history'])
        experiments.append(entry)
    trained = sum(len(trial['candidates']) for trial in experiments if not trial['reused'])
    status = ('adopted' if event['accepted'] else 'trained_not_adopted' if trained else
              'failed' if event['error'] or experiments else 'kept_parent')
    nomination = research['nomination'] or {}
    report = {'format': 'harness.platform-round.v2', 'stage_id': stage_id, 'profile': profile,
        'task': final['task'], 'device': 'cpu', 'status': status,
        'parent_checkpoint_sha256': expected_parent, 'checkpoint_sha256': published_sha,
        'parent_handoff': parent_handoff(expected_parent, history),
        'checkpoint_state_before': checkpoint_state(parent),
        'checkpoint_state_after': checkpoint_state(output/'checkpoint.npz'),
        'initial_steps': initial['steps'], 'final_steps': final['steps'],
        'initial_epochs': initial['epochs'], 'final_epochs': final['epochs'],
        'seed': final['seed'], 'parameter_count': final['parameter_count'],
        'train_rows': specification['train_rows'], 'validation_rows': len(labels),
        'train_sha256': identity['train_sha256'], 'validation_sha256': identity['validation_sha256'],
        'validation_before': before, 'validation_after': after,
        'validation_prediction_mode': event.get('prediction_source', 'independent_prediction'),
        'previous_round_comparison': previous_score_context(expected_parent, identity['validation_sha256'], before, history),
        'decision': 'adopt' if event['accepted'] else 'retain',
        'action': {'reason': nomination.get('reason'), 'execution': nomination.get('execution'),
                   'candidate': nomination.get('candidate')},
        'starting_reference': event.get('starting_reference'),
        'publication_reviews': event.get('publication_reviews', []),
        'selection': event.get('selection', {'policy': 'confirmation', 'independent_confirmation': True}),
        'published_source': 'reference' if any(r['role']=='reference' and r['accepted'] and
            r['candidate_sha256']==published_sha for r in event.get('publication_reviews', [])) else
            'research' if event['accepted'] else 'parent',
        'history_refs': [{'stage_id': row.get('platform_stage_id', row['stage']),
                          'published_sha256': row['published_sha256']} for row in history],
        'successful_training_candidates': trained, 'executed_parent_epochs': len(epochs),
        'execution_seconds': sum(trial['seconds'] for trial in experiments if not trial['reused']),
        'experiments': experiments, 'invalid_actions': research['invalid_actions'], 'error': event['error']}
    report['parameter_count_scope'] = 'native_parent'
    with np.load(output/'checkpoint.npz', allow_pickle=False) as arrays:
        state = json.loads(str(arrays['complete_state'])) if 'complete_state' in arrays.files else {}
    if state.get('kind') in ('tabicl-parent-combination','tabicl-classification-parent','tabpfn-classification-parent'):
        report['frozen_teacher_parameter_count'] = state['teacher_parameters']
        report['complete_parameter_count'] = final['parameter_count']+state['teacher_parameters']
    atomic_json(output/'training.json', report)
    (output/'history.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in epochs), encoding='utf-8')
    return {key: report[key] for key in ('status', 'successful_training_candidates',
                                         'executed_parent_epochs', 'validation_before', 'validation_after')}


def finalize_manifest(output):
    """平台补充的历史、候选和同步状态同样计入交付总量。"""
    output = Path(output)
    path = output/'manifest.json'
    manifest = json.loads(path.read_bytes())
    for _ in range(5):
        total = sum(file.stat().st_size for file in output.rglob('*') if file.is_file())
        if total > 64*1024**2:
            raise ValueError('平台完整交付超过64 MiB')
        if manifest['artifact_bytes'] == total:
            return
        manifest['artifact_bytes'] = total
        atomic_json(path, manifest)
    raise ValueError('平台交付大小记录未收敛')
