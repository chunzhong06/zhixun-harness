"""发布本轮真实实验记录，区分尝试训练、训练成功和最终采用。"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from harness.adapters.storage.local import atomic_json
from harness.evaluation.native import measure
from harness.contest.lineage import checkpoint_state, parent_handoff, previous_score_context


def _metadata(path):
    with np.load(path, allow_pickle=False) as arrays:
        return json.loads(str(arrays['metadata']))


def observed_trial_feedback(report, selected):
    """给下一次研究保留已执行提案的结果，含平分保父和训练失败。"""
    trials=[]
    for worker in report.get('experiments',[]):
        if not worker['launched_training_worker'] or not worker['hypotheses']:
            continue
        trial={'program_sha256':worker['source_sha256'],'hypothesis':worker['hypotheses'][-1][:240],
               'seconds':round(worker['seconds'],2),'returncode':worker['returncode']}
        trials.append(trial)
        if worker['returncode'] or not worker['candidates']:
            trial['error']=str(worker.get('error') or '没有保存候选')[:240]
            continue
        # 内部测量最优与宿主实际采用分别记录；同一个实验可以保存多个候选。
        best=min(worker['candidates'],key=lambda candidate:candidate['exploration']['loss'])
        training=best['training_audit']['history']
        trial.update(metric=best['exploration']['metric'],score=best['exploration']['score'],
                     candidate_count=len(worker['candidates']),epochs=len(training),
                     best_matches_selected_weights=best['checkpoint_sha256']==selected['checkpoint_sha256'])
        rates=[step['learning_rate'] for step in training if step.get('learning_rate') is not None]
        if rates:
            trial['learning_rate_first_last']=[rates[0],rates[-1]]
    if not trials:
        return None
    return {'format':'harness.observed-trials.v1','parent_sha256':report['parent_checkpoint_sha256'],
            'selected_artifact_id':selected['artifact_id'],
            'parent_public_metrics':report.get('validation_before'),'executed_proposals':len(trials),
            'recent_trials':trials[-3:]}


def carry_recent_trial_feedback(compact, events, report, selected):
    """已发布模型历史全部保留，只有最近两轮携带逐项试验反馈。旧状态可继续使用。"""
    if len(events)>1:
        previous=events[-2].get('research')
        if isinstance(previous,dict) and previous.get('format')=='harness.observed-trials.v1':
            compact[-2]['research']=previous
    observed=observed_trial_feedback(report,selected)
    if observed:
        compact[-1]['research']=observed


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
    if event.get('research_handoff'):
        report['research_handoff'] = event['research_handoff']
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
        if total > 512*1024**2:
            raise ValueError('平台完整交付超过512 MiB')
        if manifest['artifact_bytes'] == total:
            return
        manifest['artifact_bytes'] = total
        atomic_json(path, manifest)
    raise ValueError('平台交付大小记录未收敛')
