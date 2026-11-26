"""一次真实研究轮次；只使用装配层注入的学习、评价和外部服务。"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Callable
import json
import numpy as np
from harness.engine.research_recovery import file_identity, verify_files
from harness.adapters.storage.local import atomic_json
from harness.engine.publication_selection import balanced_brier, protect_incumbent, select_public_validation

@dataclass
class RoundSession:
    make_executor: Callable
    make_broker: Callable
    research: Callable
    measure: Callable
    diagnose: Callable
    predict: Callable
    save_trace: Callable
    emit_trace: Callable
    model_source: str
    instruction: str
    history: list
    capabilities: dict
    max_epochs: int = 60
    extra_references: Callable | None = None
    summarize_research: Callable | None = None
    starting_reference: Callable | None = None


def public_probe_prediction(specification, task_dir, oracle, service_x, indices, current):
    """只复用平台内部同一公开集的已核预测；真实服务或独立确认继续独立推理。"""
    if (specification.get('selection_policy') != 'public-validation'
            or specification.get('service_role') != 'public_validation_probe'
            or oracle['serve'].get('y') is not None
            or oracle['serve']['ids'] != oracle['explore']['ids']):
        return None
    path = Path(task_dir) / 'evaluation-inputs.npz'
    if sha256(path.read_bytes()).hexdigest() != specification['inputs_sha256']['evaluation-inputs.npz']:
        raise ValueError('公开评价输入已变化，不能复用发放探针')
    with np.load(path, allow_pickle=False) as arrays:
        if not np.array_equal(service_x, arrays['explore']):
            return None
    return current['explore'][indices].copy()

async def run_round(session, *, task, task_dir, specification, output, folder, parent, current,
                    oracle, references, driver, model, calls, stage, indices, service_x,
                    baseline_service, resume=False, candidate_cache=True, stage_hook=None):
    stage_parent_sha = sha256(parent.read_bytes()).hexdigest()
    policy = specification.get('selection_policy', 'confirmation')
    if policy not in ('confirmation', 'public-validation'):
        raise ValueError('未知的发布评价协议')
    public_selection = policy == 'public-validation'
    if public_selection and oracle['explore'] != oracle['adopt']:
        raise ValueError('全公开验证选案需要相同的完整评价身份和答案')
    diagnostics = session.diagnose(task, oracle['explore']['y'], current['explore'], oracle['explore']['groups'])
    program_seconds=session.capabilities.get('program_seconds',90)
    references = dict(references)
    if session.extra_references is not None:
        references.update(session.extra_references(diagnostics, session.capabilities))
    executor = session.make_executor(folder / 'executions', task_dir, parent, oracle, max_epochs=session.max_epochs, timeout=program_seconds, resume=resume,
                              cache_dir=output / 'candidate-cache' if candidate_cache else None,
                              cacheable_sources=references.values())
    starting_seconds = program_seconds
    program_seconds = min(program_seconds, specification.get('extra_program_seconds', program_seconds))
    executor.seed_parent(current['explore'])
    original_candidate = executor.records[0]['candidates'][0]
    reference_candidate = None
    start = None
    deferred = None
    try_research_first = task == 'gas' and public_selection and specification.get('defer_gas_reference') is True
    def execute_start(name):
        executor.run_reference(references[name], folder, 'starting-reference.py', starting_seconds)
        record = executor.records[-1]
        candidate = None
        if record['returncode'] == 0 and record['candidates']:
            candidate = min(record['candidates'], key=lambda c: c['score'])
        return {'name': name, 'execution': record['sequence'], 'returncode': record['returncode'],
                'candidate': candidate, 'source_sha256': record['source_sha256']}

    if session.starting_reference is not None:
        name = session.starting_reference(task, references, session.capabilities)
        if name is not None:
            if name not in references:
                raise ValueError('实际起点必须来自本轮冻结参考库')
            prior = session.history[-1].get('reference_observation') if session.history else None
            if public_selection and isinstance(prior, dict) and prior.get('name') == name:
                key = executor.reference_key(references[name])
                if (key and prior.get('key') == key and prior.get('parent_loss') == original_candidate['score']
                        and isinstance(prior.get('loss'), (float, int)) and np.isfinite(prior['loss'])
                        and prior['loss'] >= original_candidate['score']):
                    deferred = prior
            if deferred or try_research_first:
                start = {'name': name, 'execution': None, 'candidate': None, 'returncode': None,
                         'source_sha256': sha256(references[name].encode()).hexdigest(), 'previous_measurement': deferred}
                if try_research_first:
                    start['defer_reason'] = 'stable_parent_try_new_experiment_first'
            else:
                start = execute_start(name)
                reference_candidate = start['candidate']
            alternate = session.capabilities.get('comparison_reference')
            if public_selection and alternate and alternate != name:
                executor.run_reference(references[alternate], folder, 'comparison-reference.py', program_seconds)
                record = executor.records[-1]
                start['comparison'] = {'name': alternate, 'execution': record['sequence'],
                                       'returncode': record['returncode'], 'source_sha256': record['source_sha256']}
    broker = session.make_broker(folder, calls)
    description = {'task': task, 'stage': stage + 1, 'parent_sha256': sha256(parent.read_bytes()).hexdigest(),
        'training_rows': specification['train_rows'], 'max_training_epochs': session.max_epochs, 'training_capabilities': session.capabilities,
        'objective': ('保持真实续训与可独立交付，降低全部公开验证损失；验证标签不进入训练，隐藏答案不可见'
                      if public_selection else '保持真实续训与可独立交付，降低合法探索损失；最后确认和服务答案不返回'),
        'selection_policy': policy,
        'data_identity':specification.get('data_identity'),
        'parent_handoff':specification.get('parent_handoff'),
        'evaluation_rows':len(oracle['explore']['y']),
        'history_score_scope':'历轮分数只有在评价数据身份相同时可直接比较；本轮父候选都重新用本轮评价资料评分。',
        'classification': task == 'gas', 'parent_exploration': session.measure(task, oracle['explore']['y'], current['explore']),
        'parent_exploration_diagnostics': diagnostics,
        'baseline_model_source': (task_dir / 'baseline-model.py').read_text(encoding='utf-8'),
        'complete_model_source': session.model_source,
        'previous_decisions': session.history,
        'starting_reference': start,
        'publication_policy': ('宿主按全部公开验证主指标从本轮真实成功候选和父版中选优，平分保持。若起点带previous_measurement，相同条件的上轮实测已落后或持平；可尝试其他改进或显式复测原程序。'
            if public_selection else '固定参考已实际执行。研究结束后先确认参考，再判断提名相对已保留版本的增量；确认和服务结果不回传。较差提名或研究失败不撤销有效参考收益。'),
        'repeated_reference_policy': '相同父版与资料上的已登记确定性参考程序会复用原候选，明确不计作新训练；自由源码仍执行。',
        'reference_sources': references}
    error = None
    if public_selection and task == 'gas':
        description['publication_policy'] += ' 气体先比较固定参考；后续候选还须使按实际类别等权的 Brier 概率误差不增加。'
    handoff = None
    research_options = {}
    if public_selection and specification.get('final_choice_at_host') is True:
        research_options['final_choice_at_host'] = True
        description['final_selection'] = '现场程序训练成功且只剩提名名额时，下一次选择在宿主测量后进行；仍有实验名额时可继续研究。'
    substantial_reference = (
        task == 'bearing' and public_selection and research_options.get('final_choice_at_host')
        and reference_candidate is not None and original_candidate['score'] > 0
        and 0 <= reference_candidate['score'] <= .98 * original_candidate['score'])
    if substantial_reference:
        # 只在同父真实训练与当轮完整公开测量已取得明显改善时提前交还宿主。
        # 微小改进、首次失败、资料改变后未重测的参考均不能触发。
        start['stop_reason'] = 'bearing_reference_rmse_gain_at_least_two_percent'
        nominated = reference_candidate
    else:
        try:
            selected = await session.research(description, task_dir, folder / 'workspace', executor, broker,
                instruction=session.instruction, reference_sources=references, resume=resume, copy_inputs=False,
                program_seconds=program_seconds, **research_options)
            handoff = selected.get('handoff') if research_options else None
            nominated = None if handoff else selected['candidates'][0]
        except (RuntimeError, ValueError, TimeoutError) as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)}
            nominated = None
    if start and (deferred or try_research_first) and not any(r.get('launched_training_worker') and not r['returncode'] and r['candidates']
                            for r in executor.records):
        # 研究直接保父或新试验失败时，本轮仍从真实父模型续训。历史分数从不作为待交付候选。
        start.update(execute_start(start['name']))
        start['fallback_reason'] = 'no_successful_new_training'
    observation = deferred
    if public_selection and start:
        for row in executor.records:
            if (row.get('cache_key') and row['source_sha256'] == start['source_sha256']
                    and row.get('launched_training_worker') and not row['returncode'] and row['candidates']):
                result = min(row['candidates'], key=lambda c: c['score'])
                observation = None
                if np.isfinite(result['score']) and result['score'] >= original_candidate['score']:
                    observation = {'name': start['name'], 'key': row['cache_key'], 'parent_loss': original_candidate['score'],
                                   'loss': result['score'], 'checkpoint_sha256': result['weights_sha256']}
    predictions = {stage_parent_sha: (current['explore'], current['adopt'])}
    def candidate_predictions(candidate):
        key = candidate['weights_sha256']
        if key not in predictions:
            if sha256(Path(candidate['checkpoint']).read_bytes()).hexdigest() != key:
                raise ValueError('待发布候选权重身份改变')
            with np.load(Path(candidate['checkpoint']).parent / 'predictions.npz', allow_pickle=False) as arrays:
                predictions[key] = arrays['explore'].copy(), arrays['confirm'].copy()
        return predictions[key]
    if public_selection:
        pool = [(('reference' if row['actor'] == 'frozen_starting_reference' else 'research'), c)
                for row in executor.records if row['returncode'] == 0 for c in row['candidates']]
        candidate, reviews = select_public_validation(original_candidate, pool,
            validation_loss=lambda c: session.measure(task, oracle['explore']['y'], candidate_predictions(c)[0])['loss'],
            probability_loss=(lambda c: balanced_brier(oracle['explore']['y'], candidate_predictions(c)[0]))
            if task == 'gas' else None)
    else:
        candidate, reviews = protect_incumbent(original_candidate, reference_candidate, nominated,
            confirmation_loss=lambda c: session.measure(task, oracle['adopt']['y'], candidate_predictions(c)[1])['loss'])
    selection = {'policy': policy, 'independent_confirmation': not public_selection,
                 'parent': session.measure(task, oracle['explore']['y'], current['explore']),
                 'candidate': session.measure(task, oracle['explore']['y'], candidate_predictions(candidate)[0])}
    accepted = candidate['weights_sha256'] != stage_parent_sha
    confirmation = None
    if accepted:
        new_explore, new_confirm = candidate_predictions(candidate)
        if not public_selection:
            confirmation = {'parent': session.measure(task, oracle['adopt']['y'], current['adopt']),
                            'candidate': session.measure(task, oracle['adopt']['y'], new_confirm)}
        parent = Path(candidate['checkpoint'])
        current['explore'], current['adopt'] = new_explore, new_confirm
    prediction = public_probe_prediction(specification, task_dir, oracle, service_x, indices[stage], current)
    prediction_source = 'same_public_validation' if prediction is not None else 'independent_prediction'
    if prediction is None:
        prediction = session.predict(parent, service_x[indices[stage]], folder)
    decisions = folder / 'workspace/project-decisions.json'
    research = (session.summarize_research(executor.records,
                json.loads(decisions.read_bytes()) if decisions.is_file() else [])
                if session.summarize_research else None)
    event = {'stage': stage + 1, 'driver': driver,
        'model': model if driver == 'service' else ('gpt-6-luna' if driver == 'queue' else 'engineering-fixture'),
        'selected': candidate, 'accepted': bool(accepted), 'confirmation': confirmation, 'error': error,
        'nomination': nominated, 'starting_reference': start, 'reference_observation': observation,
        'publication_reviews': reviews, 'selection': selection,
        'data_identity':specification.get('data_identity'),'evaluation_rows':len(oracle['explore']['y']),
        'research': research,
        'prediction_source': prediction_source,
        'published_checkpoint': str(parent), 'published_sha256': sha256(parent.read_bytes()).hexdigest(),
        'calls': broker.calls, 'usage': getattr(broker, 'usage', None), 'rows': len(prediction)}
    if handoff:
        event['research_handoff'] = handoff
    np.savez_compressed(folder / 'issued.npz', indices=indices[stage], prediction=prediction)
    atomic_json(folder / 'decision.json', event)
    trace = session.save_trace(folder, task=task, parent_sha256=stage_parent_sha, event=event,
                executions=executor.records, sample_ids=[oracle['serve']['ids'][i] for i in indices[stage]],
                prediction=prediction, baseline=baseline_service[indices[stage]],
                service_score=(session.measure(task, np.asarray(oracle['serve']['y'])[indices[stage]], prediction) if oracle['serve']['y'] is not None else None))
    stage_artifacts = [folder / 'decision.json', folder / 'issued.npz', folder / 'trace.json', parent]
    if event['accepted']:
        stage_artifacts.append(parent.parent / 'predictions.npz')
    atomic_json(folder / 'completed.json', {'parent_sha256': stage_parent_sha,
                'artifacts': file_identity(stage_artifacts)})
    if stage_hook is not None:
        stage_hook(event)
    session.emit_trace(trace)
    print(json.dumps({'task': task, 'stage': stage + 1, 'accepted': bool(accepted), 'calls': broker.calls,
                      'error': error, 'published_sha256': event['published_sha256']}, ensure_ascii=False), flush=True)

    return parent, current, event, prediction
