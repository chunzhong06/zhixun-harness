"""官方父检查点上的真实源码研究与采用；支持Luna队列或正式PyTRiO服务。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import asyncio
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]

import numpy as np
from harness.integrations.free_research_driver import run_free_research
from harness.integrations.native_research import prepare, measure, exploration_diagnostics, export_release, NativeExecutor, INSTRUCTION, DEFAULT_SOURCE
from harness.models.complete_checkpoint import load_checkpoint
from harness.integrations.research_broker import AuthorBroker
from harness.integrations.research_recovery import RecoverableBroker, file_identity, verify_files, freeze_invocation
from harness.models.official_rul_cpu import _trainer
from harness.models.continuation_recipes import reference_programs
from harness.runtime.storage import atomic_json
from harness.integrations.native_trace import save_stage_trace, emit_trace

CURRENT_MODELS = ROOT / 'engineering/configs/tasks/current-models.json'


def current_parent(task):
    """接续本项目已核验的完整模型；赛方指定父版仍由显式参数传入。"""
    selected = json.loads(CURRENT_MODELS.read_bytes())['tasks'][task]
    path = (ROOT / selected['checkpoint']).resolve()
    if not path.is_relative_to(ROOT) or sha256(path.read_bytes()).hexdigest() != selected['sha256']:
        raise ValueError('当前完整模型缺失或内容已改变')
    return path


def credential():
    """Windows读取当前用户配置，避免沿用换号前的进程环境；其他主机用运行时变量。"""
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
                return winreg.QueryValueEx(key, 'PYTRIO_API_KEY')[0]
        except FileNotFoundError:
            pass
    return os.environ.get('PYTRIO_API_KEY')


async def run(task, output, *, driver='queue', calls=4, stages=1, model='Qwen/Qwen3.8-27B',
              resume=False, data_root=ROOT / 'data/official', parent_checkpoint=None,
              broker_factory=None, stage_hook=None, candidate_cache=True):
    output = Path(output).resolve()
    if resume:
        task_dir = output / 'task'
        oracle = json.loads((output / 'oracle.json').read_bytes())
    else:
        task_dir, oracle = prepare(task, output, data_root=data_root, parent_checkpoint=parent_checkpoint)
    source_files = [Path(__file__), ROOT / 'engineering/cli/workflow/native_continuation.py']
    source_files += [ROOT / 'engineering/src/harness' / path for path in (
        'integrations/native_research.py', 'integrations/free_research_driver.py',
        'integrations/native_trace.py',
        'integrations/research_recovery.py', 'integrations/research_broker.py',
        'integrations/service_research_broker.py', 'integrations/service_transport.py',
        'integrations/service_budget.py', 'models/complete_training.py', 'models/robust_residual_network.py',
        'integrations/pytrio_planner.py', 'integrations/candidate_selection.py',
        'integrations/research_ledger.py', 'models/native_continuation.py',
        'models/continuation_recipes.py', 'models/complete_checkpoint.py', 'models/gas_views.py',
        'models/official_rul_cpu.py', 'runtime/storage.py')]
    inputs = source_files + list(task_dir.iterdir()) + [output / name for name in (
        'oracle.json', 'service-inputs.npz', 'initial-predictions.npz', 'identity.json')]
    if parent_checkpoint is not None:
        inputs.append(Path(parent_checkpoint))
    freeze_invocation(output / 'invocation.json', {'task': task, 'driver': driver, 'calls': calls,
                      'stages': stages, 'model': model, 'candidate_cache': candidate_cache, 'data_root': str(Path(data_root).resolve()),
                      'parent_checkpoint': str(Path(parent_checkpoint).resolve()) if parent_checkpoint else None}, inputs, resume=resume)
    if resume and (output / 'completion.json').exists():
        verify_files(json.loads((output / 'completion.json').read_bytes())['artifacts'])
        return json.loads((output / 'run.json').read_bytes())
    specification = json.loads((task_dir / 'task.json').read_bytes())
    parent = task_dir / 'initial.npz'
    with np.load(output / 'initial-predictions.npz', allow_pickle=False) as arrays:
        current = {name: arrays[name].copy() for name in arrays.files}
    with np.load(output / 'service-inputs.npz', allow_pickle=False) as arrays:
        service_x = arrays['x'].copy()
    baseline_service = current['serve'].copy()
    references = reference_programs()
    if task == 'engine':
        references['源工况完整父版续训'] = 'def train(session):\n    session.build_regression_components((21,22,23),(0,2,0),epochs=60,head_epochs=12)\n'
    rounds = []
    issued = []
    indices = np.array_split(np.arange(len(service_x)), stages)
    for stage in range(stages):
        folder = output / f'stage-{stage + 1}'
        if resume and (folder / 'completed.json').exists():
            saved = json.loads((folder / 'completed.json').read_bytes())
            verify_files(saved['artifacts'])
            if saved['parent_sha256'] != sha256(parent.read_bytes()).hexdigest():
                raise ValueError('恢复阶段接续了不同父版')
            event = json.loads((folder / 'decision.json').read_bytes())
            parent = Path(event['published_checkpoint'])
            if event['accepted']:
                with np.load(parent.parent / 'predictions.npz', allow_pickle=False) as arrays:
                    current['explore'], current['adopt'] = arrays['explore'].copy(), arrays['confirm'].copy()
            with np.load(folder / 'issued.npz', allow_pickle=False) as arrays:
                if not np.array_equal(arrays['indices'], indices[stage]):
                    raise ValueError('恢复服务顺序变化')
                issued.append(arrays['prediction'].copy())
            rounds.append(event)
            if (folder / 'trace.json').exists():
                emit_trace(json.loads((folder / 'trace.json').read_bytes()))
            continue
        stage_parent_sha = sha256(parent.read_bytes()).hexdigest()
        executor = NativeExecutor(folder / 'executions', task_dir, parent, oracle, max_epochs=60, timeout=240, resume=resume,
                                  cache_dir=output / 'candidate-cache' if candidate_cache else None,
                                  cacheable_sources=references.values())
        executor.seed_parent(current['explore'])
        if broker_factory is not None:
            broker = broker_factory(folder, calls)
        elif driver == 'queue':
            broker = AuthorBroker(folder / 'queue', max_calls=calls, timeout=200)
        else:
            from harness.integrations.pytrio_planner import PyTrioConfig
            from harness.integrations.service_transport import HttpChatTransport
            from harness.integrations.service_research_broker import ServiceResearchBroker
            config = PyTrioConfig(model=model, max_tokens=8192, timeout_seconds=600, raw_completion=True)
            broker = ServiceResearchBroker(folder / 'queue', config,
                HttpChatTransport(config, credential(), allow_network=True), max_calls=calls)
        broker = RecoverableBroker(broker, folder / 'exchanges')
        description = {'task': task, 'stage': stage + 1, 'parent_sha256': sha256(parent.read_bytes()).hexdigest(),
            'training_rows': specification['train_rows'], 'max_training_epochs': 60,
            'objective': '保持真实续训与可独立交付，降低合法探索损失；最后确认和服务答案不返回',
            'classification': task == 'gas', 'parent_exploration': measure(task, oracle['explore']['y'], current['explore']),
            'parent_exploration_diagnostics': exploration_diagnostics(task, oracle['explore']['y'], current['explore'], oracle['explore']['groups']),
            'baseline_model_source': (task_dir / 'baseline-model.py').read_text(encoding='utf-8'),
            'complete_model_source': (ROOT / 'engineering/src/harness/models/complete_checkpoint.py').read_text(encoding='utf-8'),
            'previous_decisions': [{'stage': r['stage'], 'accepted': r['accepted'], 'error': r['error'],
                'selected_source_sha256': r['selected'].get('source_sha256'),
                'selected_exploration_loss': r['selected']['score'],
                'published_sha256': r['published_sha256']} for r in rounds],
            'repeated_reference_policy': '相同父版与资料上的已登记确定性参考程序会复用原候选，明确不计作新训练；自由源码仍执行。',
            'reference_sources': references}
        error = None
        try:
            selected = await run_free_research(description, task_dir, folder / 'workspace', executor, broker,
                instruction=INSTRUCTION, reference_sources=references, resume=resume, copy_inputs=False)
            candidate = selected['candidates'][0]
        except (RuntimeError, ValueError, TimeoutError) as exc:
            # 失败保留父版完整交付；已有训练记录仍可审查，不按未来得分补选模型。
            error = {'type': type(exc).__name__, 'message': str(exc)}
            candidate = executor.records[0]['candidates'][0]
        selected_parent = Path(candidate['checkpoint']) == parent
        accepted = False
        confirmation = None
        if not selected_parent:
            selected_path = Path(candidate['checkpoint'])
            with np.load(selected_path.parent / 'predictions.npz', allow_pickle=False) as arrays:
                new_explore, new_confirm = arrays['explore'].copy(), arrays['confirm'].copy()
            confirmation = {'parent': measure(task, oracle['adopt']['y'], current['adopt']),
                            'candidate': measure(task, oracle['adopt']['y'], new_confirm)}
            accepted = (candidate['score'] < measure(task, oracle['explore']['y'], current['explore'])['loss']
                        and confirmation['candidate']['loss'] < confirmation['parent']['loss'])
            if accepted:
                parent = selected_path
                current['explore'], current['adopt'] = new_explore, new_confirm
        with ZipFile(specification['archive']) as z, TemporaryDirectory(dir=folder, prefix='temporary-') as temp:
            with _trainer(z, Path(temp), specification['profile']) as trainer:
                actual_model = load_checkpoint(parent, trainer.Model)
                prediction = np.asarray([actual_model.predict(x[None, :])[0] for x in service_x[indices[stage]]])
        issued.append(prediction)
        event = {'stage': stage + 1, 'driver': driver,
            'model': model if driver == 'service' else ('gpt-6-luna' if driver == 'queue' else 'engineering-fixture'),
            'selected': candidate, 'accepted': bool(accepted), 'confirmation': confirmation, 'error': error,
            'published_checkpoint': str(parent), 'published_sha256': sha256(parent.read_bytes()).hexdigest(),
            'calls': broker.calls, 'usage': getattr(broker, 'usage', None), 'rows': len(prediction)}
        rounds.append(event)
        np.savez_compressed(folder / 'issued.npz', indices=indices[stage], prediction=prediction)
        atomic_json(folder / 'decision.json', event)
        trace = save_stage_trace(folder, task=task, parent_sha256=stage_parent_sha, event=event,
                    executions=executor.records, sample_ids=[oracle['serve']['ids'][i] for i in indices[stage]],
                    prediction=prediction, baseline=baseline_service[indices[stage]],
                    service_score=measure(task, np.asarray(oracle['serve']['y'])[indices[stage]], prediction))
        stage_artifacts = [folder / 'decision.json', folder / 'issued.npz', folder / 'trace.json', parent]
        if event['accepted']:
            stage_artifacts.append(parent.parent / 'predictions.npz')
        atomic_json(folder / 'completed.json', {'parent_sha256': stage_parent_sha,
                    'artifacts': file_identity(stage_artifacts)})
        atomic_json(output / 'run.json', {'task': task, 'stages': rounds, 'status': 'running'})
        if stage_hook is not None:
            stage_hook(event)
        emit_trace(trace)
        print(json.dumps({'task': task, 'stage': stage + 1, 'accepted': bool(accepted), 'calls': broker.calls,
                          'error': error, 'published_sha256': event['published_sha256']}, ensure_ascii=False), flush=True)
    actual = np.concatenate(issued)
    scores = {'baseline': measure(task, oracle['serve']['y'], baseline_service),
              'actual': measure(task, oracle['serve']['y'], actual)}
    np.savez_compressed(output / 'issued.npz', actual=actual, baseline=baseline_service)
    report = {'task': task, 'scope': '官方源训练权限下接续实际父模型，按完整预测链进行探索、确认和服务',
              'stages': rounds, 'scores': scores, 'status': 'completed',
              'final_checkpoint': str(parent), 'driver_completed_stages': sum(r['error'] is None for r in rounds)}
    report['release'] = export_release(task_dir, parent, output / 'release')
    atomic_json(output / 'run.json', report)
    completion_files = [output / 'run.json', output / 'issued.npz']
    completion_files += [p for p in (output / 'release').rglob('*') if p.is_file()]
    completion_files += [p for stage_dir in output.glob('stage-*') for p in stage_dir.rglob('*') if p.is_file()]
    completion_files += [p for p in (output / 'candidate-cache').glob('*.json')]
    atomic_json(output / 'completion.json', {'artifacts': file_identity(completion_files)})
    print(json.dumps({'task': task, 'scores': scores, 'driver_completed_stages': report['driver_completed_stages']}, ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['gas', 'bearing', 'engine'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--driver', choices=['queue', 'service'], default='queue')
    parser.add_argument('--calls', type=int, default=4)
    parser.add_argument('--stages', type=int, default=1)
    parser.add_argument('--model', default='Qwen/Qwen3.8-27B')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/official')
    parent_group = parser.add_mutually_exclusive_group()
    parent_group.add_argument('--parent-checkpoint', type=Path, help='本轮实际父检查点，平台指定父版优先使用此项')
    parent_group.add_argument('--current', action='store_true', help='接续本项目当前已核验的完整模型')
    parser.add_argument('--no-candidate-cache', action='store_true', help='每次参考动作均启动训练，不复用相同确定性候选')
    parser.add_argument('--recording', choices=['local','online'], default='local', help='online在运行结束后直接上传并核验逐轮过程')
    parser.add_argument('--recording-settings',type=Path,default=ROOT/'engineering/configs/tracking/swanlab.json')
    args = parser.parse_args(argv)
    if not 1 <= args.calls <= 16 or not 1 <= args.stages <= 10:
        raise ValueError('每阶段调用为1至16次，阶段数为1至10')
    result = asyncio.run(run(args.task, args.output, driver=args.driver, calls=args.calls,
                             stages=args.stages, model=args.model, resume=args.resume,
                             data_root=args.data_root, parent_checkpoint=current_parent(args.task) if args.current else args.parent_checkpoint,
                             candidate_cache=not args.no_candidate_cache))
    if args.recording=='online':
        from harness.integrations.native_recording import publish_completed_run
        recording=publish_completed_run(args.output,settings_path=args.recording_settings,allow_network=True)
        print(json.dumps({'recording':recording},ensure_ascii=False),flush=True)
        if not recording['cloud_verified']:return 3
    return 0 if result['driver_completed_stages'] == args.stages else 2


if __name__ == '__main__':
    raise SystemExit(main())
