"""执行一份续训源码，实际恢复父NPZ，输出独立可预测的官方格式模型。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import sys
import time

from harness.adapters.datasets.baseline_sources import source_package, trainer_for

import numpy as np
from harness.learning.training.native_continuation import ContinuationSession, run_program, tensor_identity
from harness.learning.inference.complete_checkpoint import load_checkpoint, prediction_sources
from harness.learning.models.official_rul_cpu import _trainer
from harness.adapters.storage.local import atomic_json


def evaluate_inputs(predict, rows, selection_policy):
    """比赛公开评价有两个输出角色；只有同一份有序输入才共享一次预测。"""
    if selection_policy != 'public-validation' or set(rows.files) != {'explore', 'confirm'}:
        return {name: predict(rows[name]) for name in rows.files}, False
    explore, confirm = rows['explore'], rows['confirm']
    identical = (explore.shape == confirm.shape and explore.dtype == confirm.dtype
                 and explore.flags.c_contiguous and confirm.flags.c_contiguous
                 and np.array_equal(explore.view(np.uint8), confirm.view(np.uint8)))
    first = predict(explore)
    return {'explore': first, 'confirm': first if identical else predict(confirm)}, identical


def execute(task_dir, parent, program, output, *, expected_parent_sha256, max_epochs=30, stage_id=None,
            teacher_cache=None):
    task_dir, parent, program, output = map(lambda p: Path(p).resolve(), (task_dir, parent, program, output))
    if output.exists():
        raise ValueError('候选输出必须使用新目录，恢复应复用已核验的原产物')
    if type(max_epochs) is not int or not 1 <= max_epochs <= 10000:
        raise ValueError('训练轮数上限必须在1至10000之间')
    task = json.loads((task_dir / 'task.json').read_bytes())
    parent_raw = parent.read_bytes()
    if sha256(parent_raw).hexdigest() != expected_parent_sha256:
        raise ValueError('实际父权重与请求指定的版本不同')
    for name, expected in task['inputs_sha256'].items():
        if sha256((task_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError('冻结训练或预测输入已改变：' + name)
    source = program.read_text(encoding='utf-8')
    if task['profile'] == 'C':
        from .engine_epoch_budget import check_engine_epochs
        check_engine_epochs(source, max_epochs)
    output.mkdir(parents=True)
    started = time.monotonic()
    cpu_started = time.process_time()
    with TemporaryDirectory(dir=output, prefix='temporary-') as temporary, source_package(task) as package:
        with trainer_for(package, Path(temporary), task['profile']) as trainer:
            model = load_checkpoint(parent, trainer.Model)
            cache = Path(teacher_cache) / expected_parent_sha256 if teacher_cache is not None else None
            if model.meta['profile'] != task['profile'] or model.meta['paths'] != task['feature_paths']:
                raise ValueError('父模型的任务或输入结构与本次任务不符')
            with np.load(task_dir / 'train.npz', allow_pickle=False) as rows:
                session = ContinuationSession(model, rows['x'], rows['y'], max_epochs=max_epochs,
                                              groups=rows['groups'] if 'groups' in rows.files else None,
                                              tabular_resource=task.get('tabular_resource'), teacher_cache=cache)
            # 加载由宿主实际完成，并逐张量核对，生成程序不能提交自报的父模型身份。
            with np.load(parent, allow_pickle=False) as stored:
                loaded = tensor_identity(model)
                for name, digest in loaded.items():
                    if sha256(stored[name].tobytes()).hexdigest() != digest:
                        raise ValueError('加载模型与指定父检查点张量不同')
            audit = run_program(source, session)
            model = session._model
            checkpoint = output / 'checkpoint.npz'
            model.save(checkpoint)
            reloaded = load_checkpoint(checkpoint, trainer.Model)
            if cache is not None and hasattr(reloaded, 'bind_round_cache'):
                reloaded.bind_round_cache(cache)
            if tensor_identity(reloaded) != audit['final_tensors'] or reloaded.meta != model.meta:
                raise ValueError('重载丢失了训练后的权重、优化器或元信息')
            from harness.learning.inference.teacher_cache import reuse_frozen_runtime
            # 教师和源上下文逐数组相同才复用；独立交付入口仍会从产物重新加载。
            teacher_reused = (reuse_frozen_runtime(reloaded, model)
                              if hasattr(reloaded, '_teacher') else False)
            evaluation_started = time.monotonic()
            with np.load(task_dir / 'evaluation-inputs.npz', allow_pickle=False) as rows:
                predict = getattr(reloaded, 'predict_batch', reloaded.predict)
                prediction, shared_evaluation = evaluate_inputs(predict, rows, task.get('selection_policy'))
            evaluation_seconds = time.monotonic() - evaluation_started
            np.savez_compressed(output / 'predictions.npz', **prediction)
            for name, payload in prediction_sources(package, checkpoint).items():
                path = output / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            (output / 'training-program.py').write_text(source, encoding='utf-8')
            report = {'format': 'harness.native-source-continuation.v1', 'task': task['task'],
                      'profile': task['profile'], 'stage_id': stage_id or task['stage_id'],
                      'parent_checkpoint_sha256': expected_parent_sha256,
                      'checkpoint_sha256': sha256(checkpoint.read_bytes()).hexdigest(),
                      'training_source_sha256': sha256(source.encode()).hexdigest(),
                      'archive_sha256': task['archive_sha256'], 'inputs_sha256': task['inputs_sha256'],
                      'independent_predict_command': 'python -B -m baselines.predict --checkpoint checkpoint.npz',
                      'training_seconds': time.monotonic() - started,
                      'worker_cpu_seconds': time.process_time() - cpu_started,
                      'evaluation_seconds': evaluation_seconds, 'shared_public_evaluation': shared_evaluation,
                      'frozen_teacher_runtime_reused': teacher_reused, **audit}
            if cache is not None and hasattr(reloaded, 'round_cache_hits'):
                report['round_teacher_cache_hits'] = {
                    'training': model.round_cache_hits['training'],
                    'evaluation': reloaded.round_cache_hits['evaluation'],
                }
                if 'device_evaluation' in reloaded.round_cache_hits:
                    report['round_teacher_cache_hits']['device_evaluation'] = reloaded.round_cache_hits['device_evaluation']
            atomic_json(output / 'training.json', report)
            with (output / 'history.jsonl').open('w', encoding='utf-8') as stream:
                for row in audit['history']:
                    stream.write(json.dumps({'stage_id': report['stage_id'], **row}, ensure_ascii=False) + '\n')
    # 所有候选文件都计入检查，不仅检查NPZ；研究队列和任务数据存放在外层。
    total = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if total > 512 * 1024**2:
        raise ValueError('本次模型及附属产物超过512 MiB')
    report['checkpoint_bytes'] = (output / 'checkpoint.npz').stat().st_size
    report['artifact_bytes'] = total
    for _ in range(5):
        atomic_json(output / 'training.json', report)
        total = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if total == report['artifact_bytes']:
            break
        report['artifact_bytes'] = total
    else:
        raise ValueError('训练产物大小记录未收敛')
    if total > 512 * 1024**2:
        raise ValueError('写入完整训练记录后产物超过512 MiB')
    return report


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', required=True, type=Path)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--expected-parent-sha256', required=True)
    parser.add_argument('--program', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--stage-id')
    parser.add_argument('--teacher-cache', type=Path)
    args = parser.parse_args(argv)
    report = execute(args.task_dir, args.parent, args.program, args.output,
                     expected_parent_sha256=args.expected_parent_sha256, max_epochs=args.max_epochs,
                     stage_id=args.stage_id, teacher_cache=args.teacher_cache)
    print(json.dumps({key: report[key] for key in ('task', 'parent_checkpoint_sha256', 'checkpoint_sha256',
        'initial_steps', 'final_steps', 'initial_epochs', 'final_epochs', 'parameter_changes', 'artifact_bytes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
