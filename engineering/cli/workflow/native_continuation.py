"""执行一份续训源码，实际恢复父NPZ，输出独立可预测的官方格式模型。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'engineering/src'))

import numpy as np
from harness.models.native_continuation import ContinuationSession, run_program, tensor_identity
from harness.models.complete_checkpoint import load_checkpoint, prediction_sources
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json


def execute(task_dir, parent, program, output, *, expected_parent_sha256, max_epochs=30, stage_id=None):
    task_dir, parent, program, output = map(lambda p: Path(p).resolve(), (task_dir, parent, program, output))
    if output.exists():
        raise ValueError('候选输出必须使用新目录，恢复应复用已核验的原产物')
    if type(max_epochs) is not int or not 1 <= max_epochs <= 10000:
        raise ValueError('训练轮数上限必须在1至10000之间')
    task = json.loads((task_dir / 'task.json').read_bytes())
    archive = Path(task['archive'])
    if sha256(archive.read_bytes()).hexdigest() != task['archive_sha256']:
        raise ValueError('官方原包已改变')
    parent_raw = parent.read_bytes()
    if sha256(parent_raw).hexdigest() != expected_parent_sha256:
        raise ValueError('实际父权重与请求指定的版本不同')
    for name, expected in task['inputs_sha256'].items():
        if sha256((task_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError('冻结训练或预测输入已改变：' + name)
    source = program.read_text(encoding='utf-8')
    output.mkdir(parents=True)
    started = time.monotonic()
    with TemporaryDirectory(dir=output, prefix='temporary-') as temporary, ZipFile(archive) as package:
        with _trainer(package, Path(temporary), task['profile']) as trainer:
            model = load_checkpoint(parent, trainer.Model)
            if model.meta['profile'] != task['profile'] or model.meta['paths'] != task['feature_paths']:
                raise ValueError('父模型的任务或输入结构与本次任务不符')
            with np.load(task_dir / 'train.npz', allow_pickle=False) as rows:
                session = ContinuationSession(model, rows['x'], rows['y'], max_epochs=max_epochs,
                                              groups=rows['groups'] if 'groups' in rows.files else None)
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
            if tensor_identity(reloaded) != audit['final_tensors'] or reloaded.meta != model.meta:
                raise ValueError('重载丢失了训练后的权重、优化器或元信息')
            with np.load(task_dir / 'evaluation-inputs.npz', allow_pickle=False) as rows:
                prediction = {name: reloaded.predict(rows[name]) for name in rows.files}
            np.savez_compressed(output / 'predictions.npz', **prediction)
            for name, payload in prediction_sources(package, checkpoint).items():
                path = output / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(payload)
            (output / 'training-program.py').write_text(source, encoding='utf-8')
            report = {'format': 'harness.native-source-continuation.v1', 'task': task['task'],
                      'profile': task['profile'], 'stage_id': stage_id or task['stage_id'],
                      'parent_checkpoint_sha256': expected_parent_sha256,
                      'checkpoint_sha256': sha256(checkpoint.read_bytes()).hexdigest(),
                      'training_source_sha256': sha256(source.encode()).hexdigest(),
                      'archive_sha256': task['archive_sha256'], 'inputs_sha256': task['inputs_sha256'],
                      'independent_predict_command': 'python -B -m baselines.predict --checkpoint checkpoint.npz',
                      'training_seconds': time.monotonic() - started, **audit}
            atomic_json(output / 'training.json', report)
            with (output / 'history.jsonl').open('w', encoding='utf-8') as stream:
                for row in audit['history']:
                    stream.write(json.dumps({'stage_id': report['stage_id'], **row}, ensure_ascii=False) + '\n')
    # 所有候选文件都计入检查，不仅检查NPZ；研究队列和任务数据存放在外层。
    total = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if total > 64 * 1024**2:
        raise ValueError('本次模型及附属产物超过64 MiB')
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
    if total > 64 * 1024**2:
        raise ValueError('写入完整训练记录后产物超过64 MiB')
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
    args = parser.parse_args(argv)
    report = execute(args.task_dir, args.parent, args.program, args.output,
                     expected_parent_sha256=args.expected_parent_sha256, max_epochs=args.max_epochs, stage_id=args.stage_id)
    print(json.dumps({key: report[key] for key in ('task', 'parent_checkpoint_sha256', 'checkpoint_sha256',
        'initial_steps', 'final_steps', 'initial_epochs', 'final_epochs', 'parameter_changes', 'artifact_bytes')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
