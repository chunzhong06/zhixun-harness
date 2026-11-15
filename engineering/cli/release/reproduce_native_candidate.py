"""从原始官方父版与已封存程序重建候选；可在解压后的源码包中运行。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
from engineering.cli.workflow.native_continuation import execute
from harness.integrations.native_research import prepare
from harness.runtime.storage import atomic_json


def reproduce(task, data_root, output, manifest_path):
    manifest = json.loads(manifest_path.read_bytes())
    info = manifest['models'][task]
    for relative, expected in manifest['files'].items():
        actual = ROOT / relative
        if sha256(actual.read_bytes()).hexdigest() != expected:
            raise ValueError('交付包文件身份变化：' + relative)
    output = output.resolve()
    task_dir, unused_oracle = prepare(task, output / 'prepared', data_root=data_root)
    parent = task_dir / 'initial.npz'
    if sha256(parent.read_bytes()).hexdigest() != info['parent_sha256']:
        raise ValueError('官方父版与封存来源不同')
    program = ROOT / info['program']
    report = execute(task_dir, parent, program, output / 'reproduced',
                     expected_parent_sha256=info['parent_sha256'], max_epochs=12)
    checkpoint = output / 'reproduced/checkpoint.npz'
    reference = ROOT / info['checkpoint']
    differences = {}
    with np.load(checkpoint, allow_pickle=False) as actual, np.load(reference, allow_pickle=False) as expected:
        if set(actual.files) != set(expected.files): raise ValueError('重建检查点字段不同')
        for key in actual.files:
            if key == 'metadata':
                if json.loads(str(actual[key])) != json.loads(str(expected[key])):
                    raise ValueError('重建元信息不同')
            else:
                differences[key] = float(np.max(np.abs(actual[key] - expected[key])))
    if max(differences.values()) > 1e-8:
        raise ValueError('同配方重建权重与封存候选的数值差异超过容差')
    expected_output = json.loads((ROOT / info['example_predictions']).read_bytes())
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    proc = subprocess.run([sys.executable, '-B', 'baselines/predict.py', '--checkpoint', 'checkpoint.npz'],
        cwd=output / 'reproduced', env=env, input=(ROOT / info['example_inputs']).read_text(encoding='utf-8'),
        capture_output=True, text=True, encoding='utf-8', timeout=30)
    if proc.returncode: raise RuntimeError(proc.stderr)
    actual_output = [json.loads(line) for line in proc.stdout.splitlines()]
    # 使用JSON的正式输出字段，不把文件存在或进程退出当作预测一致。
    def values(rows):
        return np.asarray([[v['score'] for v in r['outputs']] for r in rows]) if task == 'gas' else np.asarray([r['outputs']['value'] for r in rows])
    if not np.allclose(values(actual_output), values(expected_output), atol=1e-8, rtol=0):
        raise ValueError('重建后的独立JSON预测不一致')
    result = {'task': task, 'parent_sha256': info['parent_sha256'],
        'checkpoint_sha256': sha256(checkpoint.read_bytes()).hexdigest(),
        'expected_checkpoint_sha256': info['checkpoint_sha256'],
        'byte_identical': sha256(checkpoint.read_bytes()).hexdigest() == info['checkpoint_sha256'],
        'max_tensor_difference': max(differences.values()), 'tensor_differences': differences,
        'independent_prediction_rows': len(actual_output), 'independent_prediction_equal': True,
        'final_steps': report['final_steps'], 'model_bytes': checkpoint.stat().st_size}
    atomic_json(output / 'reproduction.json', result)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('--task', choices=['gas', 'bearing', 'engine'], required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'candidate-manifest.json')
    args = parser.parse_args()
    print(json.dumps(reproduce(args.task, args.data_root, args.output, args.manifest), ensure_ascii=False))


if __name__ == '__main__': main()
