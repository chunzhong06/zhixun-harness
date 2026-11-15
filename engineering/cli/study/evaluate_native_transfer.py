"""将事前按官方公开验证选定的NPZ原样迁移；通过独立JSON入口完整预测。"""
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
from harness.data.gas_input import read_gas_batches, REGISTERED_SHA256 as GAS_SHA
from harness.data.cmapss_input import read_cmapss, test_answers
from harness.integrations.native_research import measure, exploration_diagnostics, export_release
from harness.runtime.storage import atomic_json


def input_object(paths, row):
    result = {} if paths[0][0][0] == 'key' else []
    for parts, value in zip(paths, row, strict=True):
        node = result
        for position, (kind, key) in enumerate(parts):
            final = position == len(parts) - 1
            if kind == 'index':
                while len(node) <= key: node.append(None)
                if final: node[key] = float(value)
                else:
                    if node[key] is None: node[key] = {} if parts[position + 1][0] == 'key' else []
                    node = node[key]
            else:
                if final: node[key] = float(value)
                else:
                    if key not in node: node[key] = {} if parts[position + 1][0] == 'key' else []
                    node = node[key]
    return result


def dataset(task):
    if task == 'gas':
        batches = [b for b in read_gas_batches(ROOT / 'data/public/gas-drift/gas-drift.zip', expected_sha256=GAS_SHA) if b.batch_number >= 3]
        records = [r for b in batches for r in b.records]
        groups = [str(b.batch_number) for b in batches for r in b.records]
        return np.asarray([r.values for r in records]), [r.sample_id for r in records], groups, lambda: np.asarray([r.label for r in records])
    if task == 'bearing':
        source = json.loads((ROOT / 'data/public/xjtu-sy/fifth-devices.json').read_bytes())
        rows = source['rows']; ids = [r['member'] for r in rows]
        groups = [Path(key).parent.name for key in ids]
        return np.asarray([r['features'] for r in rows]), ids, groups, lambda: np.asarray([source['ends'][group] - int(Path(key).stem) for key, group in zip(ids, groups)])
    source = ROOT / 'data/public/cmapss/CMAPSSData.zip'
    parts = {s: read_cmapss(source, subset=s, partition='test') for s in ('FD001', 'FD002', 'FD003', 'FD004')}
    tracks = [track for group in parts.values() for track in group]
    x = np.asarray([[*t.values[-1][3:], *t.values[-1][:3]] for t in tracks])
    def truth():
        answers = {k: v for name, group in parts.items() for k, v in test_answers(source, name, group).items()}
        return np.asarray([answers[t.unit_id] for t in tracks])
    return x, [t.unit_id for t in tracks], [name for name, group in parts.items() for t in group], truth


def main():
    parser = ArgumentParser()
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    chosen = json.loads(args.selection.read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for task, specification in chosen['tasks'].items():
        folder = args.output / task; folder.mkdir()
        for name, checksum in specification['inputs_sha256'].items():
            if sha256(Path(name).read_bytes()).hexdigest() != checksum:
                raise ValueError('事前冻结的来源或权重变化：' + name)
        x, ids, groups, truth = dataset(task)
        with np.load(specification['initial'], allow_pickle=False) as checkpoint:
            # 官方NPZ元数据是字符串数组，恢复原路径顺序后重建独立预测请求。
            meta = json.loads(str(checkpoint['metadata']))
        paths = [json.loads(path) for path in meta['paths']]
        requests = '\n'.join(json.dumps({'task': meta['task'], 'inputs': input_object(paths, row)}, separators=(',', ':')) for row in x)
        env = os.environ.copy(); env.pop('PYTHONPATH', None); env['PYTHONDONTWRITEBYTECODE'] = '1'
        predictions = {}
        for arm in ('initial', 'selected'):
            release = folder / arm
            export_release(Path(specification['task_dir']), Path(specification[arm]), release)
            process = subprocess.run([sys.executable, '-B', '-m', 'baselines.predict', '--checkpoint', 'checkpoint.npz'],
                cwd=release, env=env, input=requests, capture_output=True, text=True, encoding='utf-8', timeout=90)
            if process.returncode: raise RuntimeError(process.stderr)
            outputs = [json.loads(line)['outputs'] for line in process.stdout.splitlines()]
            prediction = np.asarray([[v['score'] for v in row] for row in outputs]) if task == 'gas' else np.asarray([row['value'] for row in outputs])
            if len(prediction) != len(ids): raise ValueError('实际发放数量与输入身份不同')
            predictions[arm] = prediction
        np.savez_compressed(folder / 'issued.npz', ids=np.asarray(ids), groups=np.asarray(groups), **predictions)
        labels = truth()
        scores = {arm: {'overall': measure(task, labels, prediction),
                        'by_group': {group: measure(task, labels[np.asarray(groups) == group], prediction[np.asarray(groups) == group])
                                     for group in sorted(set(groups))}}
                  for arm, prediction in predictions.items()}
        results[task] = {'rows': len(ids), 'groups': len(set(groups)), 'scores': scores,
                         'issued_sha256': sha256((folder / 'issued.npz').read_bytes()).hexdigest(),
                         'selected_sha256': specification['selected_sha256'], 'selected_name': specification['selected_name']}
        atomic_json(folder / 'scores.json', results[task])
        print(json.dumps({'task': task, **results[task]}, ensure_ascii=False), flush=True)
    atomic_json(args.output / 'summary.json', {'scope': '冻结预测权重的跨批次与跨设备迁移；不是整个Harness在新任务上的自主训练评价', 'tasks': results})


if __name__ == '__main__': main()
