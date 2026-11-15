"""冻结官方三任务和已采集迁移数据；不按本次服务成绩挑选输入。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
from harness.integrations.engine_parent_view import transform
from harness.runtime.storage import atomic_json


def prepare(output):
    base = ROOT / 'results/experiments/local-defaults/whole-maintenance'
    for task in ('gas', 'bearing', 'engine'):
        source = base / ('official-research-entry-causal' if task == 'gas' else 'official-research-entry-verified') / task
        dest = output / task
        shutil.copytree(source / 'task', dest / 'task')
        for name in ('oracle.json', 'outer-inputs.npz', 'task-identity.json'):
            shutil.copy2(source / name, dest / name)
        identity = json.loads((dest / 'task-identity.json').read_bytes())
        if task == 'engine':
            binding = identity['parent_bindings'][5]
            for name in ('train.csv', 'validation.csv', 'test.csv'):
                frame = pd.read_csv(dest / 'task' / name)
                raw = frame.iloc[:, :24].to_numpy()
                native, parent = transform(binding, raw, ROOT / 'data/official')
                columns = [f'input_{i}' for i in range(24)] + [f'native_{i}' for i in range(24)] + ['parent_value']
                new = pd.DataFrame(np.column_stack((raw, native, frame['parent_value'])), columns=columns)
                if 'target' in frame: new['target'] = frame['target']
                new.to_csv(dest / 'task' / name, index=False)
            with np.load(dest / 'outer-inputs.npz', allow_pickle=False) as z:
                values = {k: z[k].copy() for k in z.files}
            raw = values['service_x'][:, :24]
            native, parent = transform(binding, raw, ROOT / 'data/official')
            if not np.allclose(parent, values['baseline'], rtol=0, atol=1e-8):
                raise ValueError('补充工况视图改变了原完整父版')
            values['service_x'] = np.column_stack((raw, native, values['baseline']))
            np.savez_compressed(dest / 'outer-inputs.npz', **values)
            metadata = json.loads((dest / 'task/task.json').read_bytes())
            metadata['feature_columns'] = columns
            atomic_json(dest / 'task/task.json', metadata)
            identity['extra_view'] = '原完整父版的冻结 native 工况刻度，模型和服务预测保持'
        identity['origin_identity_sha256'] = sha256((source / 'task-identity.json').read_bytes()).hexdigest()
        identity['task_files'] = {p.name: sha256(p.read_bytes()).hexdigest() for p in (dest / 'task').iterdir()}
        atomic_json(dest / 'task-identity.json', identity)
    print(json.dumps({'prepared': ['gas', 'bearing', 'engine'], 'test_scores_read': False}))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    prepare(parser.parse_args().output.resolve())
