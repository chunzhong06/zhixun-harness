"""将现有源码研究驱动接到官方NPZ续训；探索评分与最终发布留在外层。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from harness.adapters.datasets.baseline_sources import source_package
from harness.paths import package_sources,PACKAGE_ROOT
import json
import os
import subprocess
import sys
import time

import numpy as np

from harness.core.contracts import digest
from harness.adapters.datasets.official_gas import read_official_gas
from harness.learning.models.official_gas_cpu import PACKAGE_SHA
from harness.learning.models.official_rul_cpu import _trainer
from harness.learning.inference.complete_checkpoint import load_checkpoint, prediction_sources
from harness.adapters.providers.research_broker import ACTOR
from harness.engine.research_recovery import file_identity, verify_files
from harness.adapters.storage.local import atomic_json

ROOT = __import__("harness.paths", fromlist=["workspace_root"]).workspace_root()
def prepare(task, output, *, data_root=ROOT / 'data/official', parent_checkpoint=None, validation_stride=None):
    """只按身份划分公开验证资料，训练始终使用官方训练集；不改变旧研究任务。"""
    from harness.adapters.datasets.official_layout import PACKAGES
    from harness.adapters.datasets.official_layout import partition, PROTOCOL
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if task == 'gas':
        name, checksum, reader = 'cpu-a-guide-20261110-v1.zip', PACKAGE_SHA, read_official_gas
    else:
        name, checksum, reader = PACKAGES[task]
    archive = Path(data_root) / task / name
    if sha256(archive.read_bytes()).hexdigest() != checksum:
        raise ValueError('官方训练包身份变化')
    package = reader(archive)
    if task == 'gas':
        parts = {key: [] for key in ('explore', 'adopt', 'serve')}
        for row in package.validation:
            role = ('explore', 'adopt', 'serve')[int(digest(['native-continuation-v1', row.sample_id])[:8], 16) % 3]
            parts[role].append(row)
    else:
        protocol = json.loads(PROTOCOL.read_bytes())
        if validation_stride is not None:
            if type(validation_stride) is not int or validation_stride < 1:
                raise ValueError('验证观测步长须为正整数')
            protocol['development_observation_stride'] = validation_stride
        parts, assignment = partition(package, protocol)
    profile = {'gas': 'A', 'bearing': 'B', 'engine': 'C'}[task]
    task_dir = output / 'task'
    task_dir.mkdir()
    parent_bytes = Path(parent_checkpoint).read_bytes() if parent_checkpoint is not None else package.checkpoint
    (task_dir / 'initial.npz').write_bytes(parent_bytes)
    train_x = np.asarray([r.values for r in package.train], dtype=np.float64)
    train_y = np.asarray([r.label - 1 if task == 'gas' else r.label for r in package.train])
    np.savez_compressed(task_dir / 'train.npz', x=train_x, y=train_y,
                        groups=np.asarray([str(r.unit_id or r.group) for r in package.train]))
    arrays = {key: np.asarray([r.values for r in parts[key]], dtype=np.float64) for key in ('explore', 'adopt', 'serve')}
    np.savez_compressed(task_dir / 'evaluation-inputs.npz', explore=arrays['explore'], confirm=arrays['adopt'])
    np.savez_compressed(output / 'service-inputs.npz', x=arrays['serve'])
    with ZipFile(archive) as z, TemporaryDirectory(dir=output, prefix='temporary-') as temporary:
        with _trainer(z, Path(temporary), profile) as trainer:
            model = load_checkpoint(task_dir / 'initial.npz', trainer.Model)
            if model.meta['profile'] != profile:
                raise ValueError('指定父检查点与任务不符')
            # 逐任务读取器的列序必须与官方flatten后的路径完全一致。
            raw = json.loads(z.read('public/train.jsonl').splitlines()[0])
            from importlib import import_module
            data = import_module(trainer.__package__ + '.data')
            paths, values = data.flatten(raw['inputs'])
            if paths != model.meta['paths'] or not np.array_equal(train_x[0], values):
                raise ValueError('任务读取器与官方输入列序不一致')
            initial_prediction = {key: model.predict(value) for key, value in arrays.items()}
        (task_dir / 'baseline-model.py').write_bytes(z.read('baselines/model.py'))
        (task_dir / 'baseline-train.py').write_bytes(z.read('baselines/train.py'))
    atomic_json(task_dir / 'task.json', {'task': task, 'profile': profile, 'stage_id': 'initial',
        'archive': str(archive.resolve()), 'archive_sha256': checksum, 'feature_paths': paths,
        'train_rows': len(train_x), 'inputs_sha256': {
            name: sha256((task_dir / name).read_bytes()).hexdigest() for name in ('train.npz', 'evaluation-inputs.npz')}})
    oracle = {key: {'ids': [r.sample_id for r in parts[key]], 'y': [r.label for r in parts[key]],
                   'groups': [r.unit_id or r.group for r in parts[key]]} for key in ('explore', 'adopt', 'serve')}
    atomic_json(output / 'oracle.json', oracle)
    np.savez_compressed(output / 'initial-predictions.npz', **initial_prediction)
    atomic_json(output / 'identity.json', {'task': task, 'archive_sha256': checksum,
        'parent_sha256': sha256(parent_bytes).hexdigest(), 'official_initial_sha256': sha256(package.checkpoint).hexdigest(),
        'parent_source': str(Path(parent_checkpoint).resolve()) if parent_checkpoint is not None else '官方原包',
        'scope': '官方公开开发集上的原生续训新协议',
        'validation_split': '身份哈希三分' if task == 'gas' else '沿用官方寿命设备划分',
        'validation_observation_stride': None if task == 'gas' else protocol['development_observation_stride'],
        'train_rows': len(train_x), 'evaluation_rows': {key: len(v) for key, v in arrays.items()}})
    return task_dir, oracle
