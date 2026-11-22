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
def export_release(task_dir, parent, destination):
    """发布实际采用的原生模型；保持原版时也生成独立入口，不依赖任务目录。"""
    task_dir, parent, destination = map(Path, (task_dir, parent, destination))
    specification = json.loads((task_dir / 'task.json').read_bytes())
    expected = {'checkpoint.npz': parent.read_bytes()}
    with source_package(specification) as package:
        expected.update(prediction_sources(package, parent))
    for name in ('training.json', 'history.jsonl', 'training-program.py'):
        if (parent.parent / name).is_file():
            expected[name] = (parent.parent / name).read_bytes()
    destination.mkdir(parents=True, exist_ok=True)
    allowed = set(expected) | {'manifest.json'}
    allowed |= {str(Path(name).with_name('.publishing-' + Path(name).name)).replace('\\', '/') for name in expected}
    for existing in destination.rglob('*'):
        if existing.is_file() and existing.relative_to(destination).as_posix() not in allowed:
            raise ValueError('发布目录含来源不明的文件，不能自动覆盖')
    for name, payload in expected.items():
        target = destination / name
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError('已有发布文件与实际采用版不同：' + name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name('.publishing-' + target.name)
        temporary.write_bytes(payload)
        temporary.replace(target)
    report = {'task': specification['task'], 'checkpoint_sha256': sha256(parent.read_bytes()).hexdigest(),
              'source_checkpoint': str(parent.resolve()), 'archive_sha256': specification['archive_sha256'],
              'continued': (destination / 'training.json').is_file(),
              'predict_command': 'python -B -m baselines.predict --checkpoint checkpoint.npz',
              'runtime_dependencies': ['Python', 'NumPy'], 'artifact_bytes': 0}
    with np.load(parent,allow_pickle=False) as arrays:
        kind=json.loads(str(arrays['complete_state'])).get('kind') if 'complete_state' in arrays.files else None
    if kind in ('tabicl-parent-combination','tabicl-classification-parent'):
        report['runtime_dependencies']=['Python','NumPy','PyTorch','scikit-learn','SciPy','psutil',
            'requests','packaging','filelock','fsspec','PyYAML','typing-extensions']
        report['bundled_dependencies']=['TabICL 2.2.0','einops','huggingface-hub','tqdm']
    elif kind in ('tabpfn-classification-parent','tabpfn-regression-parent'):
        report['runtime_dependencies']=['Python','NumPy','PyTorch','scikit-learn','SciPy','pandas','pydantic',
            'pydantic-settings','psutil','requests','packaging','filelock','fsspec','PyYAML','typing-extensions']
        report['bundled_dependencies']=['TabPFN 10.0.0','skrub','safetensors','einops','huggingface-hub','tqdm']
    # 清单本身也计入512 MiB；字节数位数稳定后取得精确总量。
    for _ in range(5):
        atomic_json(destination / 'manifest.json', report)
        total = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
        if total == report['artifact_bytes']:
            break
        report['artifact_bytes'] = total
    else:
        raise ValueError('产物大小记录未收敛')
    if total > 512 * 1024**2:
        raise ValueError('独立发布产物超过512 MiB')
    return report
