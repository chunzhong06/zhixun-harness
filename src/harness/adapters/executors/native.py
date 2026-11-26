"""将现有源码研究驱动接到官方NPZ续训；探索评分与最终发布留在外层。"""
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from functools import lru_cache
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
from harness.evaluation.native import measure, exploration_diagnostics


@lru_cache(maxsize=1)
def training_runtime():
    """跨轮比较要区分训练实现相同、数值依赖版本不同的镜像。"""
    packages = {}
    for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'tabpfn', 'tabm'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return [sys.version, sys.platform, packages]


class NativeExecutor:
    """沿用free_research_driver记录与提名协议；原始训练输入只读共享。"""

    def __init__(self, folder, task_dir, parent, oracle, *, max_epochs=30, timeout=90, resume=False,
                 cache_dir=None, cacheable_sources=()):
        self.folder, self.task_dir, self.parent = (Path(p).resolve() for p in (folder, task_dir, parent))
        self.folder.mkdir(parents=True, exist_ok=resume)
        self.task = json.loads((self.task_dir / 'task.json').read_bytes())['task']
        self.oracle, self.max_epochs, self.timeout = oracle, max_epochs, timeout
        self.parent_sha = sha256(self.parent.read_bytes()).hexdigest()
        self.records = []
        self.resume = resume
        self.replayed_executions = 0
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cacheable_sources = {sha256(code.encode()).hexdigest() for code in cacheable_sources}

    def reference_key(self, source):
        """只复用已登记的确定性参考程序；自由随机程序继续真实执行。"""
        source_sha = sha256(source.encode()).hexdigest()
        if self.cache_dir is None or source_sha not in self.cacheable_sources:
            return None
        task = json.loads((self.task_dir / 'task.json').read_bytes())
        if sha256(self.parent.read_bytes()).hexdigest() != self.parent_sha:
            raise ValueError('实际父检查点在执行期间改变')
        for name, expected in task['inputs_sha256'].items():
            if sha256((self.task_dir / name).read_bytes()).hexdigest() != expected:
                raise ValueError('冻结训练或预测输入已改变，不能复用候选')
        with source_package(task): pass
        resource = task.get('tabular_resource')
        if resource and sha256(Path(resource['checkpoint']).read_bytes()).hexdigest() != resource['sha256']:
            raise ValueError('共享教师资产已变化，不能复用候选')
        code = package_sources()
        # SDK 按容器配额选择教师线程数；切换数值配置时重做测量。
        return digest({'schema': 3, 'source_sha256': source_sha, 'parent_sha256': self.parent_sha,
                       'profile': task['profile'], 'archive_sha256': task['archive_sha256'],
                       'inputs_sha256': task['inputs_sha256'], 'max_epochs': self.max_epochs,
                       'data_identity': task.get('data_identity'),
                       'tabular_resource': {k:v for k,v in resource.items() if k != 'checkpoint'} if resource else None,
                       'evaluation_identity': digest({k: self.oracle[k] for k in ('explore', 'adopt')}),
                       'runtime': training_runtime(), 'teacher_threads': os.environ.get('ZHIXUN_SDK_PREDICT_THREADS'),
                       'code': file_identity(code)})

    def seed_parent(self, predictions):
        """父版是已存在的保持选项，明确记作零次新训练，不伪装成候选实验。"""
        folder = self.folder / '0000'
        folder.mkdir(exist_ok=self.resume)
        source = folder / 'keep_parent.py'
        source.write_text('# 已选中父版；保持时提名执行0的parent，不执行该占位程序。\n', encoding='utf-8')
        measured = measure(self.task, self.oracle['explore']['y'], predictions)
        candidate = {'path': str(folder / 'parent.json'), 'checkpoint': str(self.parent),
            'label': '保持实际父模型', 'weights_sha256': self.parent_sha, 'parent_sha256': self.parent_sha,
            'score': measured['loss'], 'exploration_rmse': [measured['loss']], 'task_metrics': measured,
            'feedback': exploration_diagnostics(self.task, self.oracle['explore']['y'], predictions, self.oracle['explore']['groups'])}
        atomic_json(Path(candidate['path']), candidate)
        self.records.append({'sequence': 0, 'actor': 'existing_parent', 'phase': 'baseline', 'diagnostic': False,
            'directory': str(folder), 'source': str(source), 'source_sha256': sha256(source.read_bytes()).hexdigest(),
            'returncode': 0, 'stdout': '本轮已生效父模型，未执行新训练', 'stderr': '', 'execution_time': 0,
            'candidates': [candidate]})

    def run_reference(self, *args):
        """固定起点也走真实执行与恢复路径，只把来源明确记为系统参考。"""
        token = ACTOR.set('frozen_starting_reference')
        try:
            return self(*args)
        finally:
            ACTOR.reset(token)

    def __call__(self, source, run_cwd, py_filepath, exec_timeout):
        sequence = max((r['sequence'] for r in self.records), default=0) + 1
        folder = self.folder / f'{sequence:04d}'
        if folder.exists():
            if not self.resume:
                raise ValueError('候选目录已存在，不能覆盖原实验')
            saved = folder / 'execution.json'
            if not saved.is_file():
                # 训练可能已结束但还未落完整回执；保留原目录，不能假定没有消耗。
                raise RuntimeError('已有候选缺少完整执行回执，先核查再恢复')
            record = json.loads(saved.read_bytes())
            if record['source_sha256'] != sha256(source.encode()).hexdigest() or record['parent_sha256'] != self.parent_sha:
                raise ValueError('恢复候选的源码或实际父版变化')
            verify_files(record['artifact_sha256'])
            self.records.append(record)
            self.replayed_executions += 1
            return {key: record[key] for key in ('returncode', 'stdout', 'stderr', 'execution_time')}
        folder.mkdir()
        program = folder / 'training-program.py'
        program.write_text(source, encoding='utf-8')
        lookup_started = time.monotonic()
        cache_key = self.reference_key(source)
        cached = self.cache_dir / (cache_key + '.json') if cache_key is not None else None
        if cached is not None and cached.exists():
            pointer = json.loads(cached.read_bytes())
            origin = Path(pointer['execution'])
            if sha256(origin.read_bytes()).hexdigest() != pointer['execution_sha256']:
                raise ValueError('候选复用来源回执改变')
            previous = json.loads(origin.read_bytes())
            verify_files(previous['artifact_sha256'])
            if (previous['returncode'] or previous.get('cache_key') != cache_key
                    or previous['parent_sha256'] != self.parent_sha or previous['source_sha256'] != sha256(source.encode()).hexdigest()):
                raise ValueError('候选复用来源身份不符')
            record = {**previous, 'sequence': sequence, 'actor': ACTOR.get(), 'directory': str(folder),
                      'source': str(program), 'execution_reused': True, 'launched_training_worker': False,
                      'reused_from': str(origin), 'origin_execution_seconds': previous['execution_time'],
                      'execution_time': time.monotonic()-lookup_started, 'cache_key': cache_key,
                      'stdout': previous['stdout']+'\n复用相同父版、资料和确定性程序的已完成候选；本次没有重新训练。',
                      'artifact_sha256': {**previous['artifact_sha256'], **file_identity([origin, program])}}
            atomic_json(folder / 'execution.json', record)
            self.records.append(record)
            atomic_json(self.folder / 'index.json', self.records)
            return {key: record[key] for key in ('returncode', 'stdout', 'stderr', 'execution_time')}
        destination = folder / 'candidate'
        env = os.environ.copy()
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        env['PYTHONPATH'] = str(PACKAGE_ROOT.parent) + os.pathsep + env.get('PYTHONPATH','')
        started = time.monotonic()
        command = [sys.executable, '-B', '-m', 'harness.learning.training.worker',
            '--task-dir', str(self.task_dir), '--parent', str(self.parent), '--expected-parent-sha256', self.parent_sha,
            '--program', str(program), '--output', str(destination), '--max-epochs', str(self.max_epochs),
            '--stage-id', self.folder.parent.name + '-trial-' + str(sequence)]
        if self.task in ('gas','bearing'):
            command.extend(['--teacher-cache', str(self.task_dir.parent / '.frozen-teacher')])
        from harness.contest.round_budget import current
        budget = current.get()
        cap = min(exec_timeout,self.timeout)
        if budget:
            cap = budget.allowance(cap)
            budget.marker('start_worker' if cap else 'skip_worker')
        if not cap:
            result = {'returncode':1,'stdout':'','stderr':'已预留选案和交接时间，本候选未启动',
                      'launched_training_worker':False}
        else:
            try:
                process = subprocess.run(command, cwd=self.task_dir, env=env, capture_output=True, text=True,
                                         encoding='utf-8', timeout=cap)
                result = {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
                if budget and process.returncode == 0:
                    budget.observe('worker',time.monotonic()-started)
            except subprocess.TimeoutExpired:
                result = {'returncode': 1, 'stdout': '', 'stderr': '续训源码超过本次CPU时间上限'}
                if budget:
                    budget.marker('worker_timeout')
        result['timeout_seconds'] = cap
        result['execution_time'] = time.monotonic() - started
        candidates = []
        if result['returncode'] == 0:
            report = json.loads((destination / 'training.json').read_bytes())
            checkpoint = destination / 'checkpoint.npz'
            if sha256(checkpoint.read_bytes()).hexdigest() != report['checkpoint_sha256'] or report['parent_checkpoint_sha256'] != self.parent_sha:
                raise ValueError('子进程交付的父子权重身份不符')
            with np.load(destination / 'predictions.npz', allow_pickle=False) as arrays:
                measured = measure(self.task, self.oracle['explore']['y'], arrays['explore'])
                diagnostics = exploration_diagnostics(self.task, self.oracle['explore']['y'], arrays['explore'], self.oracle['explore']['groups'])
            candidate = {'path': str(folder / 'candidate-001.json'), 'checkpoint': str(checkpoint),
                         'label': '官方父权重续训', 'weights_sha256': report['checkpoint_sha256'],
                         'parent_sha256': self.parent_sha, 'source_sha256': sha256(source.encode()).hexdigest(),
                         'score': measured['loss'], 'exploration_rmse': [measured['loss']],
                         'task_metrics': measured, 'feedback': diagnostics, 'report': str(destination / 'training.json')}
            atomic_json(Path(candidate['path']), candidate)
            candidates.append(candidate)
            result['stdout'] = json.dumps({'探索评价': measured, '探索诊断': diagnostics, '累计步数': report['final_steps'],
                '训练轮数': report['final_epochs'] - report['initial_epochs'], '改变参数': report['parameter_changes'],
                '产物字节': report['artifact_bytes']}, ensure_ascii=False)
        record = {'sequence': sequence, 'actor': ACTOR.get(), 'phase': 'exploration', 'diagnostic': False,
                  'directory': str(folder), 'source': str(program), 'source_sha256': sha256(source.encode()).hexdigest(),
                  'parent_sha256': self.parent_sha,
                  'execution_reused': False, 'launched_training_worker': True, 'cache_key': cache_key,
                  'artifact_sha256': file_identity(p for p in folder.rglob('*') if p.is_file()),
                  'candidates': candidates, **result}
        atomic_json(folder / 'execution.json', record)
        self.records.append(record)
        atomic_json(self.folder / 'index.json', self.records)
        if cached is not None and result['returncode'] == 0:
            cached.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(cached, {'execution': str(folder / 'execution.json'),
                                'execution_sha256': sha256((folder / 'execution.json').read_bytes()).hexdigest()})
        return result
