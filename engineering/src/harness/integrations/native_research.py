"""将现有源码研究驱动接到官方NPZ续训；探索评分与最终发布留在外层。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import json
import os
import subprocess
import sys
import time

import numpy as np

from harness.contracts import digest
from harness.data.official_gas import read_official_gas
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.models.official_rul_cpu import _trainer
from harness.models.complete_checkpoint import load_checkpoint, prediction_sources
from harness.integrations.research_broker import ACTOR
from harness.integrations.research_recovery import file_identity, verify_files
from harness.runtime.storage import atomic_json

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE = '''def train(session):
    session.train_epoch()
'''

INSTRUCTION = '''你是Harness的现场研究驱动，目标是在官方父权重上真实续训，改善完整后续交付。
仅使用请求中的合法资料、源码和真实探索反馈。确认标签、未来答案不提供；停止不要求用完预算。
可返回以下纯JSON动作之一：
{"action":"reference","name":"已提供参考程序名称","hypothesis":"为何需要测试这一机制"}
{"action":"experiment","hypothesis":"可检验的中文判断","python_source":"完整训练程序"}
{"action":"finish","execution":实际执行号,"candidate":"候选ID","reason":"选择或保持的理由"}
执行号0、候选parent是本轮实际父模型；可以直接提名它保持原版。
程序必须提供train(session)。程序执行前，宿主已从指定检查点恢复原网络、Adam与尺度，不能重建网络冒充续训。
session.metadata是状态副本；session.features和session.targets是公开训练特征与标签副本，列序为metadata['paths']；session.training_predictions()只预测训练输入。
session.groups是合法训练资料的采集单位身份，可用于权重但不加入预测特征。
普通回归父版可调用session.build_regression_components(condition_indices,decimals,epochs=60,head_epochs=12)，只用合法源训练资料建立工况处理、真实续训父网络并训练残差头；原权重、Adam和尺度完整继承，所有组件及新优化器状态保存。已有完整父版沿用其组件，不能重复创建。
session.expand_hidden(新宽度)可在首个训练轮前扩展一次隐藏层，宽度不超过128。原权重及Adam块完整保留，新增输出连接为零，扩展时预测保持一致；后续再通过真实梯度训练新增参数。
session.train_epoch(learning_rate=可选浮点数,batch_size=可选正整数,trainable=可选参数名列表,
loss_gradient=可选函数,sample_weights=可选逐训练行权重)进行一轮真实更新并返回真实训练摘要。
已有参数名为w1,b1,w2,b2。只训练输出层可令trainable=['w2','b2']，其余权重和Adam矩保持不变。
自定义loss_gradient(logits,target)返回(逐样本损失,未取均值的输出梯度)，梯度形状与logits相同。
回归target已经使用原检查点的target_mean/target_scale标准化；分类target为从0开始的类别编号。
可用numpy、math实现数值计算，可重复调用train_epoch，总轮数须遵守task_description上限。
禁止文件、网络、外部进程、私有状态与动态执行；保存、训练历史、父身份及预测均由宿主生成。
普通父版保持原NPZ网络类型；完整发动机父版保留工况及残差并对完整输出求导；气体源训练组合保留固定投票，使用原生分类损失训练父网络，再按完整投票预测比较。独立入口加载全部数值状态。候选不满足确认条件时保留父模型。
把每次已有试验作为后续判断依据。候选和父模型使用同一探索集合，loss越低越好。
剩余模型请求为1时必须结束并提名已实际保存的候选，不能再启动试验。
'''


def prepare(task, output, *, data_root=ROOT / 'data/official', parent_checkpoint=None):
    """只按身份划分公开验证资料，训练始终使用官方训练集；不改变旧研究任务。"""
    from engineering.cli.study.evaluate_rul_baseline import PACKAGES
    from engineering.cli.study.run_rul_maintenance import partition, PROTOCOL
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
        parts, assignment = partition(package, json.loads(PROTOCOL.read_bytes()))
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
        'train_rows': len(train_x), 'evaluation_rows': {key: len(v) for key, v in arrays.items()}})
    return task_dir, oracle


def measure(task, labels, prediction):
    labels = np.asarray(labels)
    if task == 'gas':
        probability = np.asarray(prediction)
        if probability.shape != (len(labels), 6) or not np.isfinite(probability).all():
            raise ValueError('气体概率输出无效')
        predicted = probability.argmax(axis=1) + 1
        if not np.isin(labels, np.arange(1, 7)).all():
            raise ValueError('气体标签须为官方六类编号')
        # 分轮样本中可能缺少稀有类；各轮仍按官方六类等权，空分母计0。
        values = []
        for c in range(1, 7):
            denominator = np.sum(predicted == c) + np.sum(labels == c)
            values.append(2 * np.sum((predicted == c) & (labels == c)) / denominator if denominator else 0.)
        score = float(np.mean(values))
        return {'score': score, 'loss': 1 - score, 'metric': 'Macro-F1'}
    prediction = np.asarray(prediction)
    if prediction.shape != labels.shape or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError('寿命输出无效')
    score = float(np.sqrt(np.mean((prediction - labels) ** 2)))
    return {'score': score, 'loss': score, 'metric': 'RMSE'}


def exploration_diagnostics(task, labels, prediction, groups):
    """只汇总允许观察的探索标签，给驱动定位类别或设备误差的依据。"""
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    if task == 'gas':
        predicted = prediction.argmax(axis=1) + 1
        confusion = [[int(np.sum((labels == actual) & (predicted == guessed)))
                      for guessed in range(1, 7)] for actual in range(1, 7)]
        return {'labels': list(range(1, 7)), 'confusion_true_rows_prediction_columns': confusion}
    group_names = np.asarray([str(group) for group in groups])
    result = []
    for name in sorted(set(group_names)):
        selected = group_names == name
        error = prediction[selected] - labels[selected]
        result.append({'group': name, 'rows': int(selected.sum()),
                       'rmse': float(np.sqrt(np.mean(error ** 2))),
                       'prediction_minus_target_mean': float(np.mean(error)),
                       'target_mean': float(np.mean(labels[selected]))})
    return {'by_group': result}


def export_release(task_dir, parent, destination):
    """发布实际采用的原生模型；保持原版时也生成独立入口，不依赖任务目录。"""
    task_dir, parent, destination = map(Path, (task_dir, parent, destination))
    specification = json.loads((task_dir / 'task.json').read_bytes())
    archive = Path(specification['archive'])
    if sha256(archive.read_bytes()).hexdigest() != specification['archive_sha256']:
        raise ValueError('发布时官方预测源码身份变化')
    expected = {'checkpoint.npz': parent.read_bytes()}
    with ZipFile(archive) as package:
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
    # 清单本身也计入64 MiB；字节数位数稳定后取得精确总量。
    for _ in range(5):
        atomic_json(destination / 'manifest.json', report)
        total = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
        if total == report['artifact_bytes']:
            break
        report['artifact_bytes'] = total
    else:
        raise ValueError('产物大小记录未收敛')
    if total > 64 * 1024**2:
        raise ValueError('独立发布产物超过64 MiB')
    return report


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

    def _cache_key(self, source):
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
        if sha256(Path(task['archive']).read_bytes()).hexdigest() != task['archive_sha256']:
            raise ValueError('官方原包改变，不能复用候选')
        code = [Path(__file__), ROOT / 'engineering/cli/workflow/native_continuation.py',
                ROOT / 'engineering/src/harness/models/native_continuation.py',
                ROOT / 'engineering/src/harness/models/complete_checkpoint.py',
                ROOT / 'engineering/src/harness/models/gas_views.py',
                ROOT / 'engineering/src/harness/models/official_rul_cpu.py']
        return digest({'schema': 1, 'source_sha256': source_sha, 'parent_sha256': self.parent_sha,
                       'profile': task['profile'], 'archive_sha256': task['archive_sha256'],
                       'inputs_sha256': task['inputs_sha256'], 'max_epochs': self.max_epochs,
                       'evaluation_identity': digest({k: self.oracle[k] for k in ('explore', 'adopt')}),
                       'runtime': [sys.version, np.__version__], 'code': file_identity(code)})

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
        cache_key = self._cache_key(source)
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
        started = time.monotonic()
        command = [sys.executable, '-B', str(ROOT / 'engineering/cli/workflow/native_continuation.py'),
            '--task-dir', str(self.task_dir), '--parent', str(self.parent), '--expected-parent-sha256', self.parent_sha,
            '--program', str(program), '--output', str(destination), '--max-epochs', str(self.max_epochs),
            '--stage-id', self.folder.parent.name + '-trial-' + str(sequence)]
        try:
            process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True,
                                     encoding='utf-8', timeout=min(exec_timeout, self.timeout))
            result = {'returncode': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr}
        except subprocess.TimeoutExpired:
            result = {'returncode': 1, 'stdout': '', 'stderr': '续训源码超过本次CPU时间上限'}
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
