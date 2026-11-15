"""用实际入口判定比赛完成度，并在固定版本上核对性能与代码来源。"""
import ast
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import cloudpickle
import numpy as np
import pandas as pd
import torch
from engineering.cli.study.evaluate_native_transfer import dataset
from engineering.cli.study.evaluate_rul_baseline import PACKAGES
from engineering.cli.workflow.native_continuation import execute
from engineering.cli.workflow.native_research import current_parent
from harness.integrations.native_research import prepare, export_release
from harness.models.complete_checkpoint import load_checkpoint, is_complete
from harness.models.continuation_recipes import reference_programs
from harness.models.native_continuation import ContinuationSession, validate_program
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'
OUT = BASE / 'competition-readiness-audit'


def process(command, cwd, data=''):
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, '-B', *command], cwd=cwd, input=data,
                            capture_output=True, text=True, encoding='utf-8', timeout=45, env=env)
    return {'command': command, 'exit_code': result.returncode,
            'stdout_tail': result.stdout[-1000:], 'stderr_tail': result.stderr[-1800:]}


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    evidence = {'tasks': {}, 'new_model_calls': 0, 'platform_evaluation_performed': False}
    for task in ('gas', 'bearing', 'engine'):
        folder = OUT / task
        task_dir, _ = prepare(task, folder / 'from-official')
        spec = json.loads((task_dir / 'task.json').read_bytes())
        checkpoint = current_parent(task)
        release = folder / 'current-inference'
        export_release(task_dir, checkpoint, release)
        manifest = json.loads((release / 'manifest.json').read_bytes())
        with ZipFile(spec['archive']) as package, TemporaryDirectory(dir=folder, prefix='temporary-') as tmp:
            with _trainer(package, Path(tmp), spec['profile']) as trainer:
                restored = load_checkpoint(checkpoint, trainer.Model)
                try:
                    trainer.Model.load(checkpoint)
                    original_loader = {'accepted': True}
                except (ValueError, KeyError) as error:
                    original_loader = {'accepted': False, 'error': str(error)}
                raw = [json.loads(line) for line in package.read('public/validation.jsonl').splitlines()[:12]]
                requests = '\n'.join(json.dumps({'task': restored.meta['task'], 'inputs': r['inputs']}) for r in raw)
        module = process(['-m', 'baselines.predict', '--checkpoint', 'checkpoint.npz'], release, requests)
        script = process(['baselines/predict.py', '--checkpoint', 'checkpoint.npz'], release, requests)
        if module['exit_code']:
            raise ValueError('已登记的独立入口失败：' + task + module['stderr_tail'])
        source = folder / 'one-epoch.py'
        source.write_text('def train(session):\n    session.train_epoch()\n', encoding='utf-8')
        initial = task_dir / 'initial.npz'
        trained = execute(task_dir, initial, source, folder / 'official-continuation',
                          expected_parent_sha256=sha256(initial.read_bytes()).hexdigest(), max_epochs=1)
        current_has_components = is_complete(checkpoint)
        produced_has_components = is_complete(folder / 'official-continuation/checkpoint.npz')
        evidence['tasks'][task] = {
            'current_checkpoint_sha256': sha256(checkpoint.read_bytes()).hexdigest(),
            'current_contains_complete_components': current_has_components,
            'official_reference_loader': original_loader, 'module_prediction': module,
            'script_prediction': script, 'model_artifact_bytes': manifest['artifact_bytes'],
            'release_includes_training_history': manifest['continued'],
            'official_start_is_exact_baseline': not is_complete(initial),
            'actual_official_continuation': {'parent_sha256': trained['parent_checkpoint_sha256'],
                'initial_steps': trained['initial_steps'], 'final_steps': trained['final_steps'],
                'changed_parameters': trained['parameter_changes'],
                'produced_complete_components': produced_has_components}}
        atomic_json(OUT / 'interface-checks.json', evidence)
        print(json.dumps({'task_interface': task, 'module': module['exit_code'], 'script': script['exit_code'],
                          'official_loader': original_loader, 'fresh_complete': produced_has_components}, ensure_ascii=False), flush=True)
    methods = [name for name in vars(ContinuationSession) if not name.startswith('_')]
    try:
        validate_program('from harness.models.complete_checkpoint import migrate_engine\ndef train(session):\n    session.train_epoch()\n')
        import_allowed = True
    except ValueError as error:
        import_allowed = False
        import_error = str(error)
    evidence['fresh_start_capability'] = {'public_session_methods': methods,
        'reference_names': list(reference_programs()), 'component_import_allowed': import_allowed,
        'component_import_error': None if import_allowed else import_error,
        'finding': '当前现场续训接口可调整原网络；完整组件通过离线迁移函数创建，尚无从官方起点构建这些组件的现场训练方法。'}
    dockerfile = ROOT / 'engineering/configs/container/Dockerfile'
    lines = dockerfile.read_text(encoding='utf-8').splitlines()
    entry = json.loads(next(s.removeprefix('ENTRYPOINT ') for s in lines if s.startswith('ENTRYPOINT ')))
    cmd = json.loads(next(s.removeprefix('CMD ') for s in lines if s.startswith('CMD ')))
    with TemporaryDirectory(dir=OUT, prefix='temporary-container-') as tmp:
        isolated = Path(tmp)
        for line in lines:
            if not line.startswith('COPY '): continue
            _, src, dest = line.split()
            target = isolated / dest
            if (ROOT / src).is_dir():
                shutil.copytree(ROOT / src, target, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            else:
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / src, target)
        invocation = [*entry[2:], 'native-research', 'engine', '--current', '--output', str(isolated / 'output'), '--calls', '1']
        portable = process(invocation, isolated)
    evidence['container'] = {'available_runtimes': {x: shutil.which(x) for x in ('docker', 'podman', 'nerdctl', 'buildctl')},
        'actual_image_built': False, 'dockerfile_default_command': cmd,
        'copy_context_current_model_invocation': portable}
    files = ['integrations/free_research_driver.py', 'integrations/native_research.py', 'runtime/engine.py',
             'integrations/candidate_selection.py', 'integrations/research_recovery.py', 'models/complete_checkpoint.py']
    own_imports = {}
    for relative in files:
        path = ROOT / 'engineering/src/harness' / relative
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imports.extend(n.name for n in node.names)
            if isinstance(node, ast.ImportFrom) and node.module: imports.append(node.module)
        own_imports[relative] = {'sha256': sha256(path.read_bytes()).hexdigest(),
                                'comind_imports': [n for n in imports if 'comind' in n.lower()]}
    source = ROOT / 'references/implementations/comind-ml/CoMind/f93dff779b5d40dd3bf28b7b6ba5e39881e02e38'
    provenance = json.loads((source / 'source-manifest.json').read_bytes())
    matched = all(sha256((source / n).read_bytes()).hexdigest() == value for n, value in provenance['files'].items())
    evidence['design_provenance'] = {'own_execution_path': own_imports, 'author_source_manifest_matches': matched,
        'comparison_adapter': 'engineering/src/harness/integrations/author_comind.py',
        'finding': '本项目主控制、确认发布与恢复独立实现；CoMind作者代码由对照适配器导入。工况预处理、Huber、LDA、SVM等为复用成熟机制，不作为新算法发明。'}
    atomic_json(OUT / 'interface-checks.json', evidence)
    # 双方旧搜索的已采用模型冻结后，同一官方全量输入重新交付，不重选或补训。
    paired = ROOT / 'results/experiments/local-defaults/whole-maintenance/robust-shared-full-pair/engine/comind'
    decision = json.loads((paired / 'decision.json').read_bytes())['candidate']
    weights = Path(decision['path']).with_suffix('.pkl')
    if sha256(weights.read_bytes()).hexdigest() != decision['weights_sha256']:
        raise ValueError('作者实际选中的模型已改变')
    with weights.open('rb') as stream: author = cloudpickle.load(stream)
    filename, _, reader = PACKAGES['engine']
    archive = ROOT / 'data/official/engine' / filename
    package = reader(archive)
    x = np.asarray([r.values for r in package.validation])
    with ZipFile(archive) as z, TemporaryDirectory(dir=OUT, prefix='temporary-pair-') as tmp:
        with _trainer(z, Path(tmp), 'C') as trainer:
            current = load_checkpoint(current_parent('engine'), trainer.Model)
            native = current.native_features(x)
            parent = current.base.predict(native * current.scale + current.mean)
            columns = [f'input_{i}' for i in range(24)] + [f'native_{i}' for i in range(24)] + ['parent_value']
            frame = pd.DataFrame(np.column_stack((x, native, parent)), columns=columns)
            ours, theirs = [], []
            for i in range(len(frame)):
                ours.append(current.predict(x[i:i+1])[0])
                theirs.append(float(author.predict(frame.iloc[i:i+1])[0]))
                if (i+1) % 5000 == 0: print(json.dumps({'frozen_pair_rows': i+1}), flush=True)
            ours, theirs = np.asarray(ours), np.asarray(theirs)
            np.savez_compressed(OUT / 'frozen-engine-pair.npz', ids=np.asarray([r.sample_id for r in package.validation]),
                                ours=ours, comind=theirs)
    labels = np.asarray([r.label for r in package.validation])
    by_group = np.asarray([r.unit_id.split(':')[0] for r in package.validation])
    evidence['frozen_engine_full_public'] = {'scope': '固定双方已采用模型的同条件全量交付，非重新运行两套研究控制器',
        'rows': len(labels), 'ours_rmse': rmse(labels, ours), 'comind_rmse': rmse(labels, theirs),
        'by_subset': {g: {'ours': rmse(labels[by_group == g], ours[by_group == g]),
                         'comind': rmse(labels[by_group == g], theirs[by_group == g])} for g in sorted(set(by_group))},
        'our_checkpoint_sha256': sha256(current_parent('engine').read_bytes()).hexdigest(),
        'author_weights_sha256': decision['weights_sha256'], 'new_training': 0, 'new_search': 0}
    atomic_json(OUT / 'verification.json', evidence)
    print(json.dumps({'frozen_engine_full_public': evidence['frozen_engine_full_public']}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
