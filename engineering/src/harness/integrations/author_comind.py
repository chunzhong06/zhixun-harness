"""运行 CoMind ICLR 2026 作者控制器；只适配供应方、CPU环境与任务评价。

原始 Analyzer、Idea Proposer、Coordinator、并行 Coding Agent 及社区回流均直接导入。
冻结任务代替 Kaggle 网络数据；持久 Jupyter 单元格和作者选优逻辑保留。
"""
import asyncio
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from types import ModuleType

import numpy as np
import pandas as pd

from harness.integrations.author_mlestar import ACTOR
from harness.integrations.notebook_boundary import validate_source
from harness.runtime.storage import atomic_json

ROOT = Path(__file__).resolve().parents[4]
COMMIT = 'f93dff779b5d40dd3bf28b7b6ba5e39881e02e38'
SOURCE = ROOT / 'references/implementations/comind-ml/CoMind' / COMMIT


def _role(messages):
    text = '\n'.join(m.get('content', '') for m in messages)
    if 'determine whether' in text: return 'metric'
    if 'holistic overview' in text: return 'analyzer'
    if 'alternative and highly novel' in text: return 'brainstorm'
    if 'rephrase and decouple' in text: return 'rephrase'
    if 'generate a list of pipelines' in text: return 'coordinator'
    if 'converting a Jupyter notebook' in text: return 'assemble'
    if 'monitoring code execution' in text: return 'monitor'
    return 'coder'


async def run_comind(description, task, workspace, executor, broker, seed_source,
                     *, iterations=2, cells=5, branch_resume=None):
    """每个进程只装载一份作者模块；线程请求统一送回主事件循环计数。"""
    loop = asyncio.get_running_loop()
    return await asyncio.to_thread(_run, description, Path(task), Path(workspace),
                                   executor, broker, loop, seed_source, iterations, cells, branch_resume)


def _run(description, task, workspace, executor, broker, loop, seed_source, iterations, cells, branch_resume):
    manifest = json.loads((SOURCE / 'source-manifest.json').read_bytes())
    for name, checksum in manifest['files'].items():
        if sha256((SOURCE / name).read_bytes()).hexdigest() != checksum:
            raise ValueError('CoMind 作者原件身份变化：' + name)
    if 'comind.agent' in sys.modules:
        raise RuntimeError('每个作者运行需要独立进程')
    sys.path.insert(0, str(SOURCE))
    # 阻止不需要的供应方认证；真正的请求通过本次可追踪 Broker 完成。
    litellm = ModuleType('litellm')
    sys.modules['litellm'] = litellm
    kaggle_api = ModuleType('comind.kaggle.api')
    def unavailable(*args, **kwargs):
        raise RuntimeError('本任务资源已冻结，不访问 Kaggle 账号')
    kaggle_api.authenticate_kaggle_api = unavailable
    sys.modules[kaggle_api.__name__] = kaggle_api
    kagglehub = ModuleType('kagglehub')
    kagglehub.model_download = kagglehub.dataset_download = unavailable
    sys.modules['kagglehub'] = kagglehub
    import comind.agent as author
    import comind.coder as coder_module
    import comind.utils.llm as llm
    import comind.utils.jupyter_session as notebook
    from comind.config.config import Config, LLMConfig
    from comind.evaluate import EvaluationResult
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpec
    cached_resume = [branch_resume['blocked_response']] if branch_resume else []

    async def ask(actor, messages):
        ACTOR.set(actor)
        return await broker.exchange('model', {'request': {'model': 'gpt-6-luna', 'messages': messages},
                                                'search_allowed': False})

    def transport(config, messages, logger=None):
        messages = deepcopy(messages)
        # 只改任务环境文字；研究提示、角色、候选和选择循环保持作者实现。
        for m in messages:
            text = m['content']
            for old, new in [
                ('2 hours on a single A6000 with 48GB memory', '90 seconds on CPU with 1 training thread'),
                ('Time limit per run is 2 hours', 'Time limit per code cell is 90 seconds'),
                ('Your code will be executed on a single A6000 GPU. Use large batchsizes to maximize the gpu utilization.', 'Your code will run on CPU with one training thread.'),
                ('Utilize GPU resources as much as possible.', 'Use the available CPU resources.'),
                ('Always use gpu for acceleration.', 'Use CPU.'),
                ('You are allowed to load huggingface models/datasets/checkpoints.', 'Use only the supplied frozen data and checkpoints.'),
                ('You are allowed to install any packages you need', 'Use only installed packages; you may inspect the legal workspace'),
                ('You are allowed to install any packages by running `pip install <package_name>` in your script. Your installation will take effect in the NEXT cell.', 'The package set is fixed. Do not install packages or access the network.')]:
                text = text.replace(old, new)
            m['content'] = text
        if cached_resume:
            if messages != branch_resume['blocked_messages']:
                raise ValueError('恢复改变了原已生成第三格的上下文')
            return cached_resume.pop()['plan']['text']
        actor = 'comind_' + _role(messages) + '_' + re.sub('[^a-zA-Z0-9_-]', '_', threading.current_thread().name)
        response = asyncio.run_coroutine_threadsafe(ask(actor, messages), loop).result(timeout=broker.timeout + 15)
        return response['text']
    llm.query_llm_raw = transport

    original_copy = author.copytree
    author.copytree = lambda src, dst, use_symlinks=True: original_copy(src, dst, use_symlinks=False)
    community = workspace / 'community'
    community.mkdir(parents=True)
    seed = community / 'current_model.py'
    seed.write_text(seed_source, encoding='utf-8')
    metadata = {'id': 'shared/current-model', 'title': 'Current complete model source',
                'kernel_type': 'script', 'dataset_sources': [], 'model_sources': [], 'kernel_sources': []}
    author.download_kernels = lambda cfg, lower: [seed]
    author.get_kernel_metadata = lambda path: metadata

    lock = threading.Lock()
    sessions = []
    records_by_prediction = {}
    if branch_resume:
        for record in executor.records:
            for candidate in record['candidates']:
                with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as z:
                    valid, confirm = z['validation'].copy(), z['confirmation'].copy()
                records_by_prediction[sha256(valid.tobytes()).hexdigest()] = (candidate, valid, confirm)

    def measure(prediction, candidate):
        if executor.scorer:
            return executor.scorer(prediction)
        from harness.models.sparse_reference import correct, setting_of
        issued = correct(prediction[:, None], executor.oracle['validation_y'],
                         executor.oracle['validation_hours'], setting_of(candidate['feedback']))[:, 0]
        errors = []
        for mask in executor.oracle['exploration_masks']:
            use = mask & np.isfinite(executor.oracle['validation_y'])
            errors.append(float(np.sqrt(np.mean((issued[use] - executor.oracle['validation_y'][use]) ** 2))))
        score = float(np.sqrt(np.mean(np.square(errors))))
        return {'score': score, 'loss': score, 'windows': errors}

    def collect(session, before, result):
        """模型单元格落盘后由外层评分；确认分数不进入任何 LLM 请求。"""
        work = session.cfg.agent_workspace_dir
        created = [p for p in sorted((work / 'artifacts').glob('candidate-*.json')) if p.name not in before]
        with lock:
            dest = executor.folder / f'{len(executor.records) + 1:04d}'
            dest.mkdir(parents=True)
            source = '\n\n'.join(c.code for c in session.cells.cells if c.success)
            (dest / 'candidate.py').write_text(source, encoding='utf-8')
            candidates = []
            for p in created:
                for ext in ('.json', '.pkl', '.npz'):
                    shutil.copy2(p.with_suffix(ext), dest / p.with_suffix(ext).name)
                for suffix in ('-checkpoint.npz', '-training.receipt'):
                    extra = p.with_name(p.stem + suffix)
                    if extra.exists(): shutil.copy2(extra, dest/extra.name)
                saved = dest / p.name
                candidate = json.loads(saved.read_bytes())
                if sha256(saved.with_suffix('.pkl').read_bytes()).hexdigest() != candidate['weights_sha256']:
                    raise ValueError('作者模型权重身份不符')
                with np.load(saved.with_suffix('.npz'), allow_pickle=False) as z:
                    valid, confirm = z['validation'], z['confirmation']
                if valid.shape != executor.oracle['validation_y'].shape or confirm.shape != executor.oracle['confirmation_y'].shape:
                    raise ValueError('作者保存模型的预测行数不符')
                measured = measure(valid, candidate)
                candidate.update(path=str(saved), source_sha256=sha256(source.encode()).hexdigest(),
                                 score=measured['loss'], exploration_rmse=[measured['loss']], task_metrics=measured)
                candidates.append(candidate)
                records_by_prediction[sha256(valid.tobytes()).hexdigest()] = (candidate, valid.copy(), confirm.copy())
                result.output = result.output.replace('HARNESS_EVALUATION:' + p.stem,
                                                       candidate['label'] + ' | exploration loss: ' + str(candidate['score']))
            record = {'sequence': len(executor.records) + 1, 'actor': 'comind_coder',
                      'phase': 'exploration', 'diagnostic': not bool(candidates), 'directory': str(dest),
                      'coding_workspace': str(work),
                      'source': str(dest / 'candidate.py'), 'source_sha256': sha256(source.encode()).hexdigest(),
                      'input_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in task.iterdir() if p.is_file()},
                      'candidates': candidates, 'returncode': 0 if result.success else 1,
                      'stdout': result.output, 'stderr': result.error, 'execution_time': result.execution_time}
            atomic_json(dest / 'execution.json', record)
            executor.records.append(record)
            atomic_json(executor.folder / 'index.json', executor.records)

    class Session(notebook.JupyterSession):
        def __init__(self, cfg, env_name):
            self.cfg, self.env_name = cfg, env_name
            self.cells = notebook.JupyterNotebook(cfg, [])
            work = cfg.agent_workspace_dir.resolve()
            # input 同名副本支持双方完全相同的固定任务 API。
            if not (work / 'input').exists(): shutil.copytree(task, work / 'input')
            self.km = KernelManager(connection_file=str(work / 'kernel-connection.json'))
            env = {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                   'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '42',
                   'IPYTHONDIR': str(work / '.kernel-runtime/ipython'),
                   'JUPYTER_CONFIG_DIR': str(work / '.kernel-runtime/config'),
                   'JUPYTER_RUNTIME_DIR': str(work / '.kernel-runtime/runtime')}
            self.km._kernel_spec = KernelSpec(argv=[sys.executable, '-B', '-m', 'ipykernel_launcher',
                '-f', '{connection_file}', '--HistoryManager.enabled=False'], display_name=env_name, language='python', env=env)
            self.km.start_kernel(cwd=str(work))
            self.kc = self.km.client()
            self.kc.start_channels()
            self.kc.wait_for_ready(timeout=30)
            sessions.append(self)
            bootstrap = ('import sys\nsys.path.insert(0,' + repr(str(ROOT / 'engineering/src')) + ')\n'
                         'from harness.integrations.notebook_boundary import install\n'
                         'install(' + repr(str(work)) + ',' + repr(str(ROOT / 'engineering/src')) + ')')
            if branch_resume:
                bootstrap += '''
import cloudpickle as __recovery_pickle
from pathlib import Path as __recovery_Path
import random as __recovery_random
import numpy as __recovery_np
__recovery_packet=__recovery_pickle.loads(__recovery_Path('recovery-namespace.pkl').read_bytes())
globals().update(__recovery_packet['namespace'])
__recovery_random.setstate(__recovery_packet['python_random'])
__recovery_np.random.set_state(__recovery_packet['numpy_random'])
'''
            message = self.kc.execute(bootstrap, store_history=False)
            result = self._collect_iopub_outputs(message, bootstrap, '初始化合法任务边界')
            if not result.success: raise RuntimeError(result.error)

        def append_cell(self, code, goal):
            before = {p.name for p in (self.cfg.agent_workspace_dir / 'artifacts').glob('candidate-*.json')}
            try:
                validate_source(code)
            except (ValueError, SyntaxError) as error:
                # 接口拒绝也应成为作者可以调试的执行反馈，不能直接杀死整个搜索分支。
                result = notebook.ExecutionResult(False, False, False, '', str(error), 0.0)
                self.cells.append(code, False)
            else:
                result = super().append_cell(code, goal)
            collect(self, before, result)
            return result

    coder_module.JupyterSession = Session
    import comind.prepare as prepare_module
    prepare_module.JupyterSession = Session
    coder_module.CodeAgent._get_packages = lambda self: ['numpy', 'pandas', 'scipy', 'scikit-learn', 'torch CPU', 'cloudpickle']

    class FixedEvaluator:
        """固定官方评价与划分替代作者的 Kaggle 随机划分生成器。"""
        def __init__(self, cfg, lower):
            self.public_dir = task
        def setup(self): pass
        def evaluate(self, path):
            try:
                values = pd.read_csv(path)['prediction'].to_numpy(dtype=float)
                # CSV 的十进制往返可能改变末位；按实际数值核对对应已重载的模型。
                matched = [v for v in records_by_prediction.values()
                           if values.shape == v[1].shape and np.allclose(values, v[1], rtol=0, atol=1e-8)]
                if not matched: raise ValueError('提交预测没有对应已保存、重载的实际模型')
                result = measure(values, matched[-1][0])
                return EvaluationResult(result['score'], None, True, '外层真实探索评分')
            except Exception as error:
                return EvaluationResult(None, None, False, str(error))
    author.Evaluator = FixedEvaluator

    actual_model = getattr(getattr(broker, 'config', None), 'model', 'gpt-6-luna')
    cfg = Config(competition_id='frozen_task', competition_input_dir=task,
                 competition_task_desc=description, llm=LLMConfig(model=actual_model, max_retries=2),
                 agent_num_code_agents=2, agent_num_iterations=iterations,
                 agent_num_iterations_code_agents=cells + 1, agent_time_limit_code_agents=3600,
                 agent_max_referred_kernels=1, agent_workspace_dir=workspace / 'author',
                 agent_external_data_dir=community, agent_separate_validation_submission=True,
                 agent_reproduce=False, execution_inspect_interval=60, execution_timeout=90,
                 execution_max_cpu_cores=1, execution_max_gpu_count=0)
    cfg.start_time = time.time()
    atomic_json(workspace / 'author-configuration.json', {'commit': COMMIT, 'iterations': iterations,
        'coding_agents': 2, 'cells_per_coder': cells, 'max_calls': broker.max_calls,
        'model': actual_model,
        'adaptations': ['双方同一模型供应方和生成设置', '固定合法数据和已有源码替代 Kaggle 社区入口',
                        '固定官方划分和真实指标替代随机划分生成', 'CPU 单线程 Jupyter 环境',
                        '文件复制替代 Windows 符号链接', '必须交付可重载模型'],
        'retained': ['作者分析提示', '头脑风暴', '方案记忆整理', '协调分案', '并行编程',
                     '持久单元格', '执行监测', '真实探索选优', '清理源码', '结果回流社区', '全局榜单']})
    agent = None
    reports_override = None
    try:
        if branch_resume:
            from harness.integrations.comind_branch_recovery import resume_branch
            agent, reports_override = resume_branch(branch_resume, cfg=cfg, author=author,
                coder_module=coder_module, notebook=notebook, executor=executor, evaluator=FixedEvaluator)
            if cached_resume: raise ValueError('未复用原已生成的第三格，恢复不完整')
        else:
            agent = author.Agent(cfg)
            agent.run()
        successful = [r for r in agent.reports if r.submission is not None and not isinstance(r.metric, author.WorstMetricValue)]
        if not successful: raise ValueError('CoMind 完整流程未交付有效模型')
        selected_report = max(successful, key=lambda r: r.metric)
        # 作者报告的 submission 在退出时恢复到其最佳单元格；再绑定真实权重。
        selected_values = pd.read_csv(selected_report.submission)['prediction'].to_numpy(dtype=float)
        eligible = []
        for record in executor.records:
            if record['returncode']:
                continue
            if record.get('coding_workspace') != str(selected_report.output_dir):
                continue
            for candidate in record['candidates']:
                with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as z:
                    confirmation = z['confirmation']
                if confirmation.shape == selected_values.shape and np.allclose(confirmation, selected_values, rtol=0, atol=1e-8):
                    eligible.append((record, candidate))
        if not eligible: raise ValueError('CoMind 最终提交没有对应实际权重')
        record, candidate = min(eligible, key=lambda x: x[1]['score'])
        # 以作者选择的确切候选作为末项，供共同交付函数读取。
        selected = {**record, 'candidates': [candidate]}
        atomic_json(workspace / 'author-selection.json', {'report_id': selected_report.id,
            'metric': str(selected_report.metric), 'selected_execution': record['sequence'],
            'candidate': candidate, 'full_author_source': selected_report.full_code,
            'reports': reports_override or [{'id': r.id, 'metric': str(r.metric)} for r in agent.reports],
            'ideas': agent.ideas})
        return selected
    finally:
        for session in sessions:
            try: session.kc.stop_channels(); session.km.shutdown_kernel(now=True)
            except Exception: pass
        atomic_json(workspace / 'driver-complete.json', {'model_calls': broker.calls,
                    'search_calls': broker.search_calls, 'executions': len(executor.records)})
