"""官方父检查点上的真实源码研究与采用；支持Luna队列或正式PyTRiO服务。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import asyncio
import json
import os
import sys

from harness.paths import workspace_root, package_sources
ROOT = workspace_root()

import numpy as np
from harness.research.free_research_driver import run_free_research
from harness.research.research_ledger import round_summary
from harness.adapters.research.native_research import prepare, measure, exploration_diagnostics, export_release, NativeExecutor, INSTRUCTION, DEFAULT_SOURCE
from harness.learning.inference.complete_checkpoint import load_checkpoint
from harness.adapters.providers.research_broker import AuthorBroker
from harness.engine.research_recovery import RecoverableBroker, file_identity, verify_files, freeze_invocation
from harness.learning.models.official_rul_cpu import _trainer
from harness.learning.training.continuation_recipes import reference_programs, gas_reference_programs, bearing_reference_programs, starting_reference
from harness.adapters.storage.local import atomic_json
from harness.adapters.tracking.native_trace import save_stage_trace, emit_trace

CURRENT_MODELS = ROOT / 'engineering/configs/tasks/research-parent-models.json'


def current_parent(task):
    """读取历史研究协议的固定父模型；它不是最新比赛推荐模型清单。"""
    selected = json.loads(CURRENT_MODELS.read_bytes())['tasks'][task]
    path = (ROOT / selected['checkpoint']).resolve()
    if not path.is_relative_to(ROOT) or sha256(path.read_bytes()).hexdigest() != selected['sha256']:
        raise ValueError('当前完整模型缺失或内容已改变')
    return path


def credential():
    """Windows读取当前用户配置，避免沿用换号前的进程环境；其他主机用运行时变量。"""
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
                return winreg.QueryValueEx(key, 'PYTRIO_API_KEY')[0]
        except FileNotFoundError:
            pass
    return os.environ.get('PYTRIO_API_KEY')


async def run(task, output, *, driver='queue', calls=4, stages=1, model='Qwen/Qwen3.8-27B',
              resume=False, data_root=ROOT / 'data/official', parent_checkpoint=None,
              broker_factory=None, stage_hook=None, candidate_cache=True, prepared=None, history=(), max_epochs=60,
              program_seconds=90):
    output = Path(output).resolve()
    if prepared is not None:
        task_dir, oracle = prepared
    elif resume:
        task_dir = output / 'task'
        oracle = json.loads((output / 'oracle.json').read_bytes())
    else:
        task_dir, oracle = prepare(task, output, data_root=data_root, parent_checkpoint=parent_checkpoint)
    source_files = package_sources()
    inputs = source_files + list(task_dir.iterdir()) + [output / name for name in (
        'oracle.json', 'service-inputs.npz', 'initial-predictions.npz', 'identity.json')]
    if parent_checkpoint is not None:
        inputs.append(Path(parent_checkpoint))
    initial_specification = json.loads((task_dir/'task.json').read_bytes())
    if initial_specification.get('tabular_resource'):
        inputs.append(Path(initial_specification['tabular_resource']['checkpoint']))
    freeze_invocation(output / 'invocation.json', {'task': task, 'driver': driver, 'calls': calls,
                      'stages': stages, 'model': model, 'max_epochs': max_epochs, 'program_seconds':program_seconds,
                      'candidate_cache': candidate_cache, 'data_root': str(Path(data_root).resolve()),
                      'parent_checkpoint': str(Path(parent_checkpoint).resolve()) if parent_checkpoint else None}, inputs, resume=resume)
    if resume and (output / 'completion.json').exists():
        verify_files(json.loads((output / 'completion.json').read_bytes())['artifacts'])
        return json.loads((output / 'run.json').read_bytes())
    specification = json.loads((task_dir / 'task.json').read_bytes())
    parent = task_dir / 'initial.npz'
    with np.load(output / 'initial-predictions.npz', allow_pickle=False) as arrays:
        current = {name: arrays[name].copy() for name in arrays.files}
    with np.load(output / 'service-inputs.npz', allow_pickle=False) as arrays:
        service_x = arrays['x'].copy()
    baseline_service = current['serve'].copy()
    references = reference_programs(task)
    if task == 'engine':
        references['源工况完整父版续训'] = 'def train(session):\n    session.build_regression_components((21,22,23),(0,2,0),epochs=60,head_epochs=12)\n'
        references['源工况组件后均衡续训'] = '''import numpy as np
def train(session):
    session.build_regression_components((21,22,23),(0,2,0),epochs=52,head_epochs=12)
    groups = np.asarray([str(group) for group in session.groups])
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    for _ in range(8):
        session.train_epoch(learning_rate=0.001, batch_size=128, sample_weights=weights)
'''
    rounds = []
    issued = []
    indices = np.array_split(np.arange(len(service_x)), stages)
    for stage in range(stages):
        folder = output / f'stage-{stage + 1}'
        if resume and (folder / 'completed.json').exists():
            saved = json.loads((folder / 'completed.json').read_bytes())
            verify_files(saved['artifacts'])
            if saved['parent_sha256'] != sha256(parent.read_bytes()).hexdigest():
                raise ValueError('恢复阶段接续了不同父版')
            event = json.loads((folder / 'decision.json').read_bytes())
            parent = Path(event['published_checkpoint'])
            if event['accepted']:
                with np.load(parent.parent / 'predictions.npz', allow_pickle=False) as arrays:
                    current['explore'], current['adopt'] = arrays['explore'].copy(), arrays['confirm'].copy()
            with np.load(folder / 'issued.npz', allow_pickle=False) as arrays:
                if not np.array_equal(arrays['indices'], indices[stage]):
                    raise ValueError('恢复服务顺序变化')
                issued.append(arrays['prediction'].copy())
            rounds.append(event)
            if (folder / 'trace.json').exists():
                emit_trace(json.loads((folder / 'trace.json').read_bytes()))
            continue
        from harness.engine import run_round
        session = round_session(task_dir, driver, model, broker_factory, rounds=list(history)+rounds,
                                max_epochs=max_epochs, parent=parent,program_seconds=program_seconds)
        parent, current, event, prediction = await run_round(session, task=task, task_dir=task_dir,
            specification=specification, output=output, folder=folder, parent=parent, current=current,
            oracle=oracle, references=references, driver=driver, model=model, calls=calls, stage=stage,
            indices=indices, service_x=service_x, baseline_service=baseline_service, resume=resume,
            candidate_cache=candidate_cache, stage_hook=stage_hook)
        rounds.append(event)
        issued.append(prediction)
        atomic_json(output/'run.json', {'task':task,'stages':rounds,'status':'running'})
    actual = np.concatenate(issued)
    scores = ({'baseline': measure(task, oracle['serve']['y'], baseline_service),
               'actual': measure(task, oracle['serve']['y'], actual)} if oracle['serve']['y'] is not None else None)
    np.savez_compressed(output / 'issued.npz', actual=actual, baseline=baseline_service)
    report = {'task': task, 'scope': ('官方源训练权限下接续实际父模型，按全部公开验证选案并独立交付'
        if specification.get('selection_policy') == 'public-validation' else
        '官方源训练权限下接续实际父模型，按完整预测链进行探索、确认和服务'),
              'stages': rounds, 'scores': scores, 'status': 'completed',
              'final_checkpoint': str(parent), 'driver_completed_stages': sum(r['error'] is None for r in rounds)}
    report['release'] = export_release(task_dir, parent, output / 'release')
    atomic_json(output / 'run.json', report)
    completion_files = [output / 'run.json', output / 'issued.npz']
    completion_files += [p for p in (output / 'release').rglob('*') if p.is_file()]
    completion_files += [p for stage_dir in output.glob('stage-*') for p in stage_dir.rglob('*') if p.is_file()]
    completion_files += [p for p in (output / 'candidate-cache').glob('*.json')]
    atomic_json(output / 'completion.json', {'artifacts': file_identity(completion_files)})
    print(json.dumps({'task': task, 'scores': scores, 'driver_completed_stages': report['driver_completed_stages']}, ensure_ascii=False), flush=True)
    return report


def round_session(task_dir, driver, model, broker_factory=None, *, rounds=(), max_epochs=60, parent=None,
                  program_seconds=90):
    """唯一具体装配点；测试注入 broker，实际执行与采用流程保持相同。"""
    from harness.engine import RoundSession
    from harness.adapters.datasets.baseline_sources import source_package,trainer_for
    from harness.learning.inference import complete_checkpoint
    specification=json.loads((Path(task_dir)/'task.json').read_bytes())
    task = specification['task']
    def make_broker(folder, calls):
        if broker_factory is not None: broker=broker_factory(folder,calls)
        elif driver=='queue': broker=AuthorBroker(folder/'queue',max_calls=calls,timeout=200)
        elif driver=='service':
            from harness.adapters.providers.pytrio_planner import PyTrioConfig
            from harness.adapters.providers.service_transport import HttpChatTransport
            from harness.adapters.providers.service_research_broker import ServiceResearchBroker
            config=PyTrioConfig(model=model,max_tokens=8192,timeout_seconds=600,raw_completion=True)
            broker=ServiceResearchBroker(folder/'queue',config,HttpChatTransport(config,credential(),allow_network=True),max_calls=calls)
        else: raise ValueError('未知驱动，工程验证须显式注入 broker')
        return RecoverableBroker(broker,folder/'exchanges')
    def predict(parent, values, folder):
        with source_package(specification) as package, TemporaryDirectory(dir=folder,prefix='temporary-') as temp:
            with trainer_for(package,Path(temp),specification['profile']) as trainer:
                fitted=load_checkpoint(parent,trainer.Model)
                return np.asarray([fitted.predict(x[None,:])[0] for x in values])
    history=[]
    for r in rounds:
        history.append({'stage':r['stage'],'accepted':r['accepted'],'error':r['error'],
            'platform_stage_id':r.get('platform_stage_id'), 'research':r.get('research'),
            'data_identity':r.get('data_identity'), 'selection':r.get('selection'),
            'selected_source_sha256':r['selected'].get('source_sha256'),
            'selected_exploration_loss':r['selected']['score'],'published_sha256':r['published_sha256']})
    complete=complete_checkpoint.is_complete(parent or Path(task_dir)/'initial.npz')
    with np.load(parent or Path(task_dir)/'initial.npz', allow_pickle=False) as stored:
        complete_kind=json.loads(str(stored['complete_state'])).get('kind') if complete else None
        residual_resumable = all(name in stored.files for name in (
            'residual_random_state', 'residual_optimizer_0_weight_exp_avg',
            'residual_optimizer_4_bias_exp_avg_sq', 'residual_optimizer_4_bias_step'))
    classifier_backend=None;classifier_parent_weight=.25
    if task=='gas' and specification.get('tabular_resource'):
        with np.load(specification['tabular_resource']['checkpoint'],allow_pickle=False) as resource:
            classifier_backend=('tabpfn' if 'tabpfn_classifier_state' in resource.files else
                                'tabicl' if 'tabular_classifier_state' in resource.files else None)
            if classifier_backend=='tabpfn':
                descriptor=json.loads(str(resource['tabpfn_classifier_state']))
                classifier_parent_weight=float(descriptor.get('recommended_parent_weight',.25))
                if not np.isfinite(classifier_parent_weight) or not 0<classifier_parent_weight<1:
                    raise ValueError('分类教师的预定父组合比例无效')
        if classifier_backend is None:raise ValueError('未知分类共享资源')
    regressor_backend=None;regressor_parent_weight=.25
    if task=='bearing' and specification.get('tabular_resource'):
        with np.load(specification['tabular_resource']['checkpoint'],allow_pickle=False) as resource:
            regressor_backend=('tabpfn' if 'tabpfn_regressor_state' in resource.files else
                               'tabicl' if 'tabular_readout_state' in resource.files else None)
            if regressor_backend=='tabpfn':
                descriptor=json.loads(str(resource['tabpfn_regressor_state']))
                regressor_parent_weight=float(descriptor.get('recommended_parent_weight',.25))
                if not np.isfinite(regressor_parent_weight) or not 0<regressor_parent_weight<1:
                    raise ValueError('回归教师的预定父组合比例无效')
        if regressor_backend is None:raise ValueError('未知回归共享资源')
    capabilities={'trainable':['w1','b1','w2','b2'],'max_epochs':max_epochs,'program_seconds':program_seconds,
        'tabicl_source_component':task=='bearing' and regressor_backend=='tabicl' and not complete,
        'tabpfn_regression_component':task=='bearing' and regressor_backend=='tabpfn' and not complete,
        'tabpfn_regression_parent':task=='bearing' and complete_kind=='tabpfn-regression-parent',
        'regressor_parent_weight':regressor_parent_weight,
        'tabicl_classification_component':task=='gas' and classifier_backend=='tabicl' and complete_kind in (None,'gas-source-vote'),
        'tabicl_classification_parent':task=='gas' and complete_kind=='tabicl-classification-parent',
        'tabpfn_classification_component':task=='gas' and classifier_backend=='tabpfn' and complete_kind in (None,'gas-source-vote'),
        'tabpfn_classification_parent':task=='gas' and complete_kind=='tabpfn-classification-parent',
        'classifier_parent_weight':classifier_parent_weight,
        'epoch_budget_scope':'本次实验配置，可在执行时限内调整；不是赛方统一60轮上限',
        'gas_source_repair':task == 'gas' and not complete,
        'gas_readout_choices':['parent'] if task == 'gas' and complete_kind=='gas-source-vote' else [],
        'build_components_before_first_epoch':not complete,
        'existing_auxiliary_components':'residual_resumable' if residual_resumable else 'frozen' if complete else None,
        'residual_training':{'available':residual_resumable,'blocks':['readout','all'],
            'losses':['huber','scaled_squared'],'max_head_epochs':12,'normalization_and_conditions':'frozen'},
        'component_build_consumes_parent_epochs':True,'structure_expansion_before_first_epoch_only':True}
    return RoundSession(NativeExecutor,make_broker,run_free_research,measure,exploration_diagnostics,
        predict,save_stage_trace,emit_trace,Path(complete_checkpoint.__file__).read_text(encoding='utf-8'),
        INSTRUCTION,history,capabilities,max_epochs,
        gas_reference_programs if task=='gas' else bearing_reference_programs if task=='bearing' else None,
        round_summary,starting_reference)
