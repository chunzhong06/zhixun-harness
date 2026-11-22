"""官方父检查点上运行真实源码研究；模型调用由调用方显式注入。"""
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
from harness.evaluation.native import measure, exploration_diagnostics
from harness.learning.artifacts.native import export_release
from harness.adapters.executors.native import NativeExecutor
from harness.research.native_instruction import INSTRUCTION, DEFAULT_SOURCE
from harness.learning.inference.complete_checkpoint import load_checkpoint
from harness.adapters.providers.research_broker import AuthorBroker
from harness.engine.research_recovery import RecoverableBroker, file_identity, verify_files, freeze_invocation
from harness.learning.models.official_rul_cpu import _trainer
from harness.learning.training.continuation_recipes import (
    ENGINE_UNIFORM, reference_programs, gas_reference_programs, bearing_reference_programs, engine_reference_programs, starting_reference)
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


async def run(task, output, *, driver='queue', calls=4, stages=1, model='Qwen/Qwen3.8-27B',
              resume=False, data_root=ROOT / 'data/official', parent_checkpoint=None,
              broker_factory=None, stage_hook=None, candidate_cache=True, prepared=None, history=(), max_epochs=60,
              program_seconds=90):
    output = Path(output).resolve()
    if prepared is None:
        if resume:
            prepared = output/'task', json.loads((output/'oracle.json').read_bytes())
        else:
            from harness.adapters.datasets.native import prepare
            prepared = prepare(task, output, data_root=data_root, parent_checkpoint=parent_checkpoint)
    return await run_prepared(task, output, prepared=prepared, parent_checkpoint=parent_checkpoint,
        driver=driver, calls=calls, stages=stages, model=model, resume=resume, data_root=data_root,
        broker_factory=broker_factory, stage_hook=stage_hook, candidate_cache=candidate_cache,
        history=history, max_epochs=max_epochs, program_seconds=program_seconds)


async def run_prepared(task, output, *, prepared, parent_checkpoint, driver, calls, stages, model,
                       resume=False, data_root=None, broker_factory=None, stage_hook=None,
                       candidate_cache=True, history=(), max_epochs=60, program_seconds=90):
    """运行已经准备好的当轮输入。平台入口只调用这里，不会回退到研究数据 ZIP。"""
    output = Path(output).resolve()
    task_dir, oracle = prepared
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
                      'candidate_cache': candidate_cache, 'data_root': str(Path(data_root).resolve()) if data_root is not None else None,
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
        if max_epochs >= 4:
            references['发动机按行等权续训四轮'] = ENGINE_UNIFORM
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
            raise ValueError('service驱动必须注入平台宿主的rnd.model网关')
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
            'reference_observation':r.get('reference_observation'),
            'selected_source_sha256':r['selected'].get('source_sha256'),
            'selected_exploration_loss':r['selected']['score'],'published_sha256':r['published_sha256']})
    complete=complete_checkpoint.is_complete(parent or Path(task_dir)/'initial.npz')
    with np.load(parent or Path(task_dir)/'initial.npz', allow_pickle=False) as stored:
        complete_state=json.loads(str(stored['complete_state'])) if complete else {}
        complete_kind=complete_state.get('kind')
        tabm_present='tabm_readout' in stored.files
        baseline_context_present='baseline_context_x' in stored.files
        drift_present='drift_readout' in complete_state
        parent_meta=json.loads(str(stored['metadata']))
        parent_classes=parent_meta.get('classes',[])
        gas_estimators=(json.loads(str(stored['tabpfn_classifier_state'])).get('n_estimators')
                        if 'tabpfn_classifier_state' in stored.files else None)
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
                gas_estimators=descriptor.get('n_estimators')
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
    can_fit_tabm = False
    if task=='bearing' and complete_kind=='tabpfn-regression-parent' and not tabm_present and max_epochs>=7:
        from importlib.util import find_spec
        if find_spec('tabm') and find_spec('rtdl_num_embeddings'):
            with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
                x, y = batch['x'], batch['y']
                can_fit_tabm = len(y)>=16 and np.ptp(y)>0 and np.any(np.ptp(x.astype(np.float32),axis=0)>0)
    can_fit_baseline_context=False
    if (task=='gas' and not baseline_context_present and gas_estimators in (4, 8) and
            (complete_kind=='tabpfn-classification-parent' or classifier_backend=='tabpfn')):
        from harness.learning.inference.tabpfn_classifier import baseline_context_available
        with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            can_fit_baseline_context=baseline_context_available(batch['x'],batch['y'])
    can_fit_drift=False
    if task=='gas' and not drift_present and (baseline_context_present or can_fit_baseline_context):
        from harness.learning.inference.gas_drift_readout import fit_readout
        with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            # 用当前源训练判断候选能否构建。数值执行器在隔离进程中重建它。
            can_fit_drift=fit_readout(batch['x'],batch['y'],parent_classes) is not None
    can_fit_toluene = False
    if task=='gas' and drift_present and 'toluene_readout' not in complete_state:
        from harness.learning.inference.gas_toluene_readout import available
        with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            can_fit_toluene=available(batch['x'],batch['y'],parent_classes)
    can_fit_kinetics=False
    if (task=='gas' and 'kinetic_readout' not in complete_state
            and (can_fit_toluene or 'toluene_readout' in complete_state)):
        from harness.learning.inference.gas_kinetic_support import available
        with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            can_fit_kinetics=available(batch['x'],batch['y'],parent_classes)
    can_fit_density=False
    can_fit_device = False
    if task=='gas' and complete_kind == 'tabpfn-classification-parent':
        from harness.learning.inference.gas_device_context import prepare as prepare_device
        with np.load(parent or Path(task_dir)/'initial.npz',allow_pickle=False) as stored, \
             np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            can_fit_device = prepare_device(batch['x'], batch['y'], stored['classifier_source_x'],
                                            parent_classes, complete_state.get('device_context')) is not None
    after_uniform=False
    if task=='engine' and residual_resumable and complete_state.get('condition') and 'engine_condition_density' not in complete_state:
        from harness.learning.inference.engine_condition_density import available
        with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            can_fit_density=available(batch['x'],batch['y'],complete_state['condition'])
        if can_fit_density and history:
            parent_id=sha256(Path(parent or Path(task_dir)/'initial.npz').read_bytes()).hexdigest()
            uniform_id=sha256(ENGINE_UNIFORM.encode()).hexdigest()
            after_uniform=any(r['accepted'] and r['published_sha256']==parent_id and r['selected_source_sha256']==uniform_id for r in history)
    can_fit_neighbors=False
    if task=='engine' and residual_resumable and 'engine_condition_density' in complete_state and 'engine_neighbors' not in complete_state:
        from harness.learning.inference.engine_neighbors import usable_modes
        with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            can_fit_neighbors=bool(usable_modes(batch['x'],batch['y'],complete_state['condition']))
    can_align_engine = False
    if task == 'engine':
        from harness.learning.inference.engine_setting_alignment import fit as fit_alignment, setting_columns
        if setting_columns(parent_meta['paths']) is not None:
            from harness.adapters.platforms.baselines.core_model import Model
            fitted = complete_checkpoint.load_checkpoint(parent or Path(task_dir)/'initial.npz', Model)
            with np.load(Path(task_dir)/'train.npz', allow_pickle=False) as batch:
                base = getattr(fitted, 'unaligned_engine', fitted)
                can_align_engine = fit_alignment(base, batch['x'], batch['y'], parent_meta.get('engine_input_alignment', [])) is not None
    cap_available=False
    bearing_grid_available = False
    if task=='bearing' and (not complete or complete_kind in ('tabpfn-regression-parent', 'bearing-native-grid-parent')):
        from harness.learning.inference.bearing_grid_readout import checkpoint_bins, new_groups
        with np.load(parent or Path(task_dir)/'initial.npz',allow_pickle=False) as stored, \
             np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            paths = parent_meta['paths']
            source = stored['readout_source_x'] if complete_kind == 'tabpfn-regression-parent' else None
            bins = checkpoint_bins(stored['mean'], stored['scale'], paths) if source is None else None
            bearing_grid_available = bool(new_groups(batch['x'], batch['y'], source, paths,
                complete_state.get('bearing_grid_readout'), source_bins=bins))
    tabm_spec=complete_state.get('tabm_readout',{})
    if task=='bearing' and tabm_present and 'head_support' in tabm_spec and 'health_cap' not in tabm_spec:
        from harness.learning.inference.bearing_health_readout import available
        with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            cap_available=available(parent_meta['paths'],batch['y'])
    can_fit_source_repair=False
    can_fit_partial=False
    if task=='gas' and parent_meta.get('profile')=='A' and (not complete or 'gas_partial_context' in parent_meta):
        from harness.learning.inference.gas_partial_readout import prepare as prepare_partial
        from harness.adapters.platforms.baselines.core_model import Model
        fitted=complete_checkpoint.load_checkpoint(parent or Path(task_dir)/'initial.npz',Model)
        with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            prepared=prepare_partial(fitted,batch['x'],batch['y'])
            can_fit_partial=(prepared is not None and (prepared[-1] is None or prepared[-1]['source_sha256']!=prepared[3]))
    if (task=='gas' and not complete and len(parent_classes)==6 and len(parent_meta['paths'])==128
            and 'gas_pair_patch' not in parent_meta):
        from harness.learning.inference.gas_views import six_class_source
        with np.load(Path(task_dir)/'train.npz',allow_pickle=False) as batch:
            can_fit_source_repair=six_class_source(batch['y'])
    engine_limits = {}
    if task == 'engine' and complete_state.get('residual'):
        from harness.learning.training.engine_life_limit import source_limit
        with np.load(Path(task_dir)/'train.npz', allow_pickle=False) as batch:
            limit = source_limit(batch['y'])
        old_limit = complete_state.get('engine_life_limit')
        if complete_state.get('engine_parent_support'):
            enabled = complete_state['engine_parent_support'].get('quarter_pullback', False)
            engine_limits['engine_parent_pullback_action'] = 'remove' if enabled else 'apply'
        if old_limit is not None:
            engine_limits['engine_life_limit_removable'] = True
            engine_limits['comparison_reference'] = '发动机解除上限并续训'
        if limit is not None and limit != old_limit:
            engine_limits['engine_life_limit_available'] = True
            engine_limits['comparison_reference'] = '发动机源寿命上限与父版续训'
    capabilities={'trainable':['w1','b1','w2','b2'],'max_epochs':max_epochs,'program_seconds':program_seconds,
        **engine_limits,
        'bearing_health_cap_available':bool(cap_available),
        'bearing_grid_readout_available':bearing_grid_available,
        'engine_condition_density_available':can_fit_density,'engine_after_uniform':after_uniform,
        'engine_neighbors_available':can_fit_neighbors,
        'engine_setting_alignment_available':can_align_engine,
        'gas_toluene_readout_available':bool(can_fit_toluene),
        'gas_kinetic_support_available':bool(can_fit_kinetics),
        'gas_device_context_available':can_fit_device,
        'gas_drift_readout_available':can_fit_drift,
        'anonymous_engine_parent':task=='engine' and residual_resumable and
            complete_state.get('provenance',{}).get('anonymous_source_rows',0)>0,
        'gas_baseline_context_available':bool(can_fit_baseline_context),
        'tabm_readout_available':bool(can_fit_tabm),
        'tabm_source_support_available': (task == 'bearing' and complete_kind == 'tabpfn-regression-parent'
                                          and tabm_present and 'source_support' not in complete_state['tabm_readout']),
        'tabm_head_support_available': (task=='bearing' and complete_kind=='tabpfn-regression-parent'
                                      and tabm_present and 'source_support' in complete_state['tabm_readout']
                                      and 'head_support' not in complete_state['tabm_readout']),
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
        'gas_source_repair':can_fit_source_repair,
        'gas_partial_context_available':can_fit_partial,
        'gas_readout_choices':['parent'] if task == 'gas' and complete_kind=='gas-source-vote' else [],
        'build_components_before_first_epoch':not complete,
        'existing_auxiliary_components':'residual_resumable' if residual_resumable else 'frozen' if complete else None,
        'residual_training':{'available':residual_resumable,'blocks':['readout','all'],
            'losses':['huber','scaled_squared'],'max_head_epochs':12,'normalization_and_conditions':'frozen'},
        'component_build_consumes_parent_epochs':True,'structure_expansion_before_first_epoch_only':True}
    return RoundSession(NativeExecutor,make_broker,run_free_research,measure,exploration_diagnostics,
        predict,save_stage_trace,emit_trace,Path(complete_checkpoint.__file__).read_text(encoding='utf-8'),
        INSTRUCTION,history,capabilities,max_epochs,
        gas_reference_programs if task=='gas' else bearing_reference_programs if task=='bearing' else engine_reference_programs if task=='engine' else None,
        round_summary,starting_reference)
