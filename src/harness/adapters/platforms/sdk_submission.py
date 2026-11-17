"""workspace.v2边界：平台身份、测量和选择为准，训练复用既有内核。"""
import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import tomllib

from harness.adapters.storage.local import atomic_json
from .sdk_artifacts import PREDICT_ARGV, check_files, pack_candidate
from .sdk_gateway import broker_factory, model_options

TASKS = ('classification', 'regression')
_prediction_model = None
_prediction_path = None


def _implementation():
    """源码运行优先读取同目录发布版本；安装包回退到其真实发行元数据。"""
    for directory in Path(__file__).resolve().parents:
        metadata = directory/'pyproject.toml'
        if metadata.is_file():
            project = tomllib.loads(metadata.read_text(encoding='utf-8')).get('project', {})
            if project.get('name') == 'zhixun-harness':
                return 'zhixun-harness-'+project['version']
    from importlib.metadata import version
    return 'zhixun-harness-'+version('zhixun-harness')


def _workspace():
    work = Path(os.environ.get('CONTEST_WORK', '/workspace')).resolve()
    work.mkdir(parents=True, exist_ok=True)
    temporary = work/'tmp'
    temporary.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = os.environ['TEMP'] = os.environ['TMP'] = str(temporary)
    os.environ['SKB_DATA_DIRECTORY'] = str(temporary/'skrub')
    os.environ['MPLCONFIGDIR'] = str(temporary/'matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    tempfile.tempdir = str(temporary)
    return work


def _checkpoint(rnd):
    declared = rnd.parent.descriptor['checkpoint_paths']
    name = 'checkpoint.npz' if 'checkpoint.npz' in declared else declared[0] if len(declared) == 1 else None
    if not name:
        raise ValueError('父产物没有唯一可识别的检查点')
    path = rnd.parent.path(name)
    if sha256(path.read_bytes()).hexdigest() != rnd.parent.checkpoint_sha256.removeprefix('sha256:'):
        raise ValueError('父检查点与start摘要不一致')
    return path


def _limit(name, default, low, high):
    value = int(os.environ.get(name, default))
    if not low <= value <= high:
        raise ValueError(name+'超出本地执行器支持范围')
    return value


def _choose(rnd, choices, history, configuration):
    """平台测量返回之后再取得明确选择，格式错误只允许一次带反馈修正。"""
    available = {item['artifact_id'] for item in choices}
    request = {'action': 'choose_platform_artifact', 'task': rnd.task,
               'parent_artifact_id': rnd.parent.artifact_id, 'artifacts': choices,
               'history': history, 'configuration': configuration,
               'required_output': {'selected_artifact_id': '以上某个实际artifact_id',
                                   'reason': '依据平台测量和本轮真实改动说明选择'}}
    messages = [{'role': 'user', 'content': json.dumps(request, ensure_ascii=False)}]
    for attempt in range(2):
        receipt = rnd.model(messages, **model_options(rnd.model_name, {'temperature': 0}))
        try:
            decision = json.loads(receipt.text)
            if (not isinstance(decision, dict) or decision.get('selected_artifact_id') not in available
                    or not isinstance(decision.get('reason'), str) or not decision['reason'].strip()
                    or len(decision['reason'].encode('utf-8')) > 4000):
                raise ValueError('选择必须引用已登记产物并给出非空理由')
            return decision, receipt.call_id
        except (ValueError, TypeError) as exc:
            messages.extend([{'role': 'assistant', 'content': receipt.text},
                             {'role': 'user', 'content': f'选择格式无效：{exc}。只返回指定JSON，不补造产物ID。'}])
    raise ValueError('两次平台选择均无效，本轮不伪造决定或调用号')


def _native_round(rnd, parent, output, state):
    from .platform_train import run_platform
    from .shared_components import default_program_seconds, registered_tabular_resource
    profile = rnd.dataset_key
    expected_task = 'classification' if profile == 'A' else 'regression'
    if rnd.task.get('task') != expected_task:
        raise ValueError('dataset_key与冻结任务类型不一致')
    epochs = _limit('HARNESS_MAX_EPOCHS', 60, 1, 10000)
    calls = _limit('HARNESS_RESEARCH_CALLS', 3, 1, 16)
    resource = registered_tabular_resource(profile, parent, rnd.data.train_path) if profile == 'A' or epochs >= 25 else None
    default_seconds = default_program_seconds(parent, resource)
    seconds = _limit('HARNESS_PROGRAM_SECONDS', default_seconds, 1, 3600)
    if rnd.wall_seconds is not None:
        # 给宿主测量与结束回执留出时间；实际外部硬限额仍由平台执行。
        seconds = min(seconds, max(1, int(float(rnd.wall_seconds) * .7)))
    return asyncio.run(run_platform(profile, rnd.data.train_path, rnd.data.validation_path, parent, output,
        stage_id=f'sdk-{profile}-{rnd.round_index}', state_dir=state, calls=calls, max_epochs=epochs,
        driver='service', broker_factory=broker_factory(rnd), model_name=rnd.model_name,
        selection_policy='public-validation', tabular_resource=resource, program_seconds=seconds))


def train(rnd):
    work = _workspace()
    parent = _checkpoint(rnd)
    history = rnd.history()
    restored = rnd.state_workspace(work/'restored-state')
    native_state = work/'native-state'
    native_state.mkdir(exist_ok=False)
    history_name = f'history-{rnd.dataset_key}.json'
    if (restored/history_name).is_file():
        shutil.copyfile(restored/history_name, native_state/history_name)
    started = time.monotonic()
    release = work/'native-release'
    publication = _native_round(rnd, parent, release, native_state)
    report = json.loads((release/'training.json').read_bytes())
    checkpoint = release/'checkpoint.npz'
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    parent_digest = sha256(parent.read_bytes()).hexdigest()
    configuration = {'implementation': _implementation(), 'sdk': '1.0.2',
                     'dataset_key': rnd.dataset_key, 'round_index': rnd.round_index,
                     'parent_checkpoint_sha256': parent_digest,
                     'training_epochs_executed': report['executed_parent_epochs'],
                     'initial_steps': report['initial_steps'], 'final_steps': report['final_steps']}
    hypothesis = '沿平台真实父状态复用现有训练方法，以公开反馈选择可交付改进。'
    change = f"真实执行{report['executed_parent_epochs']}个父训练轮次；本地选中检查点{digest}。"
    choices = [{'artifact_id': rnd.parent.artifact_id, 'role': 'parent',
                'summary': rnd.parent.receipt.get('summary', {}), 'checkpoint_sha256': parent_digest}]
    if digest != parent_digest:
        packed = pack_candidate(checkpoint, work/'candidate', parent_sha256=parent_digest,
            training_report={k: report[k] for k in ('initial_steps', 'final_steps', 'executed_parent_epochs')})
        reply = rnd.submit_candidate(work/'candidate', hypothesis=hypothesis, change_summary=change,
            configuration=configuration, seed=int(report['seed']), checkpoint_paths=packed['checkpoint_paths'],
            code_paths=packed['code_paths'], predict_argv=PREDICT_ARGV)
        choices.append({'artifact_id': reply['artifact_id'], 'role': 'new_candidate',
                        'summary': reply.get('summary', {}), 'checkpoint_sha256': digest})
        atomic_json(work/'candidate-receipt.json', reply)
    for item in rnd.artifacts:
        if item['artifact_id'] not in {choice['artifact_id'] for choice in choices}:
            choices.append({'artifact_id': item['artifact_id'], 'role': 'historical',
                            'summary': item.get('summary', {})})
    decision, final_call_id = _choose(rnd, choices, history, configuration)
    selected_id = decision['selected_artifact_id']
    selected = next(item for item in choices if item['artifact_id'] == selected_id)
    if selected['role'] == 'historical':
        path = rnd.restore_artifact(selected_id, work/'selected-history')
        descriptor = next(item['artifact'] for item in rnd.artifacts if item['artifact_id'] == selected_id)
        names = descriptor['checkpoint_paths']
        selected_path = path/('checkpoint.npz' if 'checkpoint.npz' in names else names[0])
        selected['checkpoint_sha256'] = sha256(selected_path.read_bytes()).hexdigest()
    saved = work/'next-state'
    saved.mkdir()
    # 下轮只需要紧凑历史，训练沙箱、教师缓存和重复权重不进入平台状态包。
    native_history = json.loads((native_state/history_name).read_bytes())
    events = native_history['events']
    compact = []
    for event in events:
        compact.append({k: event.get(k) for k in ('stage', 'accepted', 'error', 'platform_stage_id',
            'data_identity', 'selection', 'published_sha256', 'round_id', 'actual_parent_sha256')})
        compact[-1]['selected'] = {k: (event.get('selected') or {}).get(k) for k in ('score', 'source_sha256')}
    if compact:
        compact[-1].update(accepted=selected_id != rnd.parent.artifact_id,
                           published_sha256=selected['checkpoint_sha256'],
                           platform_artifact_id=selected_id, selection=None)
        if selected['checkpoint_sha256'] != digest:
            # 平台最后选择可能保父或回退；本地候选成绩不能冒充实际选中模型的成绩。
            compact[-1]['selected'] = {'score': None, 'source_sha256': None}
    atomic_json(saved/history_name, {'events': compact})
    atomic_json(saved/'last-selection.json', {'artifact_id': selected_id, 'checkpoint_sha256': selected['checkpoint_sha256'],
        'model_call_id': final_call_id, 'reason': decision['reason'], 'round_index': rnd.round_index})
    check_files(saved)
    rnd.save_state(saved)
    receipt = rnd.finish(selected_artifact_id=selected_id, hypothesis=hypothesis, change_summary=change,
        reason=decision['reason'], configuration=configuration, seed=int(report['seed']),
        history_ids=[item['id'] for item in history], model_call_ids=list(rnd.call_ids))
    atomic_json(work/'sdk-result.json', {'ack': receipt, 'selected_artifact_id': selected_id,
        'checkpoint_sha256': selected['checkpoint_sha256'], 'model_call_ids': list(rnd.call_ids),
        'seconds': time.monotonic()-started, 'local_nomination_sha256': publication['published_sha256']})


def predict(sample, model_dir):
    global _prediction_model, _prediction_path
    _workspace()
    root = Path(model_dir).resolve()
    path = Path(os.environ.get('CONTEST_ARTIFACT_PATH', str(root/'checkpoint.npz'))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('预测检查点必须来自本次/model产物')
    if _prediction_path != path:
        from .baselines.model import Model
        _prediction_model = Model.load(path)
        _prediction_path = path
    if sample.task != _prediction_model.meta['task']:
        raise ValueError('样本任务与实际模型不一致')
    if sample.dataset_key and sample.dataset_key != _prediction_model.meta.get('profile'):
        raise ValueError('样本数据集与实际模型不一致')
    from .baselines.predict import predict as native_predict
    return native_predict(_prediction_model, sample.inputs)
