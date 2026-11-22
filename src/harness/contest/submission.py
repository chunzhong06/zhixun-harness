"""workspace.v2边界：平台身份、测量和选择为准，训练复用既有内核。"""
import asyncio
import errno
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import tomllib

from harness.adapters.research.response_format import unwrap_json_fence
from harness.adapters.storage.local import atomic_json
from .artifacts import PREDICT_ARGV, check_files, pack_candidate
from .gateway import broker_factory, model_options

TASKS = ('classification', 'regression')
_prediction_model = None
_prediction_path = None
_prediction_started = None
_prediction_rows = 0
_prediction_samples = []
_prediction_total_seconds = 0.


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


def _workspace(*, prediction=False):
    work = Path(os.environ.get('CONTEST_WORK', '/workspace')).resolve()
    temporary = work/'tmp'
    try:
        temporary.mkdir(parents=True, exist_ok=True)
        if prediction:
            # 已经存在的只读 tmp 目录也会通过 mkdir(exist_ok=True)；以实际创建为准。
            private = tempfile.TemporaryDirectory(prefix='zhixun-predict-', dir=temporary)
    except OSError as exc:
        # 预测宿主只挂载选中模型时，作者运行库仍需官方可写的临时目录。
        if not prediction or exc.errno not in (errno.EROFS, errno.EACCES, errno.ENOENT, errno.EPERM,
                                               errno.ENOSPC, getattr(errno, 'EDQUOT', None)):
            raise
        private = tempfile.TemporaryDirectory(prefix='zhixun-predict-', dir='/tmp')
    if prediction:
        import atexit
        atexit.register(private.cleanup)
        temporary = Path(private.name)
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


def _training_enabled(profile, round_index):
    """显式轮次仍受尊重。C 默认根据本轮预算和已确认的发放状态判断。"""
    if profile not in ("A", "C"):
        return True
    name = {"A": "HARNESS_A_TRAIN_ROUNDS", "C": "HARNESS_C_TRAIN_ROUNDS"}[profile]
    if profile == "C" and name not in os.environ:
        return True
    raw = os.environ.get(name, "1,2")
    try:
        rounds = {int(value.strip()) for value in raw.split(",") if value.strip()}
    except ValueError as exc:
        raise ValueError(name + " 必须是逗号分隔的轮次") from exc
    if not rounds or any(not 1 <= value <= 10 for value in rounds):
        raise ValueError(name + " 只能包含 1 至 10")
    return round_index in rounds


def _choose(rnd, choices, history, configuration):
    """平台测量返回之后再取得明确选择，格式错误只允许一次带反馈修正。"""
    from harness.adapters.platforms.baselines.data import json_value

    available = {item['artifact_id'] for item in choices}
    request = {'action': 'choose_platform_artifact', 'task': rnd.task,
               'parent_artifact_id': rnd.parent.artifact_id, 'artifacts': choices,
               'history': history, 'configuration': configuration,
               'required_output': {'selected_artifact_id': '以上某个实际artifact_id',
                                   'reason': '依据平台测量和本轮真实改动说明选择'}}
    messages = [{'role': 'user', 'content': json.dumps(request, ensure_ascii=False)}]
    from .round_budget import current
    budget = current.get()
    for attempt in range(2):
        if budget:
            if attempt and budget.remaining < budget.handoff:
                raise ValueError('选案无效，剩余时间不足以请求格式纠正')
            budget.marker('final_choice')
            started = budget.clock()
        receipt = rnd.model(messages, **model_options(rnd.model_name, {'temperature': 0}))
        if budget:
            budget.observe('model', budget.clock()-started)
        try:
            decision = json_value(unwrap_json_fence(receipt.text))
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
    from .training import run_platform
    from harness.learning.training.teacher_resources import default_program_seconds, registered_tabular_resource
    profile = rnd.dataset_key
    expected_task = 'classification' if profile == 'A' else 'regression'
    if rnd.task.get('task') != expected_task:
        raise ValueError('dataset_key与冻结任务类型不一致')
    epochs = _limit('HARNESS_MAX_EPOCHS', 60, 1, 10000)
    # 气体整轮的主要开销是宿主逐行测量。先保留一次新实验和提名，再由宿主测量后选案。
    calls = _limit('HARNESS_RESEARCH_CALLS', 2 if profile == 'A' else 3, 1, 16)
    resource = registered_tabular_resource(profile, parent, rnd.data.train_path) if profile == 'A' or epochs >= 25 else None
    default_seconds = default_program_seconds(parent, resource)
    seconds = _limit('HARNESS_PROGRAM_SECONDS', default_seconds, 1, 3600)
    from .round_budget import current, RoundBudget, CloseResearch
    budget = current.get() or RoundBudget(profile,rnd.wall_seconds)
    slots = 2 + calls if profile == 'A' else 1
    seconds = min(seconds, max(1,int(budget.seconds * .7 / slots)))
    if profile == 'A':
        from .round_budget import stable_gas_parent
        if stable_gas_parent(rnd, state):
            budget.defer_gas_reference = True
            epochs = min(epochs, 4)
            seconds = min(seconds, 60)
            budget.marker('gas_short_continuation')
        else:
            # 首轮或数据变化时保留完整模型及研究选案，限制单个本地候选占用的墙钟时间。
            seconds = min(seconds, 180)
    budget.program_limit = seconds
    if not budget.allowance(seconds):
        raise CloseResearch('当前时间不足以再开一次候选训练')
    previous_threads = os.environ.get('ZHIXUN_SDK_PREDICT_THREADS')
    if profile in ('A', 'B'):
        os.environ['ZHIXUN_SDK_PREDICT_THREADS'] = str(_prediction_threads(maximum=4 if profile == 'A' else 2))
    try:
        return asyncio.run(run_platform(profile, rnd.data.train_path, rnd.data.validation_path, parent, output,
            stage_id=f'sdk-{profile}-{rnd.round_index}', state_dir=state, calls=calls, max_epochs=epochs,
            driver='service', broker_factory=broker_factory(rnd), model_name=rnd.model_name,
            selection_policy='public-validation', tabular_resource=resource, program_seconds=seconds,
            final_choice_at_host=True))
    finally:
        if profile in ('A', 'B'):
            if previous_threads is None:
                os.environ.pop('ZHIXUN_SDK_PREDICT_THREADS', None)
            else:
                os.environ['ZHIXUN_SDK_PREDICT_THREADS'] = previous_threads


def _training_workspace(root):
    """首次使用原有输出路径；并发或恢复运行另外申请空目录，保留上一轮文件。"""
    outputs = ('restored-state', 'native-release', 'candidate', 'candidate-receipt.json',
               'next-state', 'selected-history', 'sdk-result.json')
    if not any(os.path.lexists(root / name) for name in outputs):
        try:
            (root / 'native-state').mkdir()
            return root
        except FileExistsError:
            pass
    work = Path(tempfile.mkdtemp(prefix='sdk-round-', dir=root))
    (work / 'native-state').mkdir()
    return work


def train(rnd):
    from .round_budget import RoundBudget
    if rnd.dataset_key == "C":
        from .engine_maintenance import EngineRoundBudget
        budget = EngineRoundBudget(rnd)
    else:
        budget = RoundBudget(rnd.dataset_key,rnd.wall_seconds)
    with budget.activate():
        return _train_timed(rnd,budget)


def _train_timed(rnd,budget):
    from contest_harness import SDK_VERSION
    from .round_budget import CloseResearch, retain_before_training
    work = _training_workspace(_workspace())
    parent = _checkpoint(rnd)
    history = rnd.history()
    restored = rnd.state_workspace(work/'restored-state')
    budget.restore(restored)
    if rnd.dataset_key == 'A':
        budget.forecast_gas_publication(rnd.data.validation_path, rnd.parent.receipt.get('summary'))
    native_state = work/'native-state'
    history_name = f'history-{rnd.dataset_key}.json'
    if (restored/history_name).is_file():
        shutil.copyfile(restored/history_name, native_state/history_name)
    if rnd.dataset_key == 'B' and (restored/'last-selection.json').is_file():
        from harness.learning.inference.teacher_cache import copy_training_hint
        try:
            previous = json.loads((restored/'last-selection.json').read_bytes())
        except (ValueError, OSError):
            previous = None
        if isinstance(previous, dict) and previous.get('checkpoint_sha256') == sha256(parent.read_bytes()).hexdigest():
            copy_training_hint(restored/'training-hint', native_state/'prior-training-hint')
    started = budget.start
    release = work/'native-release'
    checkpoint = release/'checkpoint.npz'
    try:
        if not _training_enabled(rnd.dataset_key, rnd.round_index):
            raise CloseResearch('训练策略：本轮默认保留父模型，仅完成选案、状态保存和交接')
        if not budget.allowance(_limit('HARNESS_PROGRAM_SECONDS',1800,1,3600)):
            raise CloseResearch('进入父版准备前已达到收尾窗口')
        publication = _native_round(rnd, parent, release, native_state)
    except CloseResearch as exc:
        budget.marker('retain_before_training')
        publication = retain_before_training(rnd,parent,release,native_state,exc)
        checkpoint = parent
    report = json.loads((release/'training.json').read_bytes())
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    parent_digest = sha256(parent.read_bytes()).hexdigest()
    configuration = {'implementation': _implementation(), 'sdk': SDK_VERSION,
                     'dataset_key': rnd.dataset_key, 'round_index': rnd.round_index,
                     'parent_checkpoint_sha256': parent_digest,
                     'training_epochs_executed': report['executed_parent_epochs'],
                     'initial_steps': report['initial_steps'], 'final_steps': report['final_steps']}
    if report.get('research_handoff'):
        configuration['research_handoff'] = report['research_handoff']
    hypothesis = '沿平台真实父状态复用现有训练方法，以公开反馈选择可交付改进。'
    change = f"真实执行{report['executed_parent_epochs']}个父训练轮次；本地选中检查点{digest}。"
    choices = [{'artifact_id': rnd.parent.artifact_id, 'role': 'parent',
                'summary': rnd.parent.receipt.get('summary', {}), 'checkpoint_sha256': parent_digest}]
    if digest != parent_digest and budget.can_publish():
        packed = pack_candidate(checkpoint, work/'candidate', parent_sha256=parent_digest,
            training_report={k: report[k] for k in ('initial_steps', 'final_steps', 'executed_parent_epochs')})
    if digest != parent_digest and budget.can_publish() and (work/'candidate').is_dir():
        budget.marker('submit_candidate')
        submit_started = budget.clock()
        reply = rnd.submit_candidate(work/'candidate', hypothesis=hypothesis, change_summary=change,
            configuration=configuration, seed=int(report['seed']), checkpoint_paths=packed['checkpoint_paths'],
            code_paths=packed['code_paths'], predict_argv=PREDICT_ARGV)
        budget.observe('submission',budget.clock()-submit_started)
        choices.append({'artifact_id': reply['artifact_id'], 'role': 'new_candidate',
                        'summary': reply.get('summary', {}), 'checkpoint_sha256': digest})
        atomic_json(work/'candidate-receipt.json', reply)
    elif digest != parent_digest:
        budget.marker('trained_candidate_not_submitted')
    for item in rnd.artifacts:
        if item['artifact_id'] not in {choice['artifact_id'] for choice in choices} and budget.can_restore(item):
            choices.append({'artifact_id': item['artifact_id'], 'role': 'historical',
                            'summary': item.get('summary', {})})
    configuration['time_budget'] = budget.snapshot()
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
    # 仅携带紧凑历史及可选的冻结教师训练输出，不携带验证数据、预测答案或重复权重。
    native_history = json.loads((native_state/history_name).read_bytes())
    events = native_history['events']
    compact = []
    for event in events:
        compact.append({k: event.get(k) for k in ('stage', 'accepted', 'error', 'platform_stage_id',
            'data_identity', 'selection', 'published_sha256', 'round_id', 'actual_parent_sha256', 'reference_observation')})
        compact[-1]['selected'] = {k: (event.get('selected') or {}).get(k) for k in ('score', 'source_sha256')}
    if compact:
        from .round_report import carry_recent_trial_feedback
        carry_recent_trial_feedback(compact, events, report, selected)
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
    training_hint_used = None
    if rnd.dataset_key == 'B':
        from harness.learning.inference.teacher_cache import copy_training_hint
        recording = Path(publication['recording_directory'])
        if copy_training_hint(recording/'.frozen-teacher'/parent_digest, saved/'training-hint'):
            try:
                check_files(saved)
            except ValueError:
                shutil.rmtree(saved/'training-hint')
        startup = recording/'training-cache-start.json'
        if startup.is_file():
            training_hint_used = json.loads(startup.read_bytes())['training_hit']
    if rnd.dataset_key == "C":
        budget.record_choice(report, selected)
    budget.save(saved)
    check_files(saved)
    budget.marker('save_state')
    rnd.save_state(saved)
    configuration['time_budget'] = budget.snapshot()
    receipt = rnd.finish(selected_artifact_id=selected_id, hypothesis=hypothesis, change_summary=change,
        reason=decision['reason'], configuration=configuration, seed=int(report['seed']),
        history_ids=[item['id'] for item in history], model_call_ids=list(rnd.call_ids))
    atomic_json(work/'sdk-result.json', {'ack': receipt, 'training_hint_used': training_hint_used,
        'selected_artifact_id': selected_id,
        'checkpoint_sha256': selected['checkpoint_sha256'], 'model_call_ids': list(rnd.call_ids),
        'seconds': time.monotonic()-started, 'local_nomination_sha256': publication['published_sha256']})


def _prediction_threads(cgroup_root=Path('/sys/fs/cgroup'), *, maximum=2):
    """按容器配额和进程亲和性选择已测过的线程数。无法读取限制时使用一线程。"""
    try:
        available = min(maximum, len(os.sched_getaffinity(0)))
        try:
            quota, period = (cgroup_root/'cpu.max').read_text().split()
        except FileNotFoundError:
            quota = (cgroup_root/'cpu/cpu.cfs_quota_us').read_text().strip()
            period = (cgroup_root/'cpu/cpu.cfs_period_us').read_text().strip()
        if int(period) <= 0:
            return 1
        if quota not in ('max', '-1'):
            available = min(available, int(quota) // int(period))
        return 4 if available >= 4 else 2 if available >= 2 else 1
    except (OSError, ValueError, AttributeError):
        return 1


def _record_prediction_timing(elapsed):
    """记录首行、累计和最近 20 行耗时；诊断写失败不影响正式预测。"""
    global _prediction_rows, _prediction_total_seconds
    if _prediction_started is None:
        return
    _prediction_samples.append(max(0., float(elapsed)))
    del _prediction_samples[:-20]
    _prediction_rows += 1
    _prediction_total_seconds += max(0., float(elapsed))
    if _prediction_rows != 1 and _prediction_rows % 100:
        return
    values = sorted(_prediction_samples)
    sample = {
        'rows': _prediction_rows,
        'cold_seconds': max(0., _prediction_started),
        'prediction_seconds': max(0., _prediction_total_seconds),
        'recent_mean_seconds': sum(values) / len(values),
        'recent_p95_seconds': values[min(len(values) - 1, int(len(values) * .95))],
        'cold_definition': 'Model.load wall time before the first row',
        'clock_basis': 'monotonic',
    }
    try:
        root = Path(os.environ.get('CONTEST_WORK', '/workspace'))
        atomic_json(root / 'predict-timing.json', sample)
    except (OSError, ValueError):
        pass


def predict(sample, model_dir):
    global _prediction_model, _prediction_path, _prediction_started
    global _prediction_rows, _prediction_samples, _prediction_total_seconds
    root = Path(model_dir).resolve()
    path = Path(os.environ.get('CONTEST_ARTIFACT_PATH', str(root/'checkpoint.npz'))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('预测检查点必须来自本次/model产物')
    if _prediction_path != path:
        _workspace(prediction=True)
        from harness.adapters.platforms.baselines.model import Model
        os.environ['ZHIXUN_SDK_PREDICT_THREADS'] = str(_prediction_threads(maximum=4 if sample.task == 'classification' else 2))
        loading = time.monotonic()
        _prediction_model = Model.load(path)
        _prediction_started = time.monotonic() - loading
        _prediction_rows = 0
        _prediction_samples = []
        _prediction_total_seconds = 0.
        _prediction_path = path
    if sample.task != _prediction_model.meta['task']:
        raise ValueError('样本任务与实际模型不一致')
    if sample.dataset_key and sample.dataset_key != _prediction_model.meta.get('profile'):
        raise ValueError('样本数据集与实际模型不一致')
    from harness.adapters.platforms.baselines.predict import predict as native_predict
    started = time.monotonic()
    result = native_predict(_prediction_model, sample.inputs)
    _record_prediction_timing(time.monotonic() - started)
    if sample.task == 'classification':
        _resolve_top_tie(result['outputs'])
    return result


def _resolve_top_tie(scores):
    """SDK 要求唯一最高分。精确并列时保留模型按类别顺序 argmax 的首项。"""
    top = max(item['score'] for item in scores)
    tied = [item for item in scores if item['score'] == top]
    if len(tied) > 1 and 0 < top <= 1:
        # 使用相邻可表示值，其他类别排序及原有胜出类别保持一致。
        if top < 1:
            tied[0]['score'] = math.nextafter(top, 1.)
        else:
            for item in tied[1:]:
                item['score'] = math.nextafter(top, 0.)
