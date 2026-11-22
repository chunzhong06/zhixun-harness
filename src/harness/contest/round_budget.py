"""原生 SDK 回合的时间准入。已发出的宿主请求由 SDK 独占，只有本地工作可中止。"""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from harness.adapters.storage.local import atomic_json


class CloseResearch(RuntimeError):
    """本轮需要转入选案，已有完整的数值结果可以继续使用。"""


class HostCallInterrupted(Exception):
    """送出 RPC 后结果未知，须越过研究驱动的普通候选失败回退。"""


current = ContextVar('contest_round_budget', default=None)
MIN_WORK = {'A':120, 'B':210, 'C':90}
REFERENCE_ROWS = {'A':1244, 'B':1679, 'C':23049}
PREDICT_DEFAULTS = {'A': (30.0, 0.06), 'B': (0.0, 0.0), 'C': (0.0, 0.0)}
PREDICT_UPLOAD_RESERVE = 120.0


def stable_gas_parent(rnd, state):
    """只有同一高分父版在相同发放上连续两次实测未改善，才缩短本轮探索。"""
    if state is None or rnd.round_index < 4:
        return False
    path = state/'history-A.json'
    if not path.is_file():
        return False
    from hashlib import file_digest
    score = rnd.parent.receipt.get('summary', {}).get('metrics', {}).get('macro_f1')
    if type(score) not in (int, float) or not 0.90 <= score <= 1:
        return False
    digest = rnd.parent.checkpoint_sha256.removeprefix('sha256:')
    events = json.loads(path.read_bytes()).get('events', [])[-2:]
    if len(events) != 2:
        return False
    for number, event in enumerate(events, rnd.round_index-2):
        if (event.get('platform_stage_id') != f'sdk-A-{number}' or event.get('error')
                or event.get('accepted') is not False or event.get('published_sha256') != digest
                or event.get('actual_parent_sha256') != digest):
            return False
    identity = {}
    for key, issued in [('train_sha256', rnd.data.train_path), ('validation_sha256', rnd.data.validation_path)]:
        with Path(issued).open('rb') as stream:
            identity[key] = 'sha256:' + file_digest(stream, 'sha256').hexdigest()
    return all(event.get('data_identity') == identity for event in events)


class RoundBudget:
    def __init__(self, profile, seconds, clock=time.monotonic):
        self.clock = clock
        self.start = clock()
        self.profile = profile
        self.source = 'host' if seconds is not None else 'default'
        if seconds is None:
            seconds = 3600
        if type(seconds) not in (float,int) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('整轮时限须为有限正数')
        self.seconds = seconds
        self.deadline = self.start + seconds
        self.rows = REFERENCE_ROWS[profile]
        self.measured_stream_reserve = 0.
        self.defer_gas_reference = False
        self.times = {key:[] for key in ('model','submission','worker')}
        self.events = []

    @contextmanager
    def activate(self):
        token = current.set(self)
        try:
            yield self
        finally:
            current.reset(token)

    @property
    def remaining(self):
        return max(0., self.deadline-self.clock())

    def observe(self, kind, seconds):
        self.times[kind] = (self.times[kind]+[max(0.,seconds)])[-6:]

    def restore(self, folder):
        path = folder/'round-timing.json'
        if not path.is_file():
            return
        saved = json.loads(path.read_bytes())
        if saved.get('profile') != self.profile:
            return
        for kind in self.times:
            values = saved.get(kind)
            if isinstance(values,list):
                self.times[kind] = [v for v in values[-6:] if type(v) in (float,int)
                                    and math.isfinite(v) and v>=0]

    def save(self, folder):
        atomic_json(folder/'round-timing.json',dict(profile=self.profile,**self.times))

    @property
    def gateway(self):
        return max(300.,1.5*max(self.times['model'],default=0.))

    @property
    def handoff(self):
        return self.gateway+60

    @property
    def publish(self):
        return max(300.,300*self.rows/REFERENCE_ROWS[self.profile],self.measured_stream_reserve,
                   self.estimate_predict(),
                   1.5*max(self.times['submission'],default=0.))

    def estimate_predict(self):
        """按单行预测和上传确认成本估算完整服务的保守预留。"""
        cold_default, row_default = PREDICT_DEFAULTS.get(self.profile, (0.0, 0.0))
        cold = os.environ.get('HARNESS_COLD_SECONDS', cold_default)
        per_row = os.environ.get('HARNESS_ROW_SECONDS', row_default)
        try:
            cold, per_row = float(cold), float(per_row)
        except (TypeError, ValueError) as exc:
            raise ValueError('预测成本常量须为有限非负数') from exc
        if not math.isfinite(cold) or cold < 0 or not math.isfinite(per_row) or per_row < 0:
            raise ValueError('预测成本常量须为有限非负数')
        if per_row <= 0:
            return 0.0
        return cold + self.rows * per_row + PREDICT_UPLOAD_RESERVE

    def forecast_gas_publication(self, path, measurement):
        """在父版准备前读取本轮公开规模，优先采用宿主提供的逐行观测。"""
        with Path(path).open('rb') as stream:
            self.rows = sum(1 for line in stream if line.strip())
        if not isinstance(measurement, dict):
            return
        cold = measurement.get('first_row_seconds')
        warm = measurement.get('later_p95_seconds')
        if all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (cold, warm)):
            self.measured_stream_reserve = 60 + cold + max(0, self.rows-1)*warm

    def marker(self, stage):
        remaining = self.remaining
        self.events.append({'stage':stage,'remaining_seconds':round(remaining,2)})
        print(f'[round-time] {stage} remaining={remaining:.2f}s',file=sys.stderr,flush=True)

    def allowance(self, limit, *, request=False):
        available = self.remaining-self.publish-self.handoff
        if request:
            available -= self.gateway
        cap = min(limit,math.floor(available))
        minimum = min(limit,max(MIN_WORK[self.profile],1.25*max(self.times['worker'],default=0.)))
        return cap if cap >= minimum and cap > 0 else 0

    def can_publish(self):
        return self.remaining >= self.publish+self.handoff

    def can_restore(self, artifact):
        size = sum(f.get('size_bytes',0) for f in artifact.get('files',[]))
        # 查询旧产物必须完整下载核验，优先给当前挂载父版的结束交接留足时间。
        return self.remaining >= self.handoff+max(60,size/(1024**2))

    def snapshot(self):
        return dict(source=self.source,wall_seconds=self.seconds,remaining_seconds=round(self.remaining,2),
                    model_reserve=self.gateway,submission_reserve=self.publish,events=list(self.events))

    def prepare(self, args, kwargs):
        """准备阶段只处理本地公开数据。隔离到可终止的进程，不向子进程传 SDK 通道。"""
        work = Path(args[-1])
        with TemporaryDirectory(dir=work.parents[1]) as tmp:
            job = Path(tmp)/'prepare.json'
            atomic_json(job,{'args':[str(x) for x in args], 'kwargs':{
                k:str(v) if isinstance(v,Path) else v for k,v in kwargs.items()}})
            cap = self.remaining-self.publish-self.handoff
            if self.profile == 'A' and hasattr(self, 'program_limit'):
                cap = min(cap, self.program_limit)
            if cap <= 0:
                raise CloseResearch('父版准备前进入收尾窗口')
            try:
                subprocess.run([sys.executable,'-B','-m',__name__,str(job)],timeout=cap,
                               stdin=subprocess.DEVNULL,stdout=sys.stderr,check=True)
            except subprocess.TimeoutExpired as exc:
                raise CloseResearch('父版准备到达截止时间') from exc
            except subprocess.CalledProcessError as exc:
                if exc.returncode == 137:
                    raise CloseResearch('父版准备内存不足被内核终止(137)') from exc
                raise
        oracle = json.loads((work/'oracle.json').read_bytes())
        self.rows = len(oracle['explore']['ids'])
        return work/'task',oracle


def retain_before_training(rnd, parent, release, native_state, reason):
    """在准备阶段停止，记录真实的零步训练，让共同选案和交接代码完成回合。"""
    from hashlib import file_digest
    from harness.contest.lineage import checkpoint_state,parent_handoff
    with parent.open('rb') as stream:
        digest = file_digest(stream,'sha256').hexdigest()
    identity = checkpoint_state(parent)
    history_path = native_state/f'history-{rnd.dataset_key}.json'
    history = json.loads(history_path.read_bytes()).get('events',[]) if history_path.is_file() else []
    report = dict(status='budget_retained_parent',seed=42,executed_parent_epochs=0,
                  successful_training_candidates=0,experiments=[],initial_steps=identity['steps'],
                  final_steps=identity['steps'],parent_checkpoint_sha256=digest,checkpoint_sha256=digest,
                  checkpoint_state_before=identity,checkpoint_state_after=identity,
                  parent_handoff=parent_handoff(digest,history),reason=str(reason))
    stage = f'sdk-{rnd.dataset_key}-{rnd.round_index}'
    history.append(dict(stage=rnd.round_index,platform_stage_id=stage,accepted=False,
                        actual_parent_sha256=digest,published_sha256=digest,selected={},
                        error={'type':'budget','message':str(reason)}))
    release.mkdir(exist_ok=True)
    atomic_json(release/'training.json',report)
    atomic_json(history_path,{'events':history})
    return dict(published_sha256=digest,recording_directory=str(native_state))


if __name__=='__main__':
    from .training import prepare_round
    job = json.loads(Path(sys.argv[1]).read_bytes())
    args = [job['args'][0],*(Path(s) for s in job['args'][1:])]
    if job['kwargs'].get('training_hint') is not None:
        job['kwargs']['training_hint'] = Path(job['kwargs']['training_hint'])
    prepare_round(*args,**job['kwargs'])
