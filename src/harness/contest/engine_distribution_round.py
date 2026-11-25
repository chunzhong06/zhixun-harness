"""发动机源分布候选的续训、公开比较和父版交接；宿主仍独立测量与选案。"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from harness.adapters.storage.local import atomic_json
from harness.contest.lineage import checkpoint_state, parent_handoff
from harness.contest.round_budget import CloseResearch
from harness.learning.inference.engine_source_forest import ARMS, KIND, SourceForestModel, combine


def checksum(path):
    from hashlib import file_digest
    with Path(path).open('rb') as stream:
        return file_digest(stream,'sha256').hexdigest()


def metrics(y, predicted):
    err = predicted-y
    return dict(rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),
                nasa=float(np.expm1(np.maximum(err/10,-err/13)).sum()))


def run_worker(train_path, validation_path, parent, out, source_digest):
    from harness.adapters.platforms.baselines.core_model import Model
    from harness.adapters.platforms.baselines.data import read_dataset
    from harness.learning.inference.complete_checkpoint import load_checkpoint
    from harness.learning.training.engine_source_forest import device_rows, fit_arrays

    started = monotonic()
    incoming = load_checkpoint(parent,Model)
    has_forest = isinstance(incoming,SourceForestModel)
    base = incoming.base if has_forest else incoming
    paths = base.meta['paths']
    schema = [[['key',k],['index',j]] for k,n in [('sensors',21),('settings',3)] for j in range(n)]
    if [json.loads(p) for p in paths] != schema or (not has_forest and type(base) is not Model):
        return False
    def load(path):
        return read_dataset(path,base.meta['task'],base.meta['classes'],paths)
    train, dev = load(train_path), load(validation_path)
    if set(train['ids']) & set(dev['ids']):
        raise ValueError('发动机训练和验证的标识重叠')
    if has_forest and incoming.state['source'] == source_digest:
        arrays = dict(incoming.auxiliary_arrays)
    else:
        grouping = device_rows(train['ids'],train['y'])
        if grouping is None:
            return False
        arrays = fit_arrays(train['x'], train['y'], grouping, source_digest)
    old_steps = base.meta['steps']
    target = dev['y']
    forest = SourceForestModel(base,arrays)
    mean,risk = forest.moments(dev['x'])
    prior = base.predict(dev['x'])
    if has_forest:
        if incoming.state['source'] == source_digest:
            prior = combine(incoming.state['arm'],prior,mean,risk)
        else:
            prior = incoming.predict(dev['x'])
    before = metrics(target,prior)
    # 基础神经网络继承权重、Adam 与两套尺度，确实执行一个源训练 epoch。
    base.meta['learning_rate'] = min(base.meta['learning_rate'],.00001)
    base.fit(train['x'],train['y'],1)
    trained = base.predict(dev['x'])
    options = {arm:metrics(target,combine(arm,trained,mean,risk)) for arm in ARMS}
    allowed = [k for k,v in options.items() if v['rmse']<before['rmse'] and v['nasa']<=before['nasa']]
    arm = min(allowed,key=lambda k:options[k]['rmse']) if allowed else None
    dest = out/'checkpoint.npz'
    if arm:
        arrays['complete_state'] = np.array(json.dumps({**forest.state,'arm':arm},sort_keys=True))
        raw = out/'base.npz'
        base.save(raw)
        with np.load(raw,allow_pickle=False) as stored:
            np.savez_compressed(dest,**{k:stored[k] for k in stored.files},**arrays)
        raw.unlink()
    else:
        shutil.copyfile(parent,dest)
    atomic_json(out/'outcome.json',dict(arm=arm,options=options,before=before,
                after=options[arm] if arm else before,actual_training_steps=base.meta['steps']-old_steps,
                seconds=monotonic()-started,source=source_digest))
    return True


def try_round(rnd, parent, release, state, budget, limit):
    with np.load(parent,allow_pickle=False) as ck:
        extended = 'complete_state' in ck
        if extended and json.loads(str(ck['complete_state'])).get('kind') != KIND:
            return None
    cap = budget.allowance(limit)
    if not cap:
        raise CloseResearch('保留发动机宿主测量和交接时间')
    train, dev = rnd.data.train_path, rnd.data.validation_path
    source = checksum(train)
    with TemporaryDirectory(dir=state) as tmp:
        out = Path(tmp)
        command = [sys.executable,'-B','-m',__name__,str(train),str(dev),str(parent),str(out),source]
        budget.marker('start_worker')
        try:
            process = subprocess.run(command,timeout=cap)
        except subprocess.TimeoutExpired as exc:
            raise CloseResearch('源分布训练已达到本轮候选时间上限') from exc
        if process.returncode==3:
            if extended:
                raise CloseResearch('本轮来源缺少可验证的训练机组，沿用父版')
            return None
        process.check_returncode()
        outcome = json.loads((out/'outcome.json').read_bytes())
        budget.observe('worker',outcome['seconds'])
        release.mkdir()
        (out/'checkpoint.npz').replace(release/'checkpoint.npz')
    dest = release/'checkpoint.npz'
    initial,final = checkpoint_state(parent),checkpoint_state(dest)
    parent_sha, digest = checksum(parent),checksum(dest)
    history_path = state/'history-C.json'
    events = json.loads(history_path.read_bytes())['events'] if history_path.is_file() else []
    arm = outcome['arm']
    report = dict(status='adopted' if arm else 'retained_parent',seed=42,
                  initial_steps=initial['steps'],final_steps=final['steps'],executed_parent_epochs=1,
                  parent_checkpoint_sha256=parent_sha,checkpoint_sha256=digest,
                  checkpoint_state_before=initial,checkpoint_state_after=final,
                  parent_handoff=parent_handoff(parent_sha,events),validation_before=outcome['before'],
                  validation_after=outcome['after'],successful_training_candidates=len(ARMS),experiments=[],
                  execution_seconds=outcome['seconds'],source_distribution=outcome)
    events.append(dict(stage=rnd.round_index,platform_stage_id=f'sdk-C-{rnd.round_index}',
                       actual_parent_sha256=parent_sha,published_sha256=digest,accepted=bool(arm),
                       selected={'score':outcome['after']['rmse']},error=None))
    atomic_json(release/'training.json',report)
    atomic_json(history_path,{'events':events})
    return dict(published_sha256=digest,recording_directory=str(state))


if __name__ == '__main__':
    sys.exit(0 if run_worker(*(Path(s) for s in sys.argv[1:5]),sys.argv[5]) else 3)
