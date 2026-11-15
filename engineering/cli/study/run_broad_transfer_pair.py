"""两个独立进程运行完整配对；双方发完全部预测后才打开服务答案。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.broad_transfer_task import metric, verify_core
from harness.runtime.storage import atomic_json


def main():
    parser = ArgumentParser(); parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--task', required=True, choices=['sgp40', 'secom', 'wind', 'ncmapss_ds06'])
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    verify_core(); out = args.output.resolve()/args.task
    if out.exists():
        raise ValueError('不覆盖已有完整配对')
    out.mkdir(parents=True)
    env = os.environ.copy(); env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    jobs = []
    for arm in ['project', 'comind']:
        stdout = (out/(arm+'-stdout.log')).open('w', encoding='utf-8'); stderr = (out/(arm+'-stderr.log')).open('w', encoding='utf-8')
        p = subprocess.Popen([sys.executable, '-B', str(ROOT/'engineering/cli/workflow/broad_transfer_comparison.py'),
            '--bundle', str(args.bundle.resolve()), '--task', args.task, '--arm', arm, '--output', str(out/arm)],
            cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        jobs.append((p, stdout, stderr))
    codes = []
    for p, stdout, stderr in jobs:
        codes.append(p.wait()); stdout.close(); stderr.close()
    if any(codes):
        raise RuntimeError('配对进程未完整结束：'+str(codes))
    source = args.bundle/args.task; reports = {}; issued = {}; identities = []
    for arm in ['project', 'comind']:
        folder = out/arm
        reports[arm] = json.loads((folder/'predictions-complete.json').read_bytes())
        if sha256((folder/'issued.npz').read_bytes()).hexdigest() != reports[arm]['issued_sha256']:
            raise ValueError('封存预测身份变化')
        with np.load(folder/'issued.npz', allow_pickle=False) as z:
            issued[arm] = {k: z[k] for k in z.files}
        identities.append(json.loads((folder/'input-identity.json').read_bytes()))
    for key in ['bundle_identity_sha256', 'description_sha256', 'max_model_calls']:
        if identities[0][key] != identities[1][key]:
            raise ValueError('双方条件不一致')
    atomic_json(out/'both-predictions-sealed.json', reports)
    with np.load(source/'private-answers.npz', allow_pickle=False) as z:
        y, ids = z['y'], z['ids']
    with np.load(source/'outer-inputs.npz', allow_pickle=False) as z:
        future = z['future']
    oracle = json.loads((source/'oracle.json').read_bytes()); groups = np.asarray(oracle['service_groups'])
    scores = {}
    for arm in ['project', 'comind']:
        if not np.array_equal(ids, issued[arm]['ids']):
            raise ValueError('预测ID顺序改变')
        if not np.array_equal(issued[arm]['actual'][~future], issued[arm]['baseline'][~future]):
            raise ValueError('历史交付改变')
        scores[arm] = {**reports[arm], 'complete': {}, 'after_publication': {}}
        for name in ['baseline', 'direct', 'actual']:
            values = issued[arm][name]
            scores[arm]['complete'][name] = metric(args.task, y, values, groups)
            scores[arm]['after_publication'][name] = metric(args.task, y[future], values[future], groups[future])
        atomic_json(out/arm/'scores.json', scores[arm])
    comparison = {'task': args.task, 'same_conditions_verified': True,
                  'scope': '已见资料上的冻结设计跨数据族完整配对；不是新的独立泛化集', 'arms': scores}
    atomic_json(out/'comparison.json', comparison)
    print(json.dumps(comparison, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
