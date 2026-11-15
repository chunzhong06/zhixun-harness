"""官方源训练权限下迁移旧气体组合，核对轴承及发动机完整接续。"""
import asyncio
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
from engineering.cli.study.audit_native_run import audit
from engineering.cli.study.evaluate_native_transfer import dataset
from engineering.cli.study.evaluate_rul_baseline import PACKAGES
from engineering.cli.study.verify_native_balanced_cross_task import predict
from engineering.cli.workflow.gas_observer_predict import infer
from engineering.cli.workflow.native_research import run
from harness.data.official_gas import read_official_gas
from harness.integrations.native_research import export_release, measure
from harness.models.complete_checkpoint import migrate_gas, load_checkpoint
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.gas_views import pair_model
from harness.models.native_continuation import tensor_identity
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'


class Broker:
    def __init__(self, calls): self.calls, self.max_calls = 0, calls

    async def exchange(self, kind, payload):
        if kind != 'model': raise ValueError('只接受固定工程动作')
        self.calls += 1
        action = ({'action': 'reference', 'name': 'balanced-fine-tuning', 'hypothesis': '保持已有源训练组合，按类别均衡继续训练真实父网络'}
                  if self.calls == 1 else {'action': 'finish', 'execution': 1, 'candidate': 'candidate-001',
                                         'reason': '交独立确认决定采用或保持'})
        return {'text': json.dumps(action, ensure_ascii=False)}


async def main(out):
    out.mkdir(parents=True, exist_ok=False)
    parent_run = ROOT / 'results/experiments/local-defaults/whole-maintenance/gas-feedback-coverage/integrated/gas'
    report = json.loads((parent_run / 'run.json').read_bytes())
    payload = json.loads((parent_run / 'accepted' / (report['accepted'][0]['request'] + '.json')).read_bytes())['payload']
    snapshot = parent_run / 'snapshots' / (payload['snapshot'] + '.json')
    state = json.loads(snapshot.read_bytes())
    if state['version'] != 1 or np.asarray(state['counts']).any() or payload['feedback_seen']:
        raise ValueError('气体起点包含目标域回标，不能迁入源训练协议')
    parent = parent_run / 'snapshots/models' / (state['weights'] + '.npz')
    if sha256(parent.read_bytes()).hexdigest() != state['weights']: raise ValueError('气体旧父版改变')
    archive = ROOT / 'data/official/gas/cpu-a-guide-20261110-v1.zip'
    package = read_official_gas(archive)
    if set(state['fit_ids']) != {r.sample_id for r in package.train}: raise ValueError('旧气体源资料不匹配')
    train = list(package.train)
    x = np.asarray([r.values for r in package.validation]); y = np.asarray([r.label for r in package.validation])
    ids = np.asarray([r.sample_id for r in package.validation])
    bank = GasFeedbackBank(train)
    with ZipFile(archive) as z, TemporaryDirectory(dir=out, prefix='temporary-') as tmp:
        with _trainer(z, Path(tmp), 'A') as trainer:
            native = pair_model(trainer.Model).load(parent)
            full = migrate_gas(native, bank, provenance={'source_snapshot_sha256': sha256(snapshot.read_bytes()).hexdigest(),
                'parent_checkpoint_sha256': state['weights'], 'train_ids_sha256': sha256(json.dumps(sorted(state['fit_ids'])).encode()).hexdigest(),
                'training_rows': len(train), 'target_feedback_rows': 0})
            migrated = out / 'migrated.npz'; full.save(migrated)
            reloaded = load_checkpoint(migrated, trainer.Model)
            if any(tensor_identity(reloaded)[k] != v for k, v in tensor_identity(native).items()):
                raise ValueError('原生气体父权重或Adam在迁移中改变')
            old_votes = bank.predict(native, x)
            source_votes = reloaded.source_votes(x)
            for old, new in zip(list(old_votes.values())[1:], source_votes):
                if not np.array_equal(old, new): raise ValueError('数值分类器迁移改变旧源预测')
            requests = ({'task': 'classification', 'inputs': {'features': r.tolist()}} for r in x)
            legacy = np.asarray([[v['score'] for v in row['outputs']] for row in infer(snapshot, ROOT / 'data/official', requests)])
            actual = np.asarray([reloaded.predict(r[None, :])[0] for r in x])
            if not np.allclose(actual, legacy, rtol=0, atol=1e-12) or not np.array_equal(actual.argmax(1), legacy.argmax(1)):
                raise ValueError('完整气体源预测与旧入口不同')
            np.savez_compressed(out / 'gas-migration-predictions.npz', ids=ids, legacy=legacy, migrated=actual)
            result = {'gas': {'source_rows': len(train), 'target_feedback_rows': 0,
                'all_classifiers_equivalent': True, 'max_probability_difference': float(np.max(np.abs(actual-legacy))),
                'official_validation_source_only': measure('gas', y, actual), 'checkpoint_bytes': migrated.stat().st_size}}
            atomic_json(out / 'migration-verification.json', result)
            print(json.dumps({'migration': result}, ensure_ascii=False), flush=True)
    runtime = await run('gas', out / 'gas-runtime', driver='fixture', calls=2, stages=2,
                        parent_checkpoint=migrated, broker_factory=lambda folder, calls: Broker(calls))
    result['gas']['audit'] = audit(out / 'gas-runtime')
    official_prediction = predict('gas', out / 'gas-runtime/release', x, native.meta)
    external_x, external_ids, groups, truth = dataset('gas')
    unchanged = out / 'gas-unchanged-inference'
    export_release(out / 'gas-runtime/task', migrated, unchanged)
    before = predict('gas', unchanged, external_x, native.meta)
    after = predict('gas', out / 'gas-runtime/release', external_x, native.meta)
    result['gas'].update(official_validation_after=measure('gas', y, official_prediction),
                         external_before=measure('gas', truth(), before), external_after=measure('gas', truth(), after),
                         final_checkpoint=runtime['final_checkpoint'])
    np.savez_compressed(out / 'gas-final-predictions.npz', ids=ids, prediction=official_prediction)
    np.savez_compressed(out / 'gas-external-predictions.npz', ids=np.asarray(external_ids), before=before, after=after)
    # 轴承旧强版就是原生NPZ；沿现有选择复查同一122行和全量，无需再训练一遍。
    selection = json.loads((BASE / 'native-engine-long-balanced/delivery-selection.json').read_bytes())
    chosen = selection['tasks']['bearing']
    checkpoint = Path(chosen['selected'])
    if sha256(checkpoint.read_bytes()).hexdigest() != chosen['selected_sha256']:
        raise ValueError('轴承当前权重改变')
    filename, _, reader = PACKAGES['bearing']
    bearing = reader(ROOT / 'data/official/bearing' / filename)
    with np.load(checkpoint, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays['metadata']))
    release = out / 'bearing-inference'
    export_release(Path(chosen['task_dir']), checkpoint, release)
    bx = np.asarray([r.values for r in bearing.validation]); by = np.asarray([r.label for r in bearing.validation])
    bp = predict('bearing', release, bx, metadata)
    old_path = ROOT / 'results/experiments/local-defaults/whole-maintenance/frontier-2026/bearing/project-recovered/issued.npz'
    with np.load(old_path, allow_pickle=False) as arrays:
        old_ids, old_prediction = arrays['ids'].copy(), arrays['actual'].copy()
    positions = {r.sample_id: i for i, r in enumerate(bearing.validation)}
    subset = np.asarray([positions[k] for k in old_ids])
    tx, tids, tgroups, labels = dataset('bearing')
    tp = predict('bearing', release, tx, metadata)
    result['bearing'] = {'same_old_rows': len(subset), 'old_complete': measure('bearing', by[subset], old_prediction),
        'current_same_rows': measure('bearing', by[subset], bp[subset]),
        'official_full': measure('bearing', by, bp), 'external': measure('bearing', labels(), tp),
        'same_checkpoint_sha256': chosen['selected_sha256'], 'new_training': 0}
    np.savez_compressed(out / 'bearing-predictions.npz', ids=np.asarray([r.sample_id for r in bearing.validation]),
                        prediction=bp, external_ids=np.asarray(tids), external_prediction=tp)
    result['engine_audit'] = audit(BASE / 'complete-engine-integration-v2/runtime')
    atomic_json(out / 'verification.json', result)
    print(json.dumps({'cross_task_integration': result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args().output.resolve()))
