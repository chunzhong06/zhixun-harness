"""迁移旧完整发动机父版，核对源训练统计、独立预测和真实父版接续。"""
import asyncio
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from engineering.cli.study.evaluate_native_transfer import dataset
from engineering.cli.study.evaluate_rul_baseline import PACKAGES
from engineering.cli.study.search_native_training import source
from engineering.cli.study.verify_native_balanced_cross_task import predict
from engineering.cli.workflow.native_research import run
from harness.models.complete_checkpoint import migrate_engine, load_checkpoint
from harness.models.condition_processing import ConditionStandardScaler
from harness.models.full_condition_engine import FullConditionEngine
from harness.models.model_artifacts import ModelArtifacts
from harness.models.native_continuation import tensor_identity
from harness.models.official_rul_cpu import _trainer
from harness.models.source_combinations import SourceCombinations
from harness.integrations.native_research import export_release
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/local-defaults/whole-maintenance'
PARENT = BASE / 'gas-feedback-coverage/integrated/engine'
PREVIOUS = BASE / 'robust-shared-full-pair/engine/project'
OUT = ROOT / 'results/experiments/contest-continuation/complete-engine-integration'


def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y) - p)**2)))


class FixedBroker:
    """用固定源码验证完整运行链；不把工程动作计作模型服务搜索。"""
    def __init__(self, calls): self.calls, self.max_calls = 0, calls

    async def exchange(self, kind, payload):
        if kind != 'model': raise ValueError('本实验只执行固定工程动作')
        self.calls += 1
        action = ({'action': 'experiment', 'hypothesis': '保留完整父版组件，低学习率按设备均衡更新父输出层',
                   'python_source': source(lr=.0001, epochs=1, head=True, balanced=True, robust=True)}
                  if self.calls == 1 else {'action': 'finish', 'execution': 1,
                   'candidate': 'candidate-001', 'reason': '交共同确认规则决定是否采用'})
        return {'text': json.dumps(action, ensure_ascii=False)}


async def main():
    OUT.mkdir(parents=True, exist_ok=False)
    filename, archive_sha, reader = PACKAGES['engine']
    archive = ROOT / 'data/official/engine' / filename
    if sha256(archive.read_bytes()).hexdigest() != archive_sha: raise ValueError('官方原包改变')
    package = reader(archive)
    report = json.loads((PARENT / 'run.json').read_bytes())
    state = json.loads((PARENT / 'regular/state.json').read_bytes())
    fit_ids = {k for k, row in state['samples'].items() if row['role'] == 'fit'}
    sampled = [r for r in package.train if r.sample_id in fit_ids]
    if len(sampled) != len(fit_ids): raise ValueError('旧输入处理使用了源训练以外的样本')
    lookup = SourceCombinations.restore(json.loads((PARENT / 'source-combinations.json').read_bytes()))
    if SourceCombinations([r.values for r in sampled]).state() != lookup.state():
        raise ValueError('源组合统计未由合法训练样本复现')
    header, blob = ModelArtifacts(PARENT / 'references').read(report['native_artifact'])
    weight = PREVIOUS / 'executions/0001/artifacts/candidate-001.pkl'
    if sha256(weight.read_bytes()).hexdigest() != '1678df807a9c88b902a62b6e8a37c5203a58c1f0e638edc4ed8f7949892dccec':
        raise ValueError('旧采用的残差模型身份改变')
    with weight.open('rb') as stream: residual = cloudpickle.load(stream)
    x = np.asarray([r.values for r in package.validation])
    y = np.asarray([r.label for r in package.validation])
    ids = np.asarray([r.sample_id for r in package.validation])
    original_path = OUT / 'official.npz'
    original_path.write_bytes(package.checkpoint)
    with ZipFile(archive) as pack, TemporaryDirectory(dir=OUT, prefix='temporary-') as tmp:
        with _trainer(pack, Path(tmp), 'C') as trainer:
            condition, center, native, native_sha = FullConditionEngine.load_bundle(blob, trainer)
            if native_sha != report['native_sha256']: raise ValueError('旧原生模型身份改变')
            devices = defaultdict(list)
            for row in package.train: devices[row.unit_id].append(row)
            first = [r.values for rows in devices.values() for r in sorted(rows, key=lambda r: r.time_end)[:8]]
            ref_condition = ConditionStandardScaler((21, 22, 23), tuple(package.task.condition_decimals)).fit(first)
            ref_center = StandardScaler().fit(ref_condition.transform([r.values for r in sampled]))
            if not np.array_equal(ref_center.mean_, center.mean_) or not np.array_equal(ref_center.scale_, center.scale_):
                raise ValueError('旧工况中心不是源训练统计')
            for key in condition.groups_:
                for name in ('mean_', 'scale_'):
                    if not np.array_equal(getattr(condition.groups_[key], name), getattr(ref_condition.groups_[key], name)):
                        raise ValueError('工况分组统计未复现')
            initial = tensor_identity(native)
            full = migrate_engine(native, lookup, condition, center, residual, provenance={
                'native_sha256': native_sha, 'residual_sha256': sha256(weight.read_bytes()).hexdigest(),
                'source_only_statistics_verified': True, 'source_rows': len(package.train),
                'sampled_source_rows': len(sampled), 'official_archive_sha256': archive_sha})
            migrated = OUT / 'migrated.npz'
            full.save(migrated)
            loaded = load_checkpoint(migrated, trainer.Model)
            if any(tensor_identity(loaded)[k] != v for k, v in initial.items()):
                raise ValueError('迁移改变了原网络/优化器/尺度')
            native_view = center.transform(condition.transform(lookup.transform(x)))
            if not np.array_equal(full.native_features(x), native_view):
                raise ValueError('输入处理迁移不等价')
            parent = native.predict(native_view * native.scale + native.mean)
            columns = [f'input_{i}' for i in range(24)] + [f'native_{i}' for i in range(24)] + ['parent_value']
            frame = pd.DataFrame(np.column_stack((x, native_view, parent)), columns=columns)
            # 旧正式交付是逐行推理；不能用不同批大小的Torch计算替代原参照。
            legacy = np.asarray([residual.predict(frame.iloc[i:i+1])[0] for i in range(len(frame))])
            converted = np.asarray([loaded.predict(row[None, :])[0] for row in x])
            max_difference = float(np.max(np.abs(legacy - converted)))
            if max_difference > 2e-4: raise ValueError('残差迁移误差超过Float32数值容差：' + str(max_difference))
            train_x = np.asarray([r.values for r in package.train[:64]])
            _, logits = loaded.forward(loaded.normalized(train_x))
            _, analytic = loaded.continuation_output(train_x, logits)
            eps = 1e-3
            numeric = (loaded.continuation_output(train_x, logits + eps)[0]
                       - loaded.continuation_output(train_x, logits - eps)[0]) / (2 * eps)
            grad_error = float(np.max(np.abs(analytic - numeric)))
            if grad_error > .03: raise ValueError('完整输出反向传播与差分不符：' + str(grad_error))
            official = trainer.Model.load(original_path).predict(x)
            narrow_selection = json.loads((ROOT / 'results/experiments/contest-continuation/native-engine-long-balanced/final-selection.json').read_bytes())
            narrow = trainer.Model.load(Path(narrow_selection['checkpoint'])).predict(x)
            np.savez_compressed(OUT / 'same-input-predictions.npz', ids=ids, y=y, legacy=legacy,
                                migrated=converted, official=official, narrow=narrow)
            with np.load(PREVIOUS / 'issued.npz', allow_pickle=False) as z:
                issued = {k: z[k].copy() for k in z.files}
            oracle = json.loads((BASE / 'frontier-2026-inputs/engine/oracle.json').read_bytes())
            with np.load(BASE / 'frontier-2026-inputs/engine/outer-inputs.npz', allow_pickle=False) as z:
                future = z['future'].copy()
            positions = {sid: i for i, sid in enumerate(ids)}
            indices = np.asarray([positions[sid] for sid in issued['ids']])
            if list(issued['ids']) != oracle['service_ids']: raise ValueError('旧完整交付顺序变化')
            replay = issued['baseline'].copy()
            replay[future] = converted[indices[future]]
            if np.max(np.abs(replay - issued['actual'])) > 2e-4: raise ValueError('旧十轮完整交付未复现')
            np.savez_compressed(OUT / 'replayed-delivery.npz', ids=issued['ids'], legacy=issued['actual'], migrated=replay)
            result = {'old_delivery_rows': len(indices), 'old_delivery_rmse': rmse(y[indices], issued['actual']),
                'migrated_delivery_rmse': rmse(y[indices], replay), 'migration_max_difference': max_difference,
                'complete_gradient_max_difference': grad_error, 'native_optimizer_identical': True,
                'source_statistics_reproduced': True, 'checkpoint_bytes': migrated.stat().st_size,
                'full_public_validation': {k: rmse(y, v) for k, v in
                    {'legacy_frozen': legacy, 'migrated_frozen': converted, 'official': official, 'narrow_native': narrow}.items()}}
            atomic_json(OUT / 'migration-verification.json', result)
            print(json.dumps({'migration': result}, ensure_ascii=False), flush=True)
    # 主运行器从完整父版出发，候选只看探索反馈，采用门槛另看确认，服务最后评分。
    runtime = await run('engine', OUT / 'runtime', driver='fixture', calls=2, stages=2,
                        parent_checkpoint=migrated, broker_factory=lambda folder, calls: FixedBroker(calls))
    release = OUT / 'runtime/release'
    actual = predict('engine', release, x, loaded.meta)
    external_x, external_ids, groups, truth = dataset('engine')
    unchanged_release = OUT / 'unchanged-inference'
    export_release(OUT / 'runtime/task', migrated, unchanged_release)
    before_external = predict('engine', unchanged_release, external_x, loaded.meta)
    after_external = predict('engine', release, external_x, loaded.meta)
    direct = predict('engine', unchanged_release, x[:12], loaded.meta)
    if not np.array_equal(direct, converted[:12]): raise ValueError('独立预测入口改变迁移输出')
    np.savez_compressed(OUT / 'final-predictions.npz', ids=ids, actual=actual)
    np.savez_compressed(OUT / 'external-predictions.npz', ids=np.asarray(external_ids),
                        groups=np.asarray(groups), before=before_external, after=after_external)
    result.update(full_public_after=rmse(y, actual), actual_model_calls=0,
                  external_frozen_before=rmse(truth(), before_external), external_after=rmse(truth(), after_external),
                  stage_decisions=[{'stage': s['stage'], 'accepted': s['accepted'], 'error': s['error'],
                                    'checkpoint_sha256': s['published_sha256']} for s in runtime['stages']],
                  final_checkpoint=runtime['final_checkpoint'], independent_input_only_prediction_verified=True)
    atomic_json(OUT / 'verification.json', result)
    print(json.dumps({'complete_integration': result}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    OUT = parser.parse_args().output.resolve()
    asyncio.run(main())
