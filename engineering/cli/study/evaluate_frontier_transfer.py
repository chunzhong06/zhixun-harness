"""将两组实际发布模型冻结到既有NASA测试设备，不据目标答案改模型。"""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from engineering.cli.study.audit_publication_effect import replay
from harness.data.cmapss_input import read_cmapss, test_answers, REGISTERED_SHA256
from harness.integrations.engine_parent_view import transform
from harness.runtime.storage import atomic_json

SUBSETS = ('FD001', 'FD002', 'FD003', 'FD004')


def run(args):
    args.pair, args.bundle, args.output = args.pair.resolve(), args.bundle.resolve(), args.output.resolve()
    author_run = args.comind.resolve() if args.comind else args.pair/'comind'
    args.output.mkdir(parents=True, exist_ok=False)
    identity = json.loads((args.bundle/'task-identity.json').read_bytes())
    binding = identity['parent_bindings'][5]
    archive = ROOT/'data/public/cmapss/CMAPSSData.zip'
    assert sha256(archive.read_bytes()).hexdigest() == REGISTERED_SHA256
    overlap_path = ROOT/'results/inspections/engine/NASA终点输入核验.json'
    overlap = json.loads(overlap_path.read_bytes())
    assert overlap['raw_archive'] == REGISTERED_SHA256
    assert overlap['official_archive'] == identity['archive_sha256']
    assert not any(overlap['test_endpoints_matching_official_rows_by_subset'].values())
    assert not overlap['test_prefixes_matching_train']
    # train与test各自从设备1编号；以真实轨迹排除重复，不按局部编号判断。
    starts = defaultdict(list)
    for subset in SUBSETS:
        for track in read_cmapss(archive, subset=subset, partition='train'):
            values = np.asarray(track.values, dtype='<f8')
            starts[subset, values[0].tobytes()].append(values)
    tracks, ids, subsets, raw = {}, [], [], []
    for subset in SUBSETS:
        tracks[subset] = read_cmapss(archive, subset=subset, partition='test')
        endpoint_hash = sha256()
        for track in tracks[subset]:
            values = np.asarray(track.values, dtype='<f8')
            assert not any(np.array_equal(train[:len(values)], values) for train in starts[subset, values[0].tobytes()])
            endpoint_hash.update(f'{subset}:{track.unit}:{len(values)}\n'.encode())
            endpoint_hash.update(values[-1].tobytes())
            # 官方接口顺序为传感器在前、三项设置在后。
            raw.append(np.r_[values[-1, 3:], values[-1, :3]])
            ids.append(track.unit_id)
            subsets.append(subset)
        assert endpoint_hash.hexdigest() == overlap['endpoint_hashes'][subset]
        assert len(tracks[subset]) == overlap['source_test_devices'][subset]
    assert len(ids) == len(set(ids))
    with tempfile.TemporaryDirectory(prefix='temporary-view-', dir=args.output) as temporary:
        assert Path(temporary).resolve().is_relative_to(args.output)
        previous_temp = tempfile.tempdir
        tempfile.tempdir = temporary
        try:
            native, parent = transform(binding, np.asarray(raw), ROOT/'data/official')
            # 在已核验官方输入上复现相同视图与父预测，才将变换用于其他设备。
            with np.load(args.bundle/'outer-inputs.npz', allow_pickle=False) as z:
                previous = z['service_x']
            check_native, check_parent = transform(binding, previous[:, :24], ROOT/'data/official')
        finally:
            tempfile.tempdir = previous_temp
    np.testing.assert_allclose(check_native, previous[:, 24:48], rtol=0, atol=1e-8)
    np.testing.assert_allclose(check_parent, previous[:, -1], rtol=0, atol=1e-8)
    columns = json.loads((args.bundle/'task/task.json').read_bytes())['feature_columns']
    inputs = args.output/'service.csv'
    pd.DataFrame(np.column_stack((raw, native, parent)), columns=columns).to_csv(inputs, index=False)
    predictions, published = {'parent': parent}, {}
    for arm in ('project', 'comind'):
        # 复用对照时保留其原目录，不把历史作者运行伪装成本次新调用。
        work = author_run if arm == 'comind' else args.pair/arm
        frozen = json.loads((work/'input-identity.json').read_bytes())
        assert frozen['bundle_identity_sha256'] == sha256((args.bundle/'task-identity.json').read_bytes()).hexdigest()
        decision = json.loads((work/'decision.json').read_bytes())
        if decision['adopted']:
            weights = work/'serving/model.pkl'
            checksum = sha256(weights.read_bytes()).hexdigest()
            assert checksum == decision['candidate']['weights_sha256']
            # 两组只使用原真实采用版；不在本次目标设备上重新确认或选择候选。
            predictions[arm] = replay(weights, inputs, args.output/arm, timeout=240)
            published[arm] = {'kind': 'adopted_candidate', 'weights_sha256': checksum}
        else:
            predictions[arm] = parent.copy()
            published[arm] = {'kind': 'unchanged_parent', 'binding_identity': binding['identity']}
        published[arm].update(run_directory=str(work),
                              decision_sha256=sha256((work/'decision.json').read_bytes()).hexdigest())
        assert predictions[arm].shape == (len(ids),) and np.isfinite(predictions[arm]).all()
    np.savez_compressed(args.output/'issued.npz', ids=np.asarray(ids), subsets=np.asarray(subsets), **predictions)
    atomic_json(args.output/'predictions-complete.json', {'all_predictions_sealed': True, 'rows': len(ids),
        'published': published, 'parent_binding': binding['identity'], 'archive_sha256': REGISTERED_SHA256,
        'official_view_reproduced': True, 'train_test_trajectory_overlap': 0,
        'source_overlap_audit_sha256': sha256(overlap_path.read_bytes()).hexdigest(),
        'input_sha256': sha256(inputs.read_bytes()).hexdigest()})
    # 只有全部预测封存后，才读取每台测试设备的最终RUL答案。
    answers = {}
    for subset in SUBSETS:
        answers.update(test_answers(archive, subset, tracks[subset]))
    truth = np.asarray([answers[key] for key in ids])
    subsets = np.asarray(subsets)
    scores = {}
    for arm, values in predictions.items():
        group_scores = {subset: float(np.sqrt(np.mean((values[subsets == subset]-truth[subsets == subset])**2)))
                        for subset in SUBSETS}
        scores[arm] = {'pooled_rmse': float(np.sqrt(np.mean((values-truth)**2))),
                       'macro_subset_rmse': float(np.mean(list(group_scores.values()))), 'by_subset': group_scores}
    report = {'scope': '既有NASA测试资料上新冻结发布模型的跨设备复验；旧模型曾在run49评分',
              'counts': {subset: len(tracks[subset]) for subset in SUBSETS}, 'scores': scores,
              'relative_error_change_percent': 100*(scores['project']['pooled_rmse']/scores['comind']['pooled_rmse']-1),
              'new_training': 0, 'new_model_calls': 0, 'published': published,
              'no_target_refit_or_selection': True, 'all_predictions_sealed_before_answers': True}
    if args.uncertainty:
        from engineering.cli.study.analyze_selection_stability import pair_summary
        losses = (np.column_stack((predictions['project'], predictions['comind']))-truth[:, None])**2
        report['paired_uncertainty'] = {
            'method': '同设备配对、FD子集内分层的5000次设备重采样；随机种子20261115',
            'result': pair_summary(losses, np.asarray(ids), subsets),
            'scope': '条件于当前两个冻结模型和四个固定子集；未包括再次训练、再次模型搜索及此前开发选案的不确定性。'}
    atomic_json(args.output/'scores.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser()
    for name in ('pair', 'bundle', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--comind', type=Path, help='复用的原作者运行目录；不复制、不重新选模型')
    p.add_argument('--uncertainty', action='store_true', help='附预先约定的配对设备重采样区间；不改变预测或选案')
    run(p.parse_args())
