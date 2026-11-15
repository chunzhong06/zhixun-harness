"""保留共同补训对照，另按两套系统各自的发布语义核验完整成绩。"""
from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
from engineering.cli.study.audit_publication_effect import selected_models
from engineering.cli.study.summarize_frontier_comparison import verdict
from engineering.cli.workflow import london_field_driver as london
from harness.runtime.storage import atomic_json


def summarize(folder, audit):
    original = folder / 'comparison.json'
    report = deepcopy(json.loads(original.read_bytes()))
    diagnosis = json.loads((audit / 'scores.json').read_bytes())
    london.configure('2026-07-01')
    for row in report['rows']:
        row['project']['publication_rule'] = '现有设计：官方保留确认权重；伦敦按冻结程序补训'
        row['comind']['publication_rule'] = '保留作者选优时保存的权重，不额外补训'
        if row['task'] not in ('BX2', 'CE2'):
            continue
        site = row['task']
        work = ROOT / row['comind']['path']
        decision, accepted, _ = selected_models(work, 'comind')
        assert decision['adopted']
        evidence = audit / site / 'comind' / 'issued.npz'
        item = diagnosis['results'][site + '/comind']
        assert item['original_delivery_reproduced']
        assert accepted['weights_sha256'] == item['weights_sha256']['accepted_weights']
        with np.load(evidence, allow_pickle=False) as z, np.load(work / 'issued.npz', allow_pickle=False) as old:
            np.testing.assert_array_equal(z['ids'], old['ids'])
            np.testing.assert_array_equal(z['baseline'], old['baseline'])
            np.testing.assert_allclose(z['refitted_weights'], old['comind'], rtol=0, atol=1e-9)
            ids = z['ids']
            prediction = z['accepted_weights']
            assert len(ids) == 1344 and np.isfinite(prediction).all()
            future = np.arange(len(ids)) >= 28 * 24
            np.testing.assert_array_equal(prediction[~future], old['baseline'][~future])
            truth = london.readings(ROOT / 'data/public/london-air-2026/stations.zip', site).to_numpy()[ids]
            valid = np.isfinite(truth)
            score = float(np.sqrt(np.mean((prediction[valid] - truth[valid]) ** 2)))
            after = float(np.sqrt(np.mean((prediction[valid & future] - truth[valid & future]) ** 2)))
        assert abs(score - item['metrics']['accepted_weights']['complete_rmse']) < 1e-12
        assert abs(after - item['metrics']['accepted_weights']['after_maintenance_rmse']) < 1e-12
        row['comind'].update(score=score, after_maintenance=after,
                             prediction_evidence=evidence.relative_to(ROOT).as_posix(),
                             prediction_sha256=sha256(evidence.read_bytes()).hexdigest(),
                             weights_sha256=accepted['weights_sha256'])
        row['verdict'] = verdict(row['project']['score'], score)
    for row in report['rows']:
        row['relative_error_change_percent'] = None if row['task'] == 'gas' else (
            100 * (row['project']['score'] / row['comind']['score'] - 1))
    report.update(
        comparison_scope='完整框架任务适配；各自保留发布语义。伦敦作者原生发布由原选案及保存权重重放恢复。',
        correction='共同补训使BX2 CoMind退步；主比较统一对全部CoMind案例保留原选中权重，未按未来分数选择发布方式。',
        author_source='references/implementations/comind-ml/CoMind/f93dff779b5d40dd3bf28b7b6ba5e39881e02e38/comind/coder.py',
        author_semantics='_process_metric_report和run保存best_code/best_submission；_generate_report返回原选中结果，无搜索结束后的额外补训。',
        control_evidence=original.relative_to(ROOT).as_posix(),
        control_sha256=sha256(original.read_bytes()).hexdigest(),
        audit_evidence=(audit / 'scores.json').relative_to(ROOT).as_posix(),
        transfer={v: sum(r['verdict'] == v for r in report['rows'][3:]) for v in ('胜', '平', '负')},
        counting_note='胜负字段仅按数值大小；BX2差0.018%、轴承差0.12%按表现接近解读，不宣称显著。',
        new_training=0, new_model_calls=0,
    )
    atomic_json(folder / 'author-publication-comparison.json', report)
    print(json.dumps({'rows': [{k: r[k] for k in ('task', 'relative_error_change_percent')} for r in report['rows']],
                      'model_calls': report['model_calls']}, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    args = p.parse_args()
    summarize(args.folder.resolve(), args.audit.resolve())
