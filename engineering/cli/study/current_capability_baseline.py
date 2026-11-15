"""从公开完整入口复跑现有两套配置，按实际交付计分，不搜索新方案。"""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src'), str(Path(__file__).parent)]
from run_gas_served_scale_integrated import load_issued
from score_local_defaults import official, metric
from harness.runtime.storage import atomic_json

CONFIGS = {
    'recommended': ['--gas-feedback', '32', '--engine-input-repair'],
    'development': ['--gas-feedback', '32', '--gas-field-mode', 'coverage-served-scale',
                    '--engine-continued-condition'],
}
REFERENCES = {
    'recommended': 'fixed-tree-references/setting-reference/automatic/integrated',
    'development': 'whole-maintenance/gas-feedback-coverage/integrated',
}


def execute(folder, rounds, options):
    """复用真实入口和恢复语义；子进程的失败与输出保留在本次结果内。"""
    command = [sys.executable, '-B', str(ROOT / 'engineering/cli/workflow/run_workflow.py'),
               'public-run', 'all', '--rounds', str(rounds), '--output', str(folder), *options]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    atomic_json(folder / f'command-{rounds}.json', {'command': command, 'returncode': result.returncode})
    (folder / f'stdout-{rounds}.log').write_text(result.stdout, encoding='utf-8')
    (folder / f'stderr-{rounds}.log').write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr[-1800:])


def run(output):
    if output.exists():
        raise ValueError('能力基线使用新目录，保留原实测及失败')
    output.mkdir(parents=True)
    answers = {task: official(task)[0] for task in ('gas', 'bearing', 'engine')}
    summary = {}
    for arm, options in CONFIGS.items():
        folder = output / arm
        execute(folder, 4, options)
        prefix = {task: load_issued(folder, task)[2] for task in answers}
        execute(folder, 10, options)
        table = {}
        for task, package in answers.items():
            files, records, predictions = load_issued(folder, task)
            if len(files) != 10 or any(predictions[k] != v for k, v in prefix[task].items()):
                raise ValueError('恢复改变已交预测或缺少十轮交付')
            truth = {row.sample_id: row.label for row in package.validation}
            reference_folder = ROOT / 'results/experiments/local-defaults' / REFERENCES[arm]
            reference = load_issued(reference_folder, task)[2]
            if predictions.keys() != reference.keys():
                raise ValueError('当前与历史服务对象不同，不能直接比较分数')
            changes = sum(abs(predictions[k] - reference[k]) > 1e-7 for k in predictions)
            groups = defaultdict(dict)
            for row in package.validation:
                if row.sample_id in predictions:
                    groups[str(row.unit_id or row.group)][row.sample_id] = predictions[row.sample_id]
            result = {'score': metric(task, truth, predictions), 'rows': len(predictions),
                      'rounds': len(files), 'changed_predictions_from_reference': changes,
                      'max_prediction_difference': max(abs(predictions[k] - reference[k]) for k in predictions),
                      'reference': str(reference_folder.relative_to(ROOT)),
                      'group_scores': {g: metric(task, truth, values) for g, values in groups.items()},
                      'receipt_sha256': {name: sha256((folder / task / name).read_bytes()).hexdigest() for name in files},
                      'prediction_sha256': sha256(json.dumps(predictions, sort_keys=True).encode()).hexdigest()}
            if task == 'gas':
                result['correct'] = sum(predictions[k] == truth[k] for k in predictions)
            table[task] = result
            print(json.dumps({'配置': arm, '任务': task, '完整交付': result['score'],
                              '预测数': result['rows'], '相对历史变化数': changes}, ensure_ascii=False), flush=True)
        summary[arm] = table
        atomic_json(output / 'scores.json', summary)
    atomic_json(output / 'scope.json', {'purpose': '冻结现有公开推荐与开发组合的能力基线',
                 'new_model_search': False, 'new_language_model_calls': 0,
                 'data_status': '官方公开开发资料，不作为新增独立泛化样本',
                 'resume_4_to_10_verified': True})
    return summary


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
