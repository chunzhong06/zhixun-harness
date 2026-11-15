"""核对固定分类消融的模型/预测身份，汇总种子范围和真实故障检出。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'engineering/src'))
from harness.runtime.storage import atomic_json

NAMES = {'full': '原软门控网络', 'uniform_gate': '均匀三专家', 'single_expert': '单专家网络',
         'without_parent': '网络移除父预测', 'without_auxiliary_regularization': '移除辅助正则',
         'without_weight_decay': '移除权重衰减', 'logistic-with-parent': '逻辑回归含父预测',
         'logistic-without-parent': '逻辑回归无父预测', 'forest-with-parent': '森林含父预测',
         'forest-without-parent': '森林无父预测', 'source-parent': '源父版',
         'project-published-candidate': '原Harness候选', 'comind-published-candidate': '原CoMind候选'}


def describe(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max())}


def show(value, digits=4):
    if value['min'] == value['max']:
        return f"{value['mean']:.{digits}f}"
    return f"{value['mean']:.{digits}f} [{value['min']:.{digits}f}, {value['max']:.{digits}f}]"


def main():
    parser = ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.root/'scores.json').read_bytes())
    if not all(source['reproduction'].values()) or source['state_max_error'] > 1e-7:
        raise ValueError('原默认模型尚未复现')
    output = {'scope': source['scope'], 'scores_sha256': sha256((args.root/'scores.json').read_bytes()).hexdigest(),
              'reproduction': source['reproduction'], 'state_max_error': source['state_max_error'], 'tasks': {}}
    lines = ['# 分类组件的固定消融与故障诊断', '',
             '默认提取版与原CoMind实际候选的全部网络参数和预测完全一致。下表是固定组件开发结果，尚未重新运行完整研究控制器。未来输出均由保存并重载的权重逐行产生。', '',
             '六项网络变体使用同样三个种子29、17、41；简单对照使用固定种子17。方括号为种子最小值与最大值，不是置信区间。完整分数按原探索及确认采用规则计算，未来分数是该候选直接交付表现。', '']
    for task, rows in source['results'].items():
        by, failures = {}, []
        for row in rows:
            spec, name = row['spec'], row['spec']['name']
            folder = args.root/task/name
            if not row['success']:
                failures.append(name)
                continue
            for filename, checksum in [('model.pkl', row['weights_sha256']), ('predictions.npz', row['predictions_sha256'])]:
                if sha256((folder/filename).read_bytes()).hexdigest() != checksum:
                    raise ValueError('消融模型或预测身份改变')
            variant = spec.get('variant', name)
            by.setdefault(variant, []).append(row)
        tasks = {}
        for variant, records in by.items():
            item = {'trials': len(records), 'adopted': sum(r['adopted_by_original_rule'] for r in records),
                    'accepted_complete': describe([r['accepted_complete']['score'] for r in records]),
                    'raw_complete': describe([r['raw_complete']['score'] for r in records]),
                    'source_parent_label_agreement': records[0]['source_parent_label_agreement']}
            for phase in ['train', 'explore', 'confirm', 'future']:
                keys = ['score', 'class_2_recall', 'positive_rows', 'detected_positive', 'false_alarms', 'prevalence']
                if all('average_precision' in r['phases'][phase] for r in records):
                    keys += ['average_precision']
                item[phase] = {key: describe([r['phases'][phase][key] for r in records]) for key in keys
                               if all(r['phases'][phase][key] is not None for r in records)}
            tasks[variant] = item
        output['tasks'][task] = {'variants': tasks, 'failures': failures}
        lines += ['## '+task.upper(), '', '| 固定变体 | 采用 / 次数 | 完整Macro-F1 | 未来Macro-F1 | 未来检出 / 误报 |',
                  '|---|---:|---|---|---|']
        for name, row in tasks.items():
            future = row['future']
            lines.append(f"| {NAMES[name]} | {row['adopted']}/{row['trials']} | {show(row['accepted_complete'])} | {show(future['score'])} | {show(future['detected_positive'], 1)} / {show(future['false_alarms'], 1)} |")
        reference = tasks['source-parent']
        lines += ['', f"源训练父预测与真实标签一致率：{reference['source_parent_label_agreement']:.4%}。未来类别2比例：{reference['future']['prevalence']['mean']:.4%}。",
                  '', '| 有概率接口的模型 | 源AP | 探索AP | 确认AP | 未来AP |', '|---|---|---|---|---|']
        for name, row in tasks.items():
            if all('average_precision' in row[p] for p in ['train', 'explore', 'confirm', 'future']):
                lines.append('| '+NAMES[name]+' | '+' | '.join(show(row[p]['average_precision']) for p in ['train', 'explore', 'confirm', 'future'])+' |')
        lines += ['', '失败：'+('、'.join(failures) if failures else '无')+'。', '']
    atomic_json(args.root/'summary.json', output)
    lines += ['原始逐种子分数、模型路径和真实预测身份见[scores.json](scores.json)；全部配置见[frozen-variants.json](frozen-variants.json)。', '']
    (args.root/'实验结果.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False))


if __name__ == '__main__':
    main()
