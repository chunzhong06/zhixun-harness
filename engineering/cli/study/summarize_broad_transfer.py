"""汇总四类完整对照；按任务保留单位、完整与发布后分数及真实调用，不挑选内部模型。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.broad_transfer_task import CLASSIFICATION, metric, verify_core
from harness.runtime.storage import atomic_json

NAMES = {'sgp40': '跨芯片SGP40', 'secom': '跨时段SECOM', 'wind': '跨风机Kelmarsh', 'ncmapss_ds06': '跨设备N-CMAPSS DS06'}


def main():
    parser = ArgumentParser(); parser.add_argument('--root', type=Path, required=True); parser.add_argument('--bundle', type=Path, required=True)
    args = parser.parse_args(); verify_core(); rows = []
    for task, name in NAMES.items():
        folder, bundle = args.root/task, args.bundle/task
        identity = json.loads((bundle/'task-identity.json').read_bytes())
        for filename, checksum in identity['task_files'].items():
            if sha256((bundle/'task'/filename).read_bytes()).hexdigest() != checksum:
                raise ValueError('合法任务输入发生变化')
        for filename, checksum in identity['private_files'].items():
            if sha256((bundle/filename).read_bytes()).hexdigest() != checksum:
                raise ValueError('父版、外层输入或答案身份改变')
        original = json.loads((folder/'comparison.json').read_bytes())
        oracle = json.loads((bundle/'oracle.json').read_bytes())
        with np.load(bundle/'private-answers.npz', allow_pickle=False) as z:
            truth, ids = z['y'], z['ids']
        with np.load(bundle/'outer-inputs.npz', allow_pickle=False) as z:
            future = z['future']
        arms = {}
        for arm in ['project', 'comind']:
            report = original['arms'][arm]
            envelope = json.loads((folder/arm/'input-identity.json').read_bytes())
            if envelope['bundle_identity_sha256'] != sha256((bundle/'task-identity.json').read_bytes()).hexdigest():
                raise ValueError('两臂输入未绑定当前任务包')
            # 驱动登记的是逻辑文本SHA；Windows写盘的CRLF经读回还原后再核对。
            description = (folder/arm/'task_description.txt').read_text(encoding='utf-8')
            if sha256(description.encode()).hexdigest() != envelope['description_sha256']:
                raise ValueError('任务说明身份改变')
            decision = json.loads((folder/arm/'decision.json').read_bytes())
            selected = decision['candidate']
            if arm == 'comind' and selected is not None:
                author = json.loads((folder/arm/'workspace/author-selection.json').read_bytes())
                if author['selected_execution'] != decision['selected_execution'] or author['candidate'] != selected:
                    raise ValueError('CoMind作者选中模型与外层提名不一致')
            if selected is not None:
                for weights in [Path(selected['path']).with_suffix('.pkl'), folder/arm/'serving/model.pkl']:
                    if sha256(weights.read_bytes()).hexdigest() != selected['weights_sha256']:
                        raise ValueError('选中模型与实际发布权重不一致')
            notes_path = folder/arm/'transport-notes.json'
            notes = json.loads(notes_path.read_bytes()) if notes_path.exists() else []
            if sha256((folder/arm/'issued.npz').read_bytes()).hexdigest() != report['issued_sha256']:
                raise ValueError('已发布预测SHA变化')
            with np.load(folder/arm/'issued.npz', allow_pickle=False) as z:
                if not np.array_equal(z['ids'], ids):
                    raise ValueError('交付身份改变')
                values = z['actual']
                if not np.array_equal(values[~future], z['baseline'][~future]):
                    raise ValueError('维护前既有交付被改写')
            groups = np.asarray(oracle['service_groups'])
            complete = metric(task, truth, values, groups)
            after = metric(task, truth[future], values[future], groups[future])
            for phase, measured in [('complete', complete), ('after_publication', after)]:
                recorded = report[phase]['actual']
                if abs(measured['score']-recorded['score']) > 1e-12 or measured['groups'] != recorded['groups']:
                    raise ValueError('完整或发布后分组分数不能复算')
            score = complete['score']
            arms[arm] = {'complete': score, 'after_publication': after['score'],
                         'model_calls': report['model_calls'], 'executions': report['executions'], 'adopted': report['adopted'],
                         'failed_executions': report['failed_executions'], 'execution_seconds': report['execution_seconds'],
                         'research_completed': report['research_completed'], 'delivery_failure': report['delivery_failure'],
                         'extra_transport_turns': sum(n.get('extra_agent_turns', 0) for n in notes),
                         'transport_notes': notes,
                         'groups': report['complete']['actual']['groups']}
            if task in CLASSIFICATION:
                arms[arm]['class_2_recall'] = report['complete']['actual']['class_2_recall']
                arms[arm]['after_publication_class_2_recall'] = report['after_publication']['actual']['class_2_recall']
            if task == 'secom':
                arms[arm]['future_failures'] = int((truth[future] == 2).sum())
                arms[arm]['future_detected_failures'] = int(((truth[future] == 2) & (values[future] == 2)).sum())
                arms[arm]['future_false_alarms'] = int(((truth[future] == 1) & (values[future] == 2)).sum())
        delta = arms['project']['complete']-arms['comind']['complete']
        advantage = delta if task in CLASSIFICATION else -delta
        verdict = '持平' if abs(advantage) < 1e-12 else ('我方领先' if advantage > 0 else 'CoMind领先')
        rows.append({'task': task, 'name': name, 'metric': 'Macro-F1 ↑' if task in CLASSIFICATION else 'RMSE ↓',
                     'rows': len(ids), 'future_rows': int(future.sum()), 'arms': arms, 'conclusion': verdict,
                     'advantage': advantage, 'relative_advantage_percent': 100*advantage/arms['comind']['complete'],
                     'comparison_sha256': sha256((folder/'comparison.json').read_bytes()).hexdigest()})
    atomic_json(args.root/'summary.json', {'scope': '旧资料上的当前冻结设计跨数据族完整复验', 'cases': rows})
    lines = ['# 四类既有资料的完整跨域对照', '',
             '当前完整核心在四类任务全部完成前保持冻结；双方独立运行研究、共同确认、实际权重发布和完整交付。四类资料均有历史研发暴露，本轮评价跨任务适配后的表现。', '',
             '| 类型与数据 | 完整交付数 | 指标 | Harness | CoMind | 结果 |', '|---|---:|---|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['rows']} | {r['metric']} | {r['arms']['project']['complete']:.6f} | {r['arms']['comind']['complete']:.6f} | {r['conclusion']} |")
    wind = next(r for r in rows if r['task'] == 'wind')
    lines += ['', f"风机表中记录数值方向，RMSE相对差仅{wind['relative_advantage_percent']:.5f}%，实际解释为精度基本持平。"]
    lines += ['', '## 实际采用与投入', '', '| 数据 | 发布后Harness/CoMind | 角色请求 | 程序执行 | 采用新模型 |', '|---|---|---|---|---|']
    for r in rows:
        p, c = r['arms']['project'], r['arms']['comind']
        adoption = ' / '.join('采用' if v['adopted'] else '保持父版' for v in [p, c])
        lines.append(f"| {r['name']} | {p['after_publication']:.6f} / {c['after_publication']:.6f} | {p['model_calls']} / {c['model_calls']} | {p['executions']} / {c['executions']} | {adoption} |")
    extra = [(r['name'], arm, r['arms'][arm]['extra_transport_turns']) for r in rows for arm in ['project', 'comind'] if r['arms'][arm]['extra_transport_turns']]
    transport = '；'.join(f"{name}的{'Harness' if arm == 'project' else 'CoMind'}额外{n}轮" for name, arm, n in extra) if extra else '无'
    saved_by_root = [(r['name'], arm) for r in rows for arm in ['project', 'comind']
                     if any(n.get('root_persisted_verbatim') for n in r['arms'][arm]['transport_notes'])]
    if saved_by_root:
        transport += '；主控逐字保存既有原响应、不增加模型轮次：' + '、'.join(name+' '+('Harness' if arm == 'project' else 'CoMind') for name, arm in saved_by_root)
    lines += ['', '两组顺序均为Harness / CoMind。角色请求数是研究请求数；机械传输补充轮次另计：'+transport+'。失败保留在各任务报告，单次搜索和少量物理设备不据此推算重新搜索的胜率。分类与回归分别比较，不跨任务平均不同单位分数。', '',
              '## 质检任务的实际识别能力', '']
    quality = next(r for r in rows if r['task'] == 'secom')['arms']
    p, c = quality['project'], quality['comind']
    lines.append(f"SECOM发布后共有{p['future_failures']}个不合格品，Harness检出{p['future_detected_failures']}个、误报{p['future_false_alarms']}个；CoMind检出{c['future_detected_failures']}个、误报{c['future_false_alarms']}个。主指标的相对领先需要结合这些实际识别结果评价。")
    lines += ['', '## 原始依据', '']
    for r in rows:
        lines.append(f"- [{r['name']}：完整分组、父版、直接候选与实际交付]({r['task']}/comparison.json)")
    (args.root/'实验结果.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps([{'task': r['task'], 'conclusion': r['conclusion']} for r in rows], ensure_ascii=False))


if __name__ == '__main__':
    main()
