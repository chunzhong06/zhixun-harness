"""从被接入错误中断前的真实命名空间恢复最后一轮的一条 CoMind 分支。

不重开探索，不改变已提出的方案。前两格、原第三格响应及其他分支保持，
只继续原配额内剩余三格（第三格不再调用模型）及作者报告/源码整理。
"""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import threading
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from harness.runtime.storage import atomic_json


def resume_branch(spec, *, cfg, author, coder_module, notebook, executor, evaluator):
    previous = Path(spec['previous'])
    branch = spec['branch']
    oldwork = previous / 'workspace/author' / branch / 'working'
    newbase = cfg.agent_workspace_dir / branch
    shutil.copytree(oldwork.parent, newbase)
    work = newbase / 'working'
    connection = work / 'kernel-connection.json'
    if connection.exists(): connection.unlink()
    records = [r for r in executor.records if r.get('coding_workspace') == str(oldwork)]
    if len(records) != spec['prior_cells']:
        raise ValueError('中断前单元格数量与捕获状态不同')
    cells = []
    for sequence, record in zip(spec['prior_requests'], records, strict=True):
        path = next((previous / 'queue/responses/mle-model').glob(f'{sequence:04d}-*.json'))
        answer = json.loads(path.read_bytes())['plan']['text']
        code = author.extract_fields(answer, ['code'])['code'][0]
        cells.append(notebook.CodeCell(code, record['returncode'] == 0))

    oldselection = json.loads((previous / 'workspace/author-selection.json').read_bytes())
    oldrecord = next(r for r in executor.records if r['sequence'] == oldselection['selected_execution'])
    lower = spec['is_lower_better']
    oldscore = oldselection['candidate']['task_metrics']['score']
    oldbest = SimpleNamespace(id=oldselection['report_id'],
        metric=author.MetricValue(oldscore, maximize=not lower),
        submission=Path(oldrecord['coding_workspace']) / 'best_submission/submission.csv',
        output_dir=Path(oldrecord['coding_workspace']), full_code=oldselection['full_author_source'])
    (cfg.agent_workspace_dir / 'submission').mkdir(parents=True, exist_ok=True)
    updater = author.MetricUpdater(cfg, oldbest.metric)
    cfg_coder = deepcopy(cfg)
    cfg_coder.agent_workspace_dir = work
    cfg_coder.agent_num_iterations_code_agents = spec['remaining_cells'] + 1
    draft = SimpleNamespace(id=branch, title=spec['title'], description=spec['description'],
                            code=spec['draft_code'], codebase_content=None, datasets=[])
    coder = coder_module.CodeAgent(cfg_coder, draft, lower, updater, evaluator(cfg_coder, lower), 'restored-' + branch)
    coder.jupyter_session.cells = notebook.JupyterNotebook(cfg_coder, cells)
    coder.start_time = time.time() - spec['prior_active_seconds']
    coder._post_initial_message = lambda: setattr(coder.llm, 'messages', deepcopy(spec['blocked_messages']))
    # 保留该分支原有榜首，而不是把先前有效结果在恢复时清空。
    if coder.best_submission_path.exists():
        submitted = pd.read_csv(coder.best_submission_path)['prediction'].to_numpy(dtype=float)
        matches = []
        for position, record in enumerate(records):
            if record['returncode']: continue
            for candidate in record['candidates']:
                with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as z:
                    prediction = z['confirmation']
                if prediction.shape == submitted.shape and np.allclose(prediction, submitted, rtol=0, atol=1e-8):
                    matches.append((position, candidate))
        if not matches: raise ValueError('该分支旧榜首没有对应模型')
        position, candidate = min(matches, key=lambda x: x[1]['score'])
        coder.best_metric = author.MetricValue(candidate['task_metrics']['score'], maximize=not lower)
        coder.best_code = notebook.JupyterNotebook(cfg_coder, deepcopy(cells[:position + 1]))
    thread = threading.Thread(target=updater.run)
    thread.start()
    try:
        report = coder.run()
    finally:
        updater.stop()
        thread.join()
    reports = oldselection['reports'] + [{'id': report.id, 'metric': str(report.metric)}]
    atomic_json(cfg.agent_workspace_dir.parent / 'branch-recovery-complete.json', {
        'origin': str(previous), 'branch': branch, 'retained_prior_cells': spec['prior_cells'],
        'continued_original_cells': spec['remaining_cells'], 'reused_model_response': spec['blocked_request'],
        'new_brainstorm_or_plans': False, 'comparison_uses_only_exploration': True,
        'total_finished_reports_including_seed': len(reports)})
    return SimpleNamespace(reports=[oldbest, report], ideas=oldselection['ideas']), reports
