"""用真实受限执行器验证批量候选、非末项提名和发布补训的模型绑定。"""
import asyncio
from argparse import ArgumentParser
import contextlib
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.candidate_selection import candidate_id, choose_candidate, match_publication
from harness.integrations.free_research_driver import run_free_research
from engineering.cli.workflow.mlestar_author import finish_delivery
from engineering.cli.study.audit_publication_effect import cases, selected_models
from harness.runtime.storage import atomic_json

SOURCE = '''from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from harness_task import load_data,evaluate_model
X,y,V,T=load_data()
first=LinearRegression().fit(X,y)
evaluate_model(first,label='line',feedback_gain=0.,feedback_hours=1.)
last=DummyRegressor(strategy='constant',constant=99.).fit(X,y)
evaluate_model(last,label='constant',feedback_gain=0.,feedback_hours=1.)
'''


async def verify(output):
    parent = ROOT/'results/engineering/research-workflow'
    with TemporaryDirectory(prefix='batch-selection-check-', dir=parent) as temporary:
        folder = Path(temporary).resolve()
        assert folder.is_relative_to(parent.resolve())
        task = folder/'task'
        task.mkdir()
        features = np.arange(16., dtype=float)[:, None]
        labels = 2*features[:, 0]
        pd.DataFrame({'x': features[:6, 0], 'target': labels[:6]}).to_csv(task/'train.csv', index=False)
        pd.DataFrame({'x': features[6:10, 0]}).to_csv(task/'validation.csv', index=False)
        pd.DataFrame({'x': features[10:12, 0]}).to_csv(task/'test.csv', index=False)
        oracle = {'validation_y': labels[6:10], 'confirmation_y': labels[10:12],
                  'validation_hours': np.arange(6, 10), 'confirmation_hours': np.arange(10, 12),
                  'exploration_masks': [np.asarray([1, 1, 0, 0], dtype=bool), np.asarray([0, 0, 1, 1], dtype=bool)]}
        workspace = folder/'workspace'
        executor = AuthorExecutor(folder/'executions', workspace, oracle)

        class FixtureBroker:
            max_calls = 2
            prior_model_invocations = 0
            reused_model_calls = 0
            reused_search_calls = 0
            search_calls = 0

            def __init__(self):
                self.calls = 0

            async def exchange(self, kind, payload):
                assert kind == 'model'
                self.calls += 1
                if self.calls == 1:
                    context = json.loads(payload['request']['contents'][0]['parts'][0]['text'])
                    assert context['available_references'] == [{'name': 'known-two-models', 'source_sha256': sha256(SOURCE.encode()).hexdigest()}]
                    return {'text': json.dumps({'action': 'reference', 'name': 'known-two-models', 'hypothesis': '工程合成数据：第一候选正确，最后候选故意错误'})}
                context = json.loads(payload['request']['contents'][0]['parts'][0]['text'])
                assert [c['candidate_id'] for c in context['history'][0]['candidates']] == ['candidate-001', 'candidate-002']
                assert context['best_working_programs'][0]['best_candidate_id'] == 'candidate-001'
                assert context['history'][0]['hypotheses'] == ['工程合成数据：第一候选正确，最后候选故意错误']
                return {'text': json.dumps({'action': 'finish', 'execution': 1, 'candidate': 'candidate-001', 'reason': '工程核验明确提名第一项'})}

        broker = FixtureBroker()
        selected = await run_free_research('合成数据上的接口工程核验，不是模型能力实验', task, workspace, executor, broker,
                                          reference_sources={'known-two-models': SOURCE})
        assert selected['selection_mode'] == 'explicit_candidate'
        assert len(selected['candidates']) == 1 and candidate_id(selected['candidates'][0]) == 'candidate-001'
        original = executor.records[0]
        assert original['source_sha256'] == sha256(SOURCE.encode()).hexdigest()
        assert len(original['candidates']) == 2  # 不得覆盖完整原记录。
        assert candidate_id(choose_candidate(original)) == 'candidate-002'  # 旧动作兼容。
        rejected = []
        for bad in ['candidate-999', '../candidate-001', 1]:
            try:
                choose_candidate(original, bad)
            except ValueError:
                rejected.append(str(bad))
            else:
                raise AssertionError('非法候选ID被接受')
        history = np.arange(6, 12)
        service = np.arange(6, 16)
        windows = [(np.arange(16) >= start) & (np.arange(16) < start+2) for start in (6, 8, 10)]
        args = SimpleNamespace(arm='project', site='synthetic-fixture', start='synthetic', frontier_task=False)
        with contextlib.redirect_stdout(io.StringIO()):
            finish_delivery(args, folder, features, ['x'], labels, history, service, windows, 12,
                np.zeros(len(service)), 'fixture', task, executor, workspace, broker, selected, labels,
                summary_extra={'scope': '合成数据接口测试，不能作为科研性能结果', 'model_calls': 0,
                               'total_model_invocations': 0, 'synthetic_driver_actions': 2})
        publication = json.loads((folder/'publication.json').read_bytes())
        published = executor.records[-1]
        assert len(published['candidates']) == 2
        first = choose_candidate(published, 'candidate-001')
        assert publication['published_weights_sha256'] == first['weights_sha256']
        assert sha256((folder/'serving/input/model.pkl').read_bytes()).hexdigest() == first['weights_sha256']
        np.testing.assert_allclose(np.load(folder/'serving/prediction.npy'), labels[12:], rtol=0, atol=1e-10)
        _, before, after = selected_models(folder, 'project')
        assert candidate_id(before) == candidate_id(after) == 'candidate-001'
        altered = {**published, 'candidates': [{**first, 'label': 'changed'}]}
        try:
            match_publication(before, altered)
        except ValueError:
            pass
        else:
            raise AssertionError('补训后的候选标签变化未被拒绝')
    assert not folder.exists()
    # 原四项补训诊断的身份解析也须保持，仅读元数据和权重哈希。
    old_cases = list(cases(ROOT/'results/experiments/local-defaults/whole-maintenance/frontier-2026'))
    report = {'registered_reference_executed_unchanged': True, 'reference_hypothesis_preserved': True,
              'real_executor_candidates': 2, 'explicit_nonlast_nomination_verified': True,
              'original_record_preserved': True, 'legacy_last_candidate_compatible': True,
              'publication_refit_bound_to_selected_candidate': True, 'reloaded_future_predictions_match': True,
              'unknown_or_external_candidate_rejected': rejected, 'changed_publication_label_rejected': True,
              'original_publication_cases_verified': len(old_cases), 'temporary_workspace_removed': True,
              'new_model_calls': 0, 'synthetic_fit_calls': 4,
              'scope': '合成数据与真实受限执行器的工程验证，不是性能比较'}
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    asyncio.run(verify(p.parse_args().output.resolve()))
