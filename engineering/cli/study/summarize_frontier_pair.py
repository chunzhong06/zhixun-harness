"""核验更新工具后的完整官方任务配对，保留此前六项基线。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from engineering.cli.workflow import public_maintenance as public
from harness.integrations.official_research_task import metric
from harness.runtime.storage import atomic_json


def exploration_duplicates(records):
    """统计逐条探索预测完全相同的复评；它们可能是有意的控制，不自动算失败。"""
    seen, duplicates = {}, []
    for record in records:
        for candidate in record['candidates']:
            path = Path(candidate['path'])
            with np.load(path.with_suffix('.npz'), allow_pickle=False) as z:
                values = np.asarray(z['validation'], dtype='<f8')
            key = (values.shape, sha256(values.tobytes()).hexdigest())
            reference = {'execution': record['sequence'], 'candidate': path.stem}
            if key in seen:
                duplicates.append({'first': seen[key], 'repeated': reference})
            else:
                seen[key] = reference
    return duplicates


def reference_uses(work, records):
    actions = work/'workspace/project-decisions.json'
    if not actions.exists():
        return []
    executed = iter(records)
    used = []
    for action in json.loads(actions.read_bytes()):
        if action['action'] not in ('experiment', 'reference'):
            continue
        record = next(executed)
        assert sha256(action['python_source'].encode()).hexdigest() == record['source_sha256']
        if action['action'] == 'reference':
            used.append({'name': action['name'], 'execution': record['sequence'],
                         'source_sha256': record['source_sha256'], 'returncode': record['returncode']})
    assert next(executed, None) is None
    return used


def model_text_volume(work):
    """统计已保存的模型消息文本量；这是字符量，不能当作token、费用或内部推理量。"""
    requests = sorted((work/'queue/requests/mle-model').glob('*.json'))
    request_chars, response_chars, missing = 0, 0, []
    for path in requests:
        envelope = json.loads(path.read_bytes())
        request = envelope['context']['payload']['request']
        request_chars += len(json.dumps(request, ensure_ascii=False, sort_keys=True))
        response = work/'queue/responses/mle-model'/path.name
        if response.exists():
            response_chars += len(json.loads(response.read_bytes())['plan']['text'])
        else:
            missing.append(path.name)
    return {'request_files': len(requests), 'serialized_request_characters': request_chars,
            'response_text_characters': response_chars, 'missing_response_files': missing,
            'scope': '持久化请求与最终响应字符量，不是token、费用或内部推理量'}


def summarize(bundle, folder, task, comind=None, output=None):
    """可复用已完成的同条件作者运行；保留其原目录并重新核对任务和工具身份。"""
    source = bundle/task
    identity = json.loads((source/'task-identity.json').read_bytes())
    oracle = json.loads((source/'oracle.json').read_bytes())
    archive = Path(identity['archive'])
    assert sha256(archive.read_bytes()).hexdigest() == identity['archive_sha256']
    package = public.PACKAGES[task][2](archive)
    answers = {r.sample_id: r.label for r in package.validation}
    truth = np.asarray([answers[k] for k in oracle['service_ids']])
    groups = np.asarray(oracle['service_groups'])
    with np.load(source/'outer-inputs.npz', allow_pickle=False) as z:
        future, baseline = z['future'], z['baseline']
    rows, common = {}, []
    for arm in ('project', 'comind'):
        work = comind if arm == 'comind' and comind is not None else folder/arm
        summary = json.loads((work/'scores.json').read_bytes())
        decision = json.loads((work/'decision.json').read_bytes())
        frozen = json.loads((work/'input-identity.json').read_bytes())
        assert frozen['bundle_identity_sha256'] == sha256((source/'task-identity.json').read_bytes()).hexdigest()
        description = (work/'task_description.txt').read_text(encoding='utf-8')
        assert frozen['description_sha256'] == sha256(description.encode()).hexdigest()
        common.append((frozen['bundle_identity_sha256'], frozen['description_sha256'], frozen['max_model_calls']))
        assert summary['model_calls'] <= frozen['max_model_calls']
        candidate = decision['candidate']
        assert sha256((work/'serving/model.pkl').read_bytes()).hexdigest() == candidate['weights_sha256']
        records = json.loads((work/'executions/index.json').read_bytes())
        selected = next(r for r in records if r['sequence'] == decision['selected_execution'])
        assert selected['source_sha256'] == decision['source_sha256']
        assert sha256(Path(selected['source']).read_text(encoding='utf-8').encode()).hexdigest() == selected['source_sha256']
        matches = [(i, c) for i, c in enumerate(selected['candidates'])
                   if Path(c['path']).name == Path(candidate['path']).name]
        assert len(matches) == 1, '提名必须绑定执行内唯一的实际候选'
        selected_index, original = matches[0]
        assert original['weights_sha256'] == candidate['weights_sha256']
        assert original['label'] == candidate['label'] and original['feedback'] == candidate['feedback']
        with np.load(work/'issued.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['ids'], oracle['service_ids'])
            np.testing.assert_array_equal(z['baseline'], baseline)
            np.testing.assert_array_equal(z['direct'][future], np.load(work/'serving/prediction.npy'))
            np.testing.assert_array_equal(z['actual'][~future], baseline[~future])
            np.testing.assert_array_equal(z['actual'], z['direct'] if decision['adopted'] else baseline)
            complete = metric(task, truth, z['actual'], groups)
            after = metric(task, truth[future], z['actual'][future], groups[future])
        assert abs(complete['score']-summary['scores']['actual']['score']) < 1e-12
        rows[arm] = {'complete': complete, 'after_maintenance': after, 'calls': summary['model_calls'],
                     'run_directory': str(work),
                     'adopted': decision['adopted'], 'weights_sha256': candidate['weights_sha256'],
                     'source_sha256': selected['source_sha256'], 'executions': len(records),
                     'failed_executions': sum(r['returncode'] != 0 for r in records),
                     'candidate_count': sum(len(r['candidates']) for r in records),
                     'batched_executions': sum(len(r['candidates']) > 1 for r in records),
                     'selected_execution': selected['sequence'],
                     'selected_candidate': Path(candidate['path']).stem,
                     'selected_is_last': selected_index == len(selected['candidates'])-1,
                     'reference_executions': reference_uses(work, records),
                     'model_text_volume': model_text_volume(work),
                     'identical_exploration_predictions': exploration_duplicates(records),
                     'program_seconds': sum(r['execution_time'] for r in records)}
    assert common[0] == common[1], '新工具未向双方同等提供，不能作为同条件完整对照'
    report = {'task': task, 'scope': '更新共享工具后的官方开发完整配对', 'rows': rows,
              'relative_error_change_percent': 100*(rows['project']['complete']['score']/rows['comind']['complete']['score']-1),
              'same_inputs_and_tools_verified': True, 'predictions_and_weights_verified': True,
              'reused_author_run': comind is not None,
              'original_baseline_preserved': True}
    atomic_json(output or folder/'comparison.json', report)
    print(json.dumps({'task': task, 'scores': {a: r['complete']['score'] for a, r in rows.items()},
                      'calls': {a: r['calls'] for a, r in rows.items()},
                      'relative_error_change_percent': report['relative_error_change_percent']}, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--task', choices=('engine', 'bearing'), required=True)
    p.add_argument('--comind', type=Path, help='复用的作者运行原目录；仍核验双方输入、工具和调用上限')
    p.add_argument('--output', type=Path, help='可选独立核验输出，不覆盖原报告')
    args = p.parse_args()
    summarize(args.bundle.resolve(), args.folder.resolve(), args.task,
              args.comind.resolve() if args.comind else None,
              args.output.resolve() if args.output else None)
