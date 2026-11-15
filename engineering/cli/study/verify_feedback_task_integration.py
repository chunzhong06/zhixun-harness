"""真实训练验证新共同工具与旧发放等价；脚本驱动只验证调度，不冒充模型调用。"""
from argparse import ArgumentParser
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'engineering/src'))
import cloudpickle
import numpy as np
import pandas as pd
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.feedback_task import prepare_feedback_input
from harness.integrations.free_research_driver import run_free_research
from harness.models.feedback_training import FeedbackTrainingLedger
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'
DEST = BASE / 'feedback-task-integration'
ARMS = ['source_only', 'cumulative_refit', 'recent_mix_refit']
WORKER = ROOT / 'engineering/cli/study/execute_author_program.py'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def execute_script(folder, source):
    path = folder / 'check.py'
    path.write_text(source, encoding='utf-8')
    result = subprocess.run([sys.executable, '-B', str(WORKER), '--script', str(path)], cwd=folder,
                            capture_output=True, text=True, encoding='utf-8', timeout=90)
    (folder / 'stderr.log').write_text(result.stderr, encoding='utf-8')
    (folder / 'stdout.log').write_text(result.stdout, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    return result


class ScriptedIntegrationBroker:
    """固定两步工程替身；记录明确区分脚本交换与真正的模型服务调用。"""
    max_calls = 2
    prior_model_invocations = 0
    reused_model_calls = 0

    def __init__(self, executor, folder):
        self.executor, self.folder, self.calls = executor, folder, 0

    async def exchange(self, kind, payload):
        if kind != 'model':
            raise ValueError('工程替身不执行搜索')
        self.calls += 1
        context = json.loads(payload['request']['contents'][0]['parts'][0]['text'])
        atomic_json(self.folder / f'scripted-request-{self.calls}.json', payload)
        if self.calls == 1:
            refs = {r['name'] for r in context['available_references']}
            if not set(ARMS + ['feedback-comparison']).issubset(refs):
                raise ValueError('任务能力没有进入共同驱动的参考动作')
            action = {'action': 'reference', 'name': 'feedback-comparison', 'hypothesis': '工程验证三种合法训练策略与原发放等价'}
        elif self.calls == 2:
            valid = [r for r in self.executor.records if not r['returncode']]
            if len(valid) != 1 or len(valid[0]['candidates']) != 3:
                raise ValueError('共同执行器未返回三个实际候选')
            chosen = min(valid[0]['candidates'], key=lambda c: c['score'])
            action = {'action': 'finish', 'execution': valid[0]['sequence'],
                      'candidate': Path(chosen['path']).stem, 'reason': '只核验真实候选提名和交付身份'}
        else:
            raise ValueError('工程替身超过两步')
        atomic_json(self.folder / f'scripted-response-{self.calls}.json', action)
        return {'text': json.dumps(action, ensure_ascii=False)}


async def verify_task(task, previous, count):
    out = DEST / task
    out.mkdir(parents=True, exist_ok=False)
    bundle = ROOT / 'results/experiments/local-defaults/whole-maintenance/broad-transfer-inputs' / task
    identity = read(bundle / 'task-identity.json')
    for key, base in [('task_files', bundle / 'task'), ('private_files', bundle)]:
        for name, digest in identity[key].items():
            if sha256((base / name).read_bytes()).hexdigest() != digest:
                raise ValueError('源任务身份改变')
    last = previous / f'batch-{count:02}'
    state = read(previous / f'batch-{count-1:02}/completed.json')['after']['ledger']
    ledger = FeedbackTrainingLedger.restore(state)
    issue = read(last / 'issue.json')
    with np.load(bundle / 'outer-inputs.npz', allow_pickle=False) as z:
        values = z['service_x']
    with np.load(bundle / 'private-answers.npz', allow_pickle=False) as z:
        ids, truth = z['ids'].tolist(), z['y']
    features = dict(zip(ids, values, strict=True))
    answers = dict(zip(ids, truth, strict=True))
    chosen = issue['ids'][:128]
    middle = len(chosen)//2
    validation_ids, confirmation_ids = chosen[:middle], chosen[middle:]
    manifest = prepare_feedback_input(bundle, out / 'task', ledger, features, validation_ids, confirmation_ids)
    if set(read(out / 'task/feedback-identities.json')['ids']).intersection(chosen):
        raise ValueError('探索确认身份进入训练')
    def scored(pred):
        y = np.asarray([answers[i] for i in validation_ids])
        if task == 'secom':
            from sklearn.metrics import f1_score
            return {'loss': 1-float(f1_score(y, pred, labels=[1, 2], average='macro', zero_division=0))}
        return {'loss': float(np.sqrt(np.mean((y-pred)**2)))}
    workspace = out / 'workspace'
    workspace.mkdir()
    executor = AuthorExecutor(out / 'executions', workspace,
             {'validation_y': np.asarray([answers[i] for i in validation_ids]),
              'confirmation_y': np.asarray([answers[i] for i in confirmation_ids])}, scorer=scored)
    broker = ScriptedIntegrationBroker(executor, out)
    selected = await run_free_research('共同接口工程核验；仅使用本次任务包声明的训练权限。', out / 'task', workspace,
                  executor, broker, instruction='这是明确标记的脚本驱动工程验证；不调用语言模型。')
    candidates = executor.records[0]['candidates']
    # 再启动没有评分答案的进程，用保存模型对原末批全部行作实际交付。
    delivery = out / 'delivery'
    delivery.mkdir()
    columns = read(bundle / 'task/task.json')['feature_columns']
    pd.DataFrame([features[i] for i in issue['ids']], columns=columns).to_csv(delivery / 'service.csv', index=False)
    for candidate in candidates:
        shutil.copyfile(Path(candidate['path']).with_suffix('.pkl'), delivery / (candidate['label']+'.pkl'))
    source = '''from pathlib import Path
import cloudpickle
import pandas as pd
import numpy as np
frame=pd.read_csv('service.csv',float_precision='round_trip')
predictions={}
for arm in ['source_only','cumulative_refit','recent_mix_refit']:
    model=cloudpickle.loads(Path(arm+'.pkl').read_bytes())
    predictions[arm]=np.concatenate([model.predict(frame.iloc[i:i+128]) for i in range(0,len(frame),128)])
np.savez_compressed('issued.npz',**predictions)
'''
    execute_script(delivery, source)
    if task == 'secom':
        expected = read(last / 'predictions.json')
    else:
        with np.load(last / 'predictions.npz', allow_pickle=False) as z:
            expected = {k: z[k] for k in z.files}
    errors = {}
    with np.load(delivery / 'issued.npz', allow_pickle=False) as actual:
        for arm in ARMS:
            errors[arm] = float(np.max(np.abs(actual[arm]-expected[arm])))
    if max(errors.values()) > 1e-10:
        raise ValueError('共同工具改变了原训练策略或实际发放：'+str(errors))
    # 已绑定的权限不能因生成代码重写声明而扩大，已绑定标签也不能替换。
    with TemporaryDirectory(prefix='permission-check-', dir=out) as tmp:
        scratch = Path(tmp)
        shutil.copytree(out / 'task', scratch / 'input')
        execute_script(scratch, '''from pathlib import Path
from harness_task import load_feedback_training
p=Path('input/feedback.csv')
p.write_text(p.read_text(encoding='utf-8')+'\\n',encoding='utf-8')
try: load_feedback_training()
except ValueError: print('bound_input_edit_rejected')
else: raise ValueError('修改后的训练文件仍被接受')
''')
    result = {'task': task, 'feedback_rows': manifest['feedback_rows'], 'actual_candidates': len(candidates),
              'selected_candidate': selected['candidates'][0]['label'], 'last_batch_rows': len(issue['ids']),
              'prediction_max_absolute_difference': errors, 'source_identity_and_role_separation_verified': True,
              'bound_input_edit_rejected': True, 'scripted_exchanges': broker.calls, 'actual_model_calls': 0}
    atomic_json(out / 'verification.json', result)
    return result


def verify_disabled_tasks():
    records = []
    root = ROOT / 'results/experiments/local-defaults/whole-maintenance/frontier-2026-inputs'
    for task in ['gas', 'bearing', 'engine']:
        source = root / task / 'task'
        if not source.is_dir():
            raise ValueError('缺少官方任务资料：'+str(source))
        with TemporaryDirectory(prefix='official-permission-', dir=DEST) as tmp:
            work = Path(tmp)
            shutil.copytree(source, work / 'input')
            result = execute_script(work, '''from pathlib import Path
import json
from harness_task import training_access,load_feedback_training,bind_training_access
assert training_access()['enabled'] is False
try: load_feedback_training()
except PermissionError: pass
else: raise ValueError('官方任务获得了反馈训练权限')
Path('input/feedback-training.json').write_text(json.dumps({'schema_version':1,'enabled':True}),encoding='utf-8')
try: bind_training_access('input')
except RuntimeError: pass
else: raise ValueError('生成程序重新绑定了宿主权限')
print('default_denied_and_rebind_rejected')
''')
            records.append({'task': task, 'feedback_training_disabled': True, 'rebind_rejected': True,
                            'checked_original_task_metadata': sha256((source/'task.json').read_bytes()).hexdigest()})
    return records


async def main():
    DEST.mkdir(parents=True, exist_ok=False)
    files = ['engineering/src/harness/integrations/feedback_task.py',
             'engineering/src/harness/integrations/author_task_runtime.py',
             'engineering/src/harness/integrations/free_research_driver.py',
             'engineering/src/harness/models/feedback_training.py',
             'engineering/cli/study/execute_author_program.py',
             'engineering/cli/study/verify_feedback_task_integration.py',
             'engineering/configs/experiments/feedback-task-integration.json']
    for name in files:
        target = DEST / 'frozen-code' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    results = []
    for task, previous, count in [('secom', 'feedback-training-secom-run415', 8), ('wind', 'feedback-training-wind', 9)]:
        results.append(await verify_task(task, BASE / previous, count))
    official = verify_disabled_tasks()
    summary = {'passed': True, 'tasks': results, 'official_tasks': official, 'actual_model_calls': 0,
               'new_performance_claim': False}
    atomic_json(DEST / 'verification.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
