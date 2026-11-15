"""一次真实研究选择更新程序，再按当时反馈发布；最终分数由完整发放重算。"""
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'engineering/src'))
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.feedback_task import prepare_feedback_input, describe_training_access
from harness.integrations.free_research_driver import run_free_research, INSTRUCTION
from harness.integrations.research_broker import AuthorBroker
from harness.models.feedback_training import FeedbackTrainingLedger
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'
PRIOR = BASE / 'feedback-training-secom-run415'
DEST = BASE / 'feedback-research-secom'
BUNDLE = ROOT / 'results/experiments/local-defaults/whole-maintenance/broad-transfer-inputs/secom'
WORKER = ROOT / 'engineering/cli/study/execute_author_program.py'
ARMS = ['source_only', 'cumulative_refit', 'recent_mix_refit']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def macro(y, p):
    return float(f1_score(y, p, labels=[1, 2], average='macro', zero_division=0))


def execute(folder, code, timeout=90):
    script = folder / 'program.py'
    script.write_text(code, encoding='utf-8')
    started = time.monotonic()
    result = subprocess.run([sys.executable, '-B', str(WORKER), '--script', str(script)], cwd=folder,
                            capture_output=True, text=True, encoding='utf-8', timeout=timeout)
    (folder / 'stdout.log').write_text(result.stdout, encoding='utf-8')
    (folder / 'stderr.log').write_text(result.stderr, encoding='utf-8')
    receipt = {'returncode': result.returncode, 'seconds': time.monotonic()-started,
               'source_sha256': sha256(code.encode()).hexdigest()}
    atomic_json(folder / 'execution.json', receipt)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return receipt


async def main():
    DEST.mkdir(parents=True, exist_ok=False)
    sources = ['engineering/cli/study/run_feedback_research.py', 'engineering/src/harness/integrations/feedback_task.py',
               'engineering/src/harness/integrations/free_research_driver.py', 'engineering/src/harness/integrations/author_task_runtime.py',
               'engineering/src/harness/integrations/author_executor.py', 'engineering/src/harness/models/feedback_training.py',
               'engineering/cli/study/execute_author_program.py', 'engineering/configs/experiments/feedback-research-secom.json']
    for name in sources:
        copy = DEST / 'frozen-code' / name
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, copy)
    registration = read(BUNDLE / 'task-identity.json')
    for key, base in [('task_files', BUNDLE / 'task'), ('private_files', BUNDLE)]:
        for name, expected in registration[key].items():
            if sha256((base / name).read_bytes()).hexdigest() != expected:
                raise ValueError('原数据身份改变')
    task = read(BUNDLE / 'task/task.json')
    cuts = task['target_batch_cuts']
    with np.load(BUNDLE / 'outer-inputs.npz', allow_pickle=False) as z:
        x, parent_prediction = z['service_x'], z['baseline']
    with np.load(BUNDLE / 'private-answers.npz', allow_pickle=False) as z:
        ids, truth = z['ids'].tolist(), z['y']
    features = dict(zip(ids, x, strict=True))
    ledger = FeedbackTrainingLedger(task['train_ids'], allow_feedback_training=True)
    actual = []
    search_ledger = None
    for i in range(4):
        start, stop = cuts[i:i+2]
        folder = DEST / f'batch-{i+1:02}'
        folder.mkdir()
        issued = {'ids': ids[start:stop], 'prediction': parent_prediction[start:stop].tolist(), 'model': 'source_parent'}
        atomic_json(folder / 'issued.json', issued)
        actual.extend(issued['prediction'])
        ledger.issue(issued['ids'], sha256((folder / 'issued.json').read_bytes()).hexdigest())
        ledger.observe(issued['ids'], truth[start:stop])
        atomic_json(folder / 'ledger.json', ledger.state())
        if i == 1:
            search_ledger = FeedbackTrainingLedger.restore(ledger.state())
    train_input = DEST / 'search-task'
    prepare_feedback_input(BUNDLE, train_input, search_ledger, features, ids[cuts[2]:cuts[3]], ids[cuts[3]:cuts[4]])
    description = ('半导体质检二分类，class1=正常、class2=故障，指标为固定两类Macro-F1。源626行；前两批235行回标可训练，'
                   '第三批117行只供外层探索，第四批118行仅外层确认。前四批已经保持源父版。最多4次请求；'
                   '请实际检验已有工具或新程序，再明确提名候选。仅当探索和确认均严格胜过父版才采用。'
                   '采用后同一程序、候选ID和标签将在每个后续批次前，按新的合法已回标训练包重新执行；'
                   '发布时validation和test为空，只保存权重不返回训练内成绩。模型须可独立重载并处理单条输入。'
                   '原父parent.pkl是固定源模型，可调用load_parent_model；source_only选项用于保持。'
                   + describe_training_access(train_input))
    (DEST / 'description.txt').write_text(description, encoding='utf-8')
    workspace = DEST / 'workspace'
    workspace.mkdir()
    executor = AuthorExecutor(DEST / 'executions', workspace,
             {'validation_y': truth[cuts[2]:cuts[3]], 'confirmation_y': truth[cuts[3]:cuts[4]]},
             scorer=lambda prediction: {'loss': 1-macro(truth[cuts[2]:cuts[3]], prediction)})
    broker = AuthorBroker(DEST / 'queue', max_calls=4, timeout=180)
    instruction = INSTRUCTION.replace('两个探索周的真实RMSE用于比较', '外层真实Macro-F1用于比较，损失为1减Macro-F1')
    instruction = instruction.replace('最后确认周不返回', '第四批确认标签和成绩不返回')
    instruction += '\n本任务训练和发布时序以task_description为准；无需运行原接口中只针对伦敦的反馈校准。'
    selected, failure = None, None
    try:
        selected = await run_free_research(description, train_input, workspace, executor, broker, instruction=instruction)
    except Exception:
        failure = traceback.format_exc()
        (DEST / 'research-failure.log').write_text(failure, encoding='utf-8')
    parent_scores = [macro(truth[a:b], parent_prediction[a:b]) for a, b in [(cuts[2], cuts[3]), (cuts[3], cuts[4])]]
    adopted, candidate, checked = False, None, None
    if selected:
        candidate = selected['candidates'][0]
        with np.load(Path(candidate['path']).with_suffix('.npz'), allow_pickle=False) as z:
            checked = [macro(truth[cuts[2]:cuts[3]], z['validation']), macro(truth[cuts[3]:cuts[4]], z['confirmation'])]
        adopted = all(a > b+1e-12 for a, b in zip(checked, parent_scores, strict=True))
    actual_calls = sum((DEST / 'queue/responses/mle-model' / p.name).is_file()
                       for p in (DEST / 'queue/requests/mle-model').glob('*.json'))
    decision = {'adopted': adopted, 'candidate': candidate, 'candidate_scores': checked, 'parent_scores': parent_scores,
                'model_calls': actual_calls, 'model_requests': broker.calls,
                'executions': len(executor.records), 'research_failed': failure is not None}
    atomic_json(DEST / 'decision.json', decision)
    publication = []
    for i in range(4, 8):
        start, stop = cuts[i:i+2]
        folder = DEST / f'batch-{i+1:02}'
        folder.mkdir()
        prediction = parent_prediction[start:stop].copy()
        used, error, elapsed = 'source_parent', None, 0.
        if adopted:
            try:
                prepare_feedback_input(BUNDLE, folder / 'input', ledger, features, [], [])
                code = Path(selected['source']).read_text(encoding='utf-8')
                receipt = execute(folder, code)
                elapsed += receipt['seconds']
                stem = Path(candidate['path']).stem
                published = read(folder / 'artifacts' / (stem+'.json'))
                if published['label'] != candidate['label'] or published['feedback'] != candidate['feedback']:
                    raise ValueError('发布改变所选候选身份')
                weights = folder / 'artifacts' / (stem+'.pkl')
                if sha256(weights.read_bytes()).hexdigest() != published['weights_sha256']:
                    raise ValueError('发布权重摘要不符')
                delivery = folder / 'delivery'
                delivery.mkdir()
                shutil.copyfile(weights, delivery / 'model.pkl')
                pd.DataFrame(x[start:stop], columns=task['feature_columns']).to_csv(delivery / 'service.csv', index=False)
                pred_code = '''from pathlib import Path
import cloudpickle
import numpy as np
import pandas as pd
model=cloudpickle.loads(Path('model.pkl').read_bytes())
frame=pd.read_csv('service.csv',float_precision='round_trip')
values=np.asarray([float(model.predict(frame.iloc[[i]])[0]) for i in range(len(frame))])
if not np.isin(values,[1,2]).all():raise ValueError('实际交付必须是合法类别')
np.save('prediction.npy',values)
'''
                elapsed += execute(delivery, pred_code)['seconds']
                prediction = np.load(delivery / 'prediction.npy', allow_pickle=False)
                if prediction.shape != (stop-start,):
                    raise ValueError('发布预测不完整')
                used = published['weights_sha256']
            except Exception:
                error = traceback.format_exc()
                (folder / 'publication-failure.log').write_text(error, encoding='utf-8')
                prediction = parent_prediction[start:stop].copy()
                used = 'source_parent'
        issued = {'ids': ids[start:stop], 'prediction': prediction.tolist(), 'model': used,
                  'feedback_rows_before': sum(len(b['ids']) for b in ledger.feedback), 'publication_failed': error is not None}
        atomic_json(folder / 'issued.json', issued)
        actual.extend(prediction.tolist())
        ledger.issue(issued['ids'], sha256((folder / 'issued.json').read_bytes()).hexdigest())
        ledger.observe(issued['ids'], truth[start:stop])
        atomic_json(folder / 'ledger.json', ledger.state())
        publication.append({'batch': i+1, 'program_seconds': elapsed, 'failed': error is not None})
    # 选择和全段交付封存后，才合并固定对照及全部答案评分。
    outputs = {'source_only': parent_prediction, 'harness': np.asarray(actual)}
    for arm in ARMS[1:]:
        old = [read(PRIOR / f'batch-{i+1:02}/predictions.json')[arm] for i in range(8)]
        outputs['fixed_same_start_'+arm] = np.r_[parent_prediction[:cuts[4]], np.concatenate(old[4:])]
        outputs['fixed_early_'+arm] = np.concatenate(old)
    def score(y, pred):
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[1, 2]).ravel()
        return {'macro_f1': macro(y, pred), 'detected': int(tp), 'missed': int(fn), 'false_alarms': int(fp)}
    table = {a: {'complete': score(truth, pred), 'future': score(truth[cuts[4]:], pred[cuts[4]:])} for a, pred in outputs.items()}
    summary = {'rows': len(ids), 'future_rows': len(ids)-cuts[4], 'adopted': adopted, 'arms': table,
               'model_calls': actual_calls, 'model_requests': broker.calls, 'research_completed': selected is not None,
               'executions': len(executor.records), 'execution_seconds': sum(r['execution_time'] for r in executor.records),
               'publication': publication, 'fixed_control_source': str(PRIOR.relative_to(ROOT)),
               'new_fixed_model_training': False, 'protocol': 'feedback-research-secom'}
    np.savez_compressed(DEST / 'issued.npz', ids=np.asarray(ids), **outputs)
    atomic_json(DEST / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
