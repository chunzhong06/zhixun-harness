"""用已保存真实模型核验概率能力、类别语义和原有预测；不重新训练。"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'engineering/src'))
import numpy as np
from harness.integrations.feedback_task import prepare_feedback_input, describe_training_access
from harness.models.feedback_training import FeedbackTrainingLedger
from harness.runtime.storage import atomic_json

BASE = ROOT / 'results/experiments/contest-continuation'
DEST = BASE / 'feedback-probability-interface'
ARMS = ['source_only', 'cumulative_refit', 'recent_mix_refit']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


CODE = '''from pathlib import Path
import json
import cloudpickle
import numpy as np
import pandas as pd
from harness_task import training_access,evaluate_model
access=training_access()
assert '全部为1' in access['policy_definitions']['cumulative_refit'] or '都是1' in access['policy_definitions']['cumulative_refit']
frame=pd.read_csv('input/validation.csv',float_precision='round_trip')
models=[cloudpickle.loads(Path(a+'.pkl').read_bytes()) for a in ['source_only','cumulative_refit','recent_mix_refit']]
report={'capabilities':access['parent_capabilities'],'parent_prediction_equal':True}
for model in models:
    assert np.array_equal(model.predict(frame),np.maximum(0.,model.model.predict(frame[model.columns])))
if access['parent_capabilities']['predict_proba']:
    for model in models:
        assert np.array_equal(model.predict_proba(frame),model.model.predict_proba(frame[model.columns]))
        assert np.array_equal(model.classes_[model.predict_proba(frame).argmax(axis=1)],model.predict(frame))
    class Blend:
        def __init__(self,members): self.members=members;self.classes_=members[0].classes_
        def predict(self,X):
            p=np.mean([m.predict_proba(X) for m in self.members],axis=0)
            return self.classes_[np.argmax(p,axis=1)]
    evaluate_model(Blend(models[1:]),label='probability_blend_with_real_labels')
    class InvalidLabels:
        def predict(self,X): return np.zeros(len(X))
    try: evaluate_model(InvalidLabels(),label='invalid_zero_based_labels')
    except ValueError as exc:
        assert '真实类别' in str(exc)
    else: raise ValueError('错误类别索引未在探索阶段拒绝')
    report.update(probability_equal=True,class_mapping_equal=True,blend_saved=True,invalid_labels_rejected=True)
else:
    for model in models:
        try:model.predict_proba(frame)
        except TypeError:pass
        else:raise ValueError('回归后端伪装成分类概率')
    report['regression_probability_rejected']=True
Path('verification.json').write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8')
'''


def main():
    DEST.mkdir(parents=True, exist_ok=False)
    results = []
    for task, name, count in [('secom', 'feedback-training-secom-run415', 8), ('wind', 'feedback-training-wind', 9)]:
        bundle = ROOT / 'results/experiments/local-defaults/whole-maintenance/broad-transfer-inputs' / task
        prior = BASE / name
        state = read(prior / f'batch-{count-1:02}/completed.json')['after']['ledger']
        ledger = FeedbackTrainingLedger.restore(state)
        ids = read(bundle / 'oracle.json')['service_ids']
        with np.load(bundle / 'outer-inputs.npz', allow_pickle=False) as z:
            features = dict(zip(ids, z['service_x'], strict=True))
        current = read(prior / f'batch-{count:02}/issue.json')['ids']
        with TemporaryDirectory(prefix='runtime-check-', dir=DEST) as tmp:
            work = Path(tmp)
            prepare_feedback_input(bundle, work / 'input', ledger, features, current[:32], current[32:64])
            text = describe_training_access(work / 'input')
            if 'sample_weights全部为1' not in text or 'classes_' not in text:
                raise ValueError('权重或概率语义未向驱动说明')
            for arm in ARMS:
                shutil.copyfile(BASE / 'feedback-task-integration' / task / 'delivery' / (arm+'.pkl'), work / (arm+'.pkl'))
            script = work / 'check.py'
            script.write_text(CODE, encoding='utf-8')
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'engineering/cli/study/execute_author_program.py'), '--script', str(script)],
                                    cwd=work, capture_output=True, text=True, encoding='utf-8', timeout=90)
            (DEST / (task+'-stdout.log')).write_text(result.stdout, encoding='utf-8')
            (DEST / (task+'-stderr.log')).write_text(result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(result.stderr[-3000:])
            report = {'task': task, **read(work / 'verification.json')}
            results.append(report)
    summary = {'passed': True, 'tasks': results, 'new_training_calls': 0, 'actual_model_calls': 0}
    atomic_json(DEST / 'verification.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
