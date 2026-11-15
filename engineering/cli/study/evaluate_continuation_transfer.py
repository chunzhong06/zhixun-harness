"""在其他任务的原强父版旁试用续训组件；完成选案、确认和逐条交付后评分。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'engineering/src')]
import numpy as np
import pandas as pd
from harness.integrations.author_executor import AuthorExecutor
from harness.integrations.research_broker import ACTOR
from harness.integrations.broad_transfer_task import metric, accepted, CLASSIFICATION
from harness.runtime.storage import atomic_json


def program(classification):
    return '''from copy import deepcopy
from pathlib import Path
import json
import cloudpickle
from sklearn.base import BaseEstimator
from harness.models.continuation_estimator import ContinuationEstimator
from harness_task import load_data,evaluate_model
X,y,V,T=load_data()
groups=json.loads(Path('input/task.json').read_text(encoding='utf-8'))['train_groups']
class KeepParent(BaseEstimator):
    def predict(self,X): return X.iloc[:,-1].to_numpy()
evaluate_model(KeepParent(),label='keep deployed parent')
model=ContinuationEstimator(classification=''' + repr(classification) + ''').fit(X,y)
stem=evaluate_model(model,label='source initial native learner')
base=cloudpickle.loads(Path('artifacts/'+stem+'.pkl').read_bytes())
for recipe in ['stable-fine-tuning','balanced-fine-tuning','expanded-balanced-fine-tuning']:
    candidate=deepcopy(base)
    candidate.continue_fit(X,y,groups,recipe=recipe)
    evaluate_model(candidate,label=recipe)
'''


def parent_family_program(classification):
    common = '''from copy import deepcopy
from pathlib import Path
from hashlib import sha256
import json
import cloudpickle
import numpy as np
from sklearn.base import BaseEstimator
from harness_task import load_data,evaluate_model
X,y,V,T=load_data()
parent=cloudpickle.loads(Path('input/parent.pkl').read_bytes())
columns=list(X.columns[:-1])
class ParentFamily(BaseEstimator):
    def __init__(self,model,columns): self.model=model;self.columns=columns
    def predict(self,X): return np.maximum(0.,self.model.predict(X[self.columns]))
for frame in (X,V,T):
    if not np.allclose(np.maximum(0.,parent.predict(frame[columns])),frame.iloc[:,-1],atol=1e-8,rtol=0):
        raise ValueError('实际父模型没有复现原父预测特征')
evaluate_model(ParentFamily(parent,columns),label='keep deployed parent')
records=[]
'''
    if classification:
        return common + '''for regularization in (0.01,0.1,10.0):
    model=deepcopy(parent)
    scaler=model[:-1]
    learner=model.steps[-1][1]
    initial_coefficients=sha256(learner.coef_.tobytes()+learner.intercept_.tobytes()).hexdigest()
    learner.set_params(C=regularization,warm_start=True,max_iter=1000)
    learner.fit(scaler.transform(X[columns]),y)
    records.append({'C':regularization,'initial_coefficients_sha256':initial_coefficients,
                    'semantics':'保留源端尺度及线性系数热启动；L-BFGS不保存跨fit优化器历史'})
    evaluate_model(ParentFamily(model,columns),label='parent logistic C='+str(regularization))
Path('parent-continuation.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
'''
    return common + '''def tree_id(model):
    return [[sha256(tree.nodes.tobytes()+tree.raw_left_cat_bitsets.tobytes()+tree.binned_left_cat_bitsets.tobytes()).hexdigest() for tree in stage] for stage in model._predictors]
original=tree_id(parent)
for extra in (48,96,288):
    model=deepcopy(parent)
    model.set_params(warm_start=True,max_iter=parent.n_iter_+extra)
    model.fit(X[columns],y)
    if tree_id(model)[:len(original)]!=original:
        raise ValueError('增量提升树改变了原父树')
    records.append({'requested_additional_iterations':extra,'original_iterations':len(original),
                    'actual_iterations':model.n_iter_,'original_tree_tensors_preserved':True})
    evaluate_model(ParentFamily(model,columns),label='parent boosting +'+str(extra))
Path('parent-continuation.json').write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
'''


PREDICT = '''from pathlib import Path
import cloudpickle
import numpy as np
import pandas as pd
model=cloudpickle.loads(Path('model.pkl').read_bytes())
data=pd.read_csv('service.csv')
prediction=np.asarray([float(model.predict(data.iloc[[i]])[0]) for i in range(len(data))])
if not np.isfinite(prediction).all(): raise ValueError('独立交付产生非有限预测')
np.save('prediction.npy',prediction)
'''


def run(task, bundle, out, family='native'):
    source = bundle / task
    identity = json.loads((source / 'task-identity.json').read_bytes())
    for kind, base in [('task_files', source / 'task'), ('private_files', source)]:
        for filename, digest in identity[kind].items():
            if sha256((base / filename).read_bytes()).hexdigest() != digest:
                raise ValueError('原合法数据或父版身份变化')
    out.mkdir(parents=True, exist_ok=False)
    oracle = json.loads((source / 'oracle.json').read_bytes())
    def measured(values): return metric(task, oracle['explore_y'], values, oracle['explore_groups'])
    with TemporaryDirectory(dir=out, prefix='temporary-workspace-') as tmp:
        workspace = Path(tmp)
        shutil.copytree(source / 'task', workspace / 'input')
        if family == 'parent': shutil.copy2(source / 'parent.pkl', workspace / 'input/parent.pkl')
        executor = AuthorExecutor(out / 'executions', workspace,
            {'validation_y': np.asarray(oracle['explore_y']), 'confirmation_y': np.asarray(oracle['confirm_y'])},
            scorer=measured)
        ACTOR.set('deterministic_continuation_component_screen')
        code = parent_family_program(task in CLASSIFICATION) if family == 'parent' else program(task in CLASSIFICATION)
        result = executor(code, workspace, 'candidate.py', 90)
    if result['returncode']: raise RuntimeError(result['stderr'][-4000:])
    candidates = executor.records[0]['candidates']
    # 初始强父版作为保持候选，不因为新学习器能训练就强制替换。
    selected = min(candidates, key=lambda candidate: candidate['score'])
    parent = [metric(task, oracle[name + '_y'], pd.read_csv(source / 'task' / filename).iloc[:, -1], oracle[name + '_groups'])
              for name, filename in [('explore', 'validation.csv'), ('confirm', 'test.csv')]]
    with np.load(Path(selected['path']).with_suffix('.npz'), allow_pickle=False) as prediction:
        checks = [metric(task, oracle[name + '_y'], prediction[key], oracle[name + '_groups'])
                  for name, key in [('explore', 'validation'), ('confirm', 'confirmation')]]
    adopted = all(accepted(task, a, b) for a, b in zip(parent, checks, strict=True))
    decision = {'policy': '事前固定程序池，按探索最优提名，再用原独立确认规则决定采用',
                'scope': '新组件的完整交付试点，不是新Luna或作者系统配对',
                'selected': selected, 'parent_metrics': parent, 'candidate_metrics': checks,
                'adopted': adopted, 'model_calls': 0,
                'source_bundle_identity_sha256': sha256((source / 'task-identity.json').read_bytes()).hexdigest()}
    atomic_json(out / 'decision.json', decision)
    with np.load(source / 'outer-inputs.npz', allow_pickle=False) as data:
        future, baseline, x = data['future'].copy(), data['baseline'].copy(), data['service_x'].copy()
    serving = out / 'serving'; serving.mkdir()
    weights = Path(selected['path']).with_suffix('.pkl')
    if sha256(weights.read_bytes()).hexdigest() != selected['weights_sha256']:
        raise ValueError('选中权重变化')
    shutil.copy2(weights, serving / 'model.pkl')
    columns = json.loads((source / 'task/task.json').read_bytes())['feature_columns']
    pd.DataFrame(x[future], columns=columns).to_csv(serving / 'service.csv', index=False)
    script = serving / 'predict.py'; script.write_text(PREDICT, encoding='utf-8')
    env = os.environ.copy(); env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    proc = subprocess.run([sys.executable, '-B', str(ROOT / 'engineering/cli/study/execute_author_program.py'), '--script', str(script)],
                          cwd=serving, env=env, capture_output=True, text=True, encoding='utf-8', timeout=90)
    (serving / 'stderr.log').write_text(proc.stderr, encoding='utf-8')
    if proc.returncode: raise RuntimeError(proc.stderr[-4000:])
    values = np.load(serving / 'prediction.npy', allow_pickle=False)
    if values.shape != (int(future.sum()),): raise ValueError('独立交付行数不足')
    direct = baseline.copy(); direct[future] = values
    actual = direct if adopted else baseline.copy()
    np.savez_compressed(out / 'issued.npz', ids=np.asarray(oracle['service_ids']), baseline=baseline, direct=direct, actual=actual)
    # 提案、确认及全部输出封存后再开服务答案，不用候选的未来得分回头改决定。
    with np.load(source / 'private-answers.npz', allow_pickle=False) as data:
        y, ids = data['y'].copy(), data['ids'].copy()
    if not np.array_equal(ids, np.asarray(oracle['service_ids'])): raise ValueError('服务身份顺序变化')
    groups = np.asarray(oracle['service_groups'])
    scores = {'complete': {name: metric(task, y, p, groups) for name, p in [('baseline', baseline), ('direct', direct), ('actual', actual)]},
              'after_publication': {name: metric(task, y[future], p[future], groups[future]) for name, p in [('baseline', baseline), ('direct', direct), ('actual', actual)]}}
    report = {'task': task, 'family': family, 'adopted': adopted, 'selected_label': selected['label'], 'scores': scores,
              'candidates': [{'label': c['label'], 'exploration_loss': c['score']} for c in candidates],
              'rows': len(y), 'future_rows': int(future.sum()), 'model_calls': 0,
              'source_only_initial_fit': family == 'native', 'continuation_programs': 3,
              'execution_seconds': result['execution_time'], 'issued_sha256': sha256((out / 'issued.npz').read_bytes()).hexdigest()}
    atomic_json(out / 'scores.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def main():
    parser = ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--family', choices=['native', 'parent'], default='native')
    args = parser.parse_args()
    result = [run(task, args.bundle.resolve(), args.output.resolve() / task, args.family) for task in ('secom', 'wind')]
    atomic_json(args.output / 'summary.json', {'tasks': result})


if __name__ == '__main__': main()
