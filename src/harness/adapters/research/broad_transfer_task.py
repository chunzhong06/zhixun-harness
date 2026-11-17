"""跨数据族任务的薄适配：共享评分、源码和任务说明，不修改冻结的研究控制器。"""
import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score

from harness.adapters.research.frontier_task import DELIVERY

ROOT = __import__("harness.paths", fromlist=["workspace_root"]).workspace_root()
CLASSIFICATION = {'sgp40', 'secom'}


def verify_core():
    manifest = ROOT / 'results/engineering/research-workflow/broad-transfer-frozen-core/manifest.json'
    for record in json.loads(manifest.read_bytes())['files']:
        if sha256((ROOT / record['source']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('宽域对照冻结核心改变：' + record['source'])


def metric(task, truth, prediction, groups):
    """分类固定两类；回归主指标按行汇总，同时保留每台设备和最差设备。"""
    y, p, g = np.asarray(truth), np.asarray(prediction), np.asarray(groups)
    if y.shape != p.shape or g.shape != y.shape or not len(y) or not np.isfinite(p).all():
        raise ValueError('评分行数或数值不合法')
    if task in CLASSIFICATION:
        if not np.isin(p, [1, 2]).all():
            raise ValueError('分类交付必须是类别1或2，不能交概率或回归值')
        score = lambda a, b: float(f1_score(a, b, labels=[1, 2], average='macro', zero_division=0))
    else:
        score = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
    by = {str(k): score(y[g == k], p[g == k]) for k in np.unique(g)}
    value = score(y, p)
    result = {'score': value, 'loss': 1-value if task in CLASSIFICATION else value,
              'groups': by, 'group_mean': float(np.mean(list(by.values()))),
              'worst': (min if task in CLASSIFICATION else max)(by.values()), 'rows': len(y)}
    if task in CLASSIFICATION:
        result['class_2_recall'] = float(recall_score(y, p, pos_label=2, zero_division=0))
    return result


def accepted(task, parent, candidate):
    # 延续官方分类严格提升和回归严格降低；寿命任务保留最差设备不退步1%的闸门。
    if task in CLASSIFICATION:
        return candidate['score'] > parent['score'] + 1e-12
    return (candidate['score'] < parent['score'] and
            (task != 'ncmapss_ds06' or candidate['worst'] <= 1.01*parent['worst']))


def sources(task):
    from harness.research.parent_reference import SOURCE
    shared = {'complete-parent': SOURCE}
    prefix = '''from harness_task import load_data,evaluate_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.ensemble import RandomForestClassifier,ExtraTreesRegressor
from sklearn.svm import SVC
X,y,V,T=load_data()
'''
    if task in CLASSIFICATION:
        models = {
            'balanced-logistic': "make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=17))",
            'balanced-forest': "RandomForestClassifier(n_estimators=96,max_depth=6,class_weight='balanced',n_jobs=1,random_state=17)",
            'balanced-svm': "make_pipeline(StandardScaler(),SVC(C=1.,gamma='scale',class_weight='balanced'))"}
    else:
        models = {'ridge': 'make_pipeline(StandardScaler(),Ridge(alpha=100.))',
                  'extra-trees': 'ExtraTreesRegressor(n_estimators=96,max_depth=6,n_jobs=1,random_state=17)'}
    for name, model in models.items():
        shared[name] = prefix + f"model={model}.fit(X,y)\nevaluate_model(model,label='{name}')\n"
    if task in CLASSIFICATION:
        # 精确复现、三种子消融和完整配对后登记为通用分类候选；不按数据集强制采用。
        shared['soft-state-classifier'] = (ROOT / 'src/harness/learning/models/soft_state_classifier.py').read_text(encoding='utf-8') + '''
from harness_task import load_data,evaluate_model
X,y,V,T=load_data()
model=SoftStateClassifier().fit(X,y)
evaluate_model(model,label='soft state classifier')
'''
    if task not in CLASSIFICATION:
        # 原有效组件直接提供给双方；不根据目标任务重调网络超参。
        shared['balanced-device-huber-residual'] = (ROOT / 'src/harness/learning/models/robust_residual_network.py').read_text(encoding='utf-8') + '''
from harness_task import load_data,evaluate_model
import json
from pathlib import Path
X,y,V,T=load_data()
groups=json.loads(Path('input/task.json').read_text(encoding='utf-8'))['train_groups']
model=RobustResidualNetwork().fit(X,y,groups)
evaluate_model(model,label='balanced device Huber residual')
'''
    return shared


def description(task, folder, reference_sources=None):
    metadata = json.loads((Path(folder) / 'task.json').read_bytes())
    return ('Task: frozen cross-domain maintenance ' + task + '.\n' + metadata['description'] + '\n'
            'Metric: ' + ('binary Macro-F1, classes 1 and 2; larger is better. Outer search loss is 1-Macro-F1.'
                          if task in CLASSIFICATION else 'RMSE, smaller is better.') + '\n'
            'The last input column parent_value is the common source-only parent prediction. '
            'Raw missing values were filled using source-training medians (zero for all-missing columns); '
            'present_* columns explicitly indicate actual measurements. No target labels were used in preprocessing. '
            'Feature names and source device/batch identities are in input/task.json. '
            'Both arms receive the same capabilities below; none has an observed score for this comparison.\n'
            + DELIVERY + '\nShared existing program sources:\n'
            + '\n'.join('### '+k+'\n```python\n'+v+'\n```' for k, v in
                        (sources(task) if reference_sources is None else reference_sources).items()))
