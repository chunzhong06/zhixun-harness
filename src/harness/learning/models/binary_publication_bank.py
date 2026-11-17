"""独立发布方自己的冻结分类器、源父版和公共森林工具，不接收对手的产物。"""
from hashlib import sha256
import json
from pathlib import Path
import time

import cloudpickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

from harness.adapters.storage.local import atomic_json

THRESHOLDS = (.01, .02, .03, .05, .075, .1, .15, .2, .3, .4, .5)
FAMILIES = ('parent', 'published', 'forest_parent', 'forest_raw')


def prepare_bank(folder, train, target, parent_weights, published_weights, *, published_is_parent):
    """维护点只用源标签训练两个共同工具，所有模型保存后供独立控制规则重载。"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    models = {}
    for name, source, raw in [('parent', Path(parent_weights), True),
                              ('published', Path(published_weights), published_is_parent)]:
        payload = source.read_bytes()
        (folder/(name+'.pkl')).write_bytes(payload)
        models[name] = {'weights_sha256': sha256(payload).hexdigest(), 'parent_input': raw,
                        'source_weights': str(source), 'fit_seconds': 0.}
    for name, include_parent in [('forest_parent', True), ('forest_raw', False)]:
        columns = list(train.columns) if include_parent else [c for c in train.columns if c != 'parent_value']
        if not include_parent and len(columns) == train.shape[1]:
            raise ValueError('公共森林需要命名父预测列才能做有/无父预测对照')
        model = make_pipeline(ColumnTransformer([('features', 'passthrough', columns)], remainder='drop'),
            RandomForestClassifier(n_estimators=96, max_depth=6, class_weight='balanced', n_jobs=1, random_state=17))
        started = time.monotonic()
        model.fit(train, target)
        seconds = time.monotonic()-started
        payload = cloudpickle.dumps(model)
        (folder/(name+'.pkl')).write_bytes(payload)
        models[name] = {'weights_sha256': sha256(payload).hexdigest(), 'parent_input': False,
                        'source_weights': None, 'fit_seconds': seconds}
    names = ['published_fixed', 'parent_fixed', 'constant_1', 'constant_2']
    definitions = [{'family': 'published', 'threshold': None}, {'family': 'parent', 'threshold': None},
                   {'constant': 1}, {'constant': 2}]
    for family in FAMILIES:
        for threshold in THRESHOLDS:
            names.append(f'{family}@{threshold:g}')
            definitions.append({'family': family, 'threshold': threshold})
    identity = {'models': models, 'arm_names': names, 'definitions': definitions, 'source_fit_rows': len(train),
                'new_fits': 2, 'uses_opponent_model': False, 'thresholds': THRESHOLDS}
    atomic_json(folder/'identity.json', identity)
    return identity


class BinaryPublicationBank:
    """逐行调用已重载的真实权重；未提供概率接口的发布版保留其硬分类行为。

    无概率接口时使用原类别的一位有效分数，阈值动作与硬预测一致，不能
    将该分数解释为校准概率。每个控制规则独立重载，避免模型内部状态串用。
    """
    def __init__(self, folder):
        folder = Path(folder)
        self.identity = json.loads((folder/'identity.json').read_bytes())
        self.models = {}
        for name, record in self.identity['models'].items():
            payload = (folder/(name+'.pkl')).read_bytes()
            if sha256(payload).hexdigest() != record['weights_sha256']:
                raise ValueError('发布工具权重发生变化')
            self.models[name] = cloudpickle.loads(payload)
        self.score_kind = {}
        for name, model in self.models.items():
            self.score_kind[name] = 'probability' if hasattr(model, 'predict_proba') and np.array_equal(getattr(model, 'classes_', []), [1, 2]) else 'one_hot_from_class'

    def predict_row(self, frame):
        if len(frame) != 1:
            raise ValueError('发布接口只接收当前一行')
        if frame.columns[-1] != 'parent_value':
            raise ValueError('源父模型要求最后一列为命名父预测，不能按位置误删原始特征')
        labels, probabilities = {}, {}
        for name, model in self.models.items():
            values = frame.iloc[:, :-1].to_numpy() if self.identity['models'][name]['parent_input'] else frame
            prediction = np.asarray(model.predict(values))
            if prediction.shape != (1,) or prediction[0] not in [1, 2]:
                raise ValueError('原模型没有交付合法二分类标签')
            labels[name] = int(prediction[0])
            if self.score_kind[name] == 'probability':
                probability = np.asarray(model.predict_proba(values))
                if probability.shape != (1, 2) or not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any() or not np.allclose(probability.sum(axis=1), 1., atol=1e-6):
                    raise ValueError('原模型概率输出非法')
                probabilities[name] = float(probability[0, 1])
            else:
                probabilities[name] = float(prediction[0] == 2)
        result = []
        for definition in self.identity['definitions']:
            if 'constant' in definition:
                result.append(definition['constant'])
            elif definition['threshold'] is None:
                result.append(labels[definition['family']])
            else:
                result.append(2 if probabilities[definition['family']] > definition['threshold'] else 1)
        return np.asarray(result, dtype=int), probabilities
