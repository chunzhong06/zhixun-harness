"""宿主明确提供的固定历史加当前合法训练数据，构成纯 TabICLv2 上下文。

可把本文件完整源码交给双方研究驱动。历史设备的去重、未来轨迹排除及
数据合法性由宿主准备；本模型只核验约定的CSV与元信息，不读取原件或答案。
"""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


class HistoricalTabularRegressor(RegressorMixin, BaseEstimator):
    """历史在前、当前数据在后，以固定八成员TFM重新建立完整上下文。

    history-metadata.json 必须含 history_sha256、history_rows、feature_columns。
    若含逐行 training_ids/training_cases/training_groups，也核对长度及ID唯一性。
    再次fit保持历史身份不变，只接收调用方已经合法扩充的当前X/y。
    """

    def __init__(self, runtime_assets_path='input/runtime-assets.json',
                 history_path='input/history.csv', history_metadata_path='input/history-metadata.json'):
        self.runtime_assets_path = runtime_assets_path
        self.history_path = history_path
        self.history_metadata_path = history_metadata_path

    def fit(self, X, y):
        import torch
        from tabicl import TabICLRegressor

        columns = [name for name in X.columns if name != 'parent_value']
        if not columns or len(set(columns)) != len(columns):
            raise ValueError('当前训练必须提供非空且不重复的特征列')
        current = X.loc[:, columns]
        labels = np.asarray(y, dtype=np.float64)
        if labels.shape != (len(current),) or not len(current) or not np.isfinite(labels).all():
            raise ValueError('当前标签须为与X逐行对应的有限一维数组')
        metadata_bytes = Path(self.history_metadata_path).read_bytes()
        metadata = json.loads(metadata_bytes)
        history_bytes = Path(self.history_path).read_bytes()
        history_sha = sha256(history_bytes).hexdigest()
        if history_sha != metadata['history_sha256']:
            raise ValueError('历史CSV身份与宿主元信息不同')
        if getattr(self, 'history_sha256_', history_sha) != history_sha:
            raise ValueError('同一模型再次拟合不能更换已固定的历史池')
        history = pd.read_csv(BytesIO(history_bytes))
        if metadata['feature_columns'] != columns or list(history) != columns+['target']:
            raise ValueError('历史与当前物理列或顺序不同')
        if len(history) != metadata['history_rows'] or not len(history):
            raise ValueError('历史行数与宿主元信息不同')
        for key in ('training_ids', 'training_cases', 'training_groups'):
            if key in metadata and len(metadata[key]) != len(history):
                raise ValueError('历史逐行元信息长度不同：'+key)
        if 'training_ids' in metadata and len(set(metadata['training_ids'])) != len(history):
            raise ValueError('历史全局训练ID重复')
        history_y = history['target'].to_numpy(dtype=np.float64)
        if (not np.isfinite(history.to_numpy(dtype=np.float64)).all()
                or not np.isfinite(current.to_numpy(dtype=np.float64)).all()):
            raise ValueError('本入口要求宿主完成源端缺失处理并提供有限数值')
        combined = pd.concat([history.loc[:, columns], current], ignore_index=True)
        combined_y = np.concatenate([history_y, labels])

        assets = json.loads(Path(self.runtime_assets_path).read_bytes())
        checkpoints = [(Path(path), expected) for path, expected in assets['read_only_files'].items()
                       if Path(path).name == 'tabicl-regressor-v2-20260212.ckpt']
        if len(checkpoints) != 1 or not checkpoints[0][0].is_absolute():
            raise ValueError('只读清单须提供唯一的回归checkpoint绝对路径')
        checkpoint, expected = checkpoints[0]
        actual = sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''):
                actual.update(block)
        if actual.hexdigest() != expected:
            raise ValueError('只读checkpoint身份与清单不同')
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            model = TabICLRegressor(model_path=str(checkpoint), n_estimators=8, batch_size=1,
                kv_cache=True, device='cpu', n_jobs=1, allow_auto_download=False,
                random_state=42, use_amp=False).fit(combined, combined_y)
        # 所有拟合完成后再替换状态；发布阶段调用同一fit仅扩充current部分。
        self.model_, self.columns_ = model, columns
        self.history_sha256_ = history_sha
        self.history_metadata_sha256_ = sha256(metadata_bytes).hexdigest()
        self.history_rows_, self.current_rows_ = len(history), len(current)
        self.training_rows_ = len(combined)
        self.checkpoint_sha256_ = expected
        return self

    def __getstate__(self):
        # 每次保存都重设官方选项，使已重载对象再次保存仍携带权重、训练上下文和KV。
        if hasattr(self, 'model_'):
            self.model_._save_model_weights = True
            self.model_._save_training_data = True
            self.model_._save_kv_cache = True
        return self.__dict__.copy()

    def predict(self, X):
        check_is_fitted(self, ['columns_', 'model_', 'history_sha256_'])
        with threadpool_limits(limits=1):
            prediction = np.asarray(self.model_.predict(X.loc[:, self.columns_]), dtype=np.float64)
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError('模型须对每行返回一个有限预测')
        return np.maximum(0., prediction)
