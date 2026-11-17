"""HGB 与 TabICLv2 同步拟合的固定等权参考，可整份源码公开给研究驱动。

fit 只使用调用方传入的 X/y，两端都排除 parent_value；合法回标后再次
fit 会同时重建两个模型。运行环境须预先提供 tabicl 及只读 checkpoint，
本文件不安装依赖、不下载资产，也不读取训练标签以外的评价资料。
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


class SynchronizedTabularRegressor(RegressorMixin, BaseEstimator):
    """以同一份 DataFrame 和一维标签同步重训，两端非负预测各占一半。

    runtime_assets_path 只定位宿主声明的只读权重。保存后的对象携带
    TabICL 实际权重与上下文缓存，cloudpickle 重载预测不再读取该清单。
    """

    def __init__(self, runtime_assets_path='input/runtime-assets.json'):
        self.runtime_assets_path = runtime_assets_path

    def fit(self, X, y):
        import torch
        from tabicl import TabICLRegressor

        columns = [name for name in X.columns if name != 'parent_value']
        if not columns or len(set(columns)) != len(columns):
            raise ValueError('训练特征为空或列名重复')
        features = X.loc[:, columns]
        labels = np.asarray(y, dtype=np.float64)
        if labels.shape != (len(features),) or not np.isfinite(labels).all():
            raise ValueError('训练标签须为与X逐行对应的有限一维数组')
        assets = json.loads(Path(self.runtime_assets_path).read_text(encoding='utf-8'))
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
        # 先完成两端拟合再替换对象状态，避免一次失败留下不同训练代际的组合。
        with threadpool_limits(limits=1):
            tree = HistGradientBoostingRegressor(max_iter=96, max_depth=6, max_leaf_nodes=15,
                l2_regularization=1., random_state=17).fit(features, labels)
            tabicl = TabICLRegressor(model_path=str(checkpoint), n_estimators=8, batch_size=1,
                kv_cache=True, device='cpu', n_jobs=1, allow_auto_download=False,
                random_state=42, use_amp=False).fit(features, labels)
        # TabICL 2.2.0 的原序列化接口读取这些选项；包含权重后无需依赖原文件重载。
        tabicl._save_model_weights = True
        tabicl._save_training_data = True
        tabicl._save_kv_cache = True
        self.columns_ = columns
        self.tree_, self.tabicl_ = tree, tabicl
        self.checkpoint_sha256_ = expected
        return self

    def predict(self, X):
        check_is_fitted(self, ['columns_', 'tree_', 'tabicl_'])
        features = X.loc[:, self.columns_]
        with threadpool_limits(limits=1):
            tree = np.maximum(0., np.asarray(self.tree_.predict(features), dtype=np.float64))
            tabicl = np.maximum(0., np.asarray(self.tabicl_.predict(features), dtype=np.float64))
        if tree.shape != (len(features),) or tabicl.shape != tree.shape:
            raise ValueError('组件预测形状不符')
        if not np.isfinite(tree).all() or not np.isfinite(tabicl).all():
            raise ValueError('组件预测存在非有限值')
        return (tree+tabicl)/2
