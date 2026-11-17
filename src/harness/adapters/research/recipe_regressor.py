"""把现有可执行训练配方转成可检查、可编辑的源码，保持训练时序与残差语义。"""
from dataclasses import asdict

import numpy as np

from harness.adapters.research.generated_regressor import inspect_source
from harness.learning.models.station_learners import make


def code_inputs(tools):
    """最后一列为连续小时坐标，用于精确选历史，缺标签不能压缩经过的时间。"""
    return np.column_stack((tools.x,np.arange(len(tools.x),dtype=float))),[*tools.names,"elapsed_hour"]


def source_for_rule(tools,rule):
    """导出现有工具的同一算法；此源码是已有能力，不计作现场驱动的新发现。"""
    tools.parse({"hypothesis":"导出已存在的训练配方","recipe":asdict(rule)})
    background=tools.names.index(rule.background)
    columns=[] if rule.correction=="bias" else [background]
    if rule.correction in ("weather","network"):columns.extend(tools.names.index(n) for n in tools.weather)
    if rule.correction=="network":columns=list(dict.fromkeys(columns+[tools.names.index(n) for n in tools.backgrounds[:-2]]))
    columns=list(dict.fromkeys(columns+[tools.names.index(n) for n in rule.features]))
    if rule.where=="all":selection=""
    else:
        field,side=rule.where.split(":");operator="<=" if side=="low" else ">"
        selection=f"        selected = selected & (X[:, {tools.names.index(field)}] {operator} {rule.limit!r})\n"
    estimator=make(rule.learner,rule.parameters)
    # 导入公开模块名，避免生成源码依赖sklearn内部模块路径。
    modules={"Ridge":"sklearn.linear_model","HuberRegressor":"sklearn.linear_model",
             "KNeighborsRegressor":"sklearn.neighbors","SVR":"sklearn.svm",
             "ExtraTreesRegressor":"sklearn.ensemble","RandomForestRegressor":"sklearn.ensemble",
             "HistGradientBoostingRegressor":"sklearn.ensemble"}
    module=modules[rule.learner];parameters=estimator.get_params(deep=False)
    model_fit=(f"        self.model_ = {rule.learner}(**{parameters!r})\n"
               "        self.model_.fit((X[:, self.columns_] - self.center_) / self.scale_, residual)\n") if columns else ""
    prediction=("self.model_.predict((X[:, self.columns_] - self.center_) / self.scale_)") if columns else "np.ones(X.shape[0]) * self.bias_"
    source=f'''import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from {module} import {rule.learner}

class ReferenceRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # 时间列表示实际经过的小时；按最后一条合法训练参考截取历史。
        selected = X[:, -1] >= np.max(X[:, -1]) - 24 * {rule.days!r} + 1
{selection}        X, y = X[selected], y[selected]
        if len(y) < 12:
            raise ValueError("筛选后训练参考少于12条")
        self.training_hours_ = X[:, -1].copy()
        self.columns_ = {columns!r}
        residual = y - X[:, {background}]
        self.center_ = np.mean(X[:, self.columns_], axis=0)
        self.scale_ = np.std(X[:, self.columns_], axis=0)
        self.scale_ = np.where(self.scale_ < 1e-8, 1., self.scale_)
        self.bias_ = float(np.mean(residual))
{model_fit}        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        residual = {prediction}
        return np.maximum(0., X[:, {background}] + {rule.strength!r} * residual)

def build_model():
    return ReferenceRegressor()
'''
    inspect_source(source)
    return source
