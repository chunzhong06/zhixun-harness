"""训练方案按未参与对应拟合的整组机队实测来选择，评价与最后发布分开。"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def train_recipe(name, x, y):
    learner = name.split("/")[-1]
    if learner == "森林":
        return RandomForestRegressor(n_estimators=100, max_depth=6, random_state=17, n_jobs=1).fit(x, y)
    if learner == "岭":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.)).fit(x, y)
    raise ValueError("未知训练方法")


def fleet_trials(cohorts, fold_features, *, collect_predictions=False):
    """每个源机队含多台设备；先在每台内求均方误差，再在机队内等权汇总。"""
    names = tuple(cohorts)
    if len(names) < 2:
        raise ValueError("需要至少两组可整组留出的历史设备")
    views = tuple(cohorts[names[0]][0])
    tools, trace, held_predictions = {}, {}, {}
    for view in views:
        for learner in ("森林", "岭"):
            key = view + "/" + learner
            folds = {}
            for held in names:
                others = [n for n in names if n != held]
                matrices = fold_features[held]
                model = train_recipe(key, np.vstack([matrices[n][view] for n in others]),
                                     np.concatenate([cohorts[n][1] for n in others]))
                _, y, units = cohorts[held]
                prediction = model.predict(matrices[held][view])
                if collect_predictions:
                    held_predictions.setdefault(held,{})[key] = prediction
                err = (prediction - y)**2
                folds[held] = float(np.sqrt(np.mean([np.mean(err[units == u]) for u in np.unique(units)])))
            trace[key] = folds
            tools[key] = train_recipe(key, np.vstack([cohorts[n][0][view] for n in names]),
                                      np.concatenate([cohorts[n][1] for n in names]))
    selection = min(trace, key=lambda k: (max(trace[k].values()), np.mean(list(trace[k].values()))))
    if collect_predictions:
        return selection, tools, trace, held_predictions
    return selection, tools, trace
