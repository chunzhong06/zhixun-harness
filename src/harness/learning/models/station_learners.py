"""站点差值学习器的公开构造接口，超参由现场提案给出并计入模型身份。"""
import math

from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


CLASSES={"Ridge":Ridge,"HuberRegressor":HuberRegressor,"KNeighborsRegressor":KNeighborsRegressor,
         "ExtraTreesRegressor":ExtraTreesRegressor,"RandomForestRegressor":RandomForestRegressor,
         "HistGradientBoostingRegressor":HistGradientBoostingRegressor,"SVR":SVR}
DEFAULTS={
    "Ridge":{"alpha":8.},
    "HuberRegressor":{"max_iter":200},
    "KNeighborsRegressor":{"n_neighbors":16,"weights":"distance","n_jobs":1},
    "ExtraTreesRegressor":{"n_estimators":64,"min_samples_leaf":4,"n_jobs":1,"random_state":17},
    "RandomForestRegressor":{"n_estimators":64,"min_samples_leaf":4,"max_depth":8,"n_jobs":1,"random_state":17},
    "HistGradientBoostingRegressor":{"max_iter":90,"max_leaf_nodes":9,"l2_regularization":1.,"random_state":17},
    "SVR":{"C":10.,"epsilon":.1},
}


def make(name,parameters=()):
    if name not in CLASSES:
        raise ValueError("未知的学习器")
    parameters=dict(parameters)
    allowed=CLASSES[name]().get_params(deep=False)
    if any(k not in allowed for k in parameters):
        raise ValueError("该学习器不存在提案中的参数")
    for value in parameters.values():
        if not isinstance(value,(str,int,float,bool,type(None))):
            raise ValueError("模型参数只接收标量")
        if isinstance(value,(int,float)) and not math.isfinite(value):
            raise ValueError("模型参数需要有限值")
    args={**DEFAULTS[name],**parameters}
    if args.get("n_jobs",1)!=1 or args.get("n_estimators",1)>128 or args.get("max_iter",1)>256:
        raise ValueError("本实验单线程，树数量至多128，迭代次数至多256")
    return CLASSES[name](**args)


def references():
    for alpha in (.1,100.):
        yield "Ridge",(("alpha",alpha),)
    for name in CLASSES:
        if name!="Ridge":
            yield name,()
