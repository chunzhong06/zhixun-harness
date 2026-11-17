"""在源设备整台留出上选择工况树或固定组合，封存选择后才对服务观测推理。"""
from collections import defaultdict
from hashlib import sha256

import numpy as np
from catboost import CatBoostRegressor
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from harness.core.contracts import digest
from harness.learning.models.condition_processing import ConditionStandardScaler
from harness.learning.models.model_artifacts import ModelArtifacts
from harness.learning.models.source_combinations import SourceCombinations
from harness.learning.models.training_tools import SklearnBackend


def fit_half(x,y):
    forest = RandomForestRegressor(n_estimators=100,max_depth=6,random_state=17,n_jobs=1).fit(x,y)
    linear = make_pipeline(StandardScaler(),Ridge(alpha=1.)).fit(x,y)
    return forest,linear


def row_groups(x,op_scale,cluster):
    return np.column_stack((x[:,:14],cluster.predict(op_scale.transform(x[:,14:]))))


def split_official(units):
    families = defaultdict(list)
    for u in units:
        families[u.split(":")[0]].append(u)
    folds = [[] for _ in range(5)]
    for group in families.values():
        ordered = sorted(group,key=lambda u:sha256(("source-choice/"+u).encode()).digest())
        for j,u in enumerate(ordered):
            folds[j%5].append(u)
    return folds


def source_choice(schema,rows,early,uniform,folds):
    if schema not in ("official","flight"):
        raise ValueError("需要已登记的发动机输入契约")
    records = []
    all_units = set(rows)
    for held in folds:
        train = sorted(all_units-set(held))
        x = np.vstack([rows[u][0] for u in train])
        y = np.concatenate([rows[u][1] for u in train])
        query = np.vstack([rows[u][0] for u in held])
        initial = np.vstack([early[u] for u in train])
        steady = np.vstack([uniform[u] for u in train])
        if schema=="official":
            query = SourceCombinations(x).transform(query)
            condition = ConditionStandardScaler((21,22,23),(0,2,0)).fit(initial)
            z,p = condition.transform(x),condition.transform(query)
        else:
            scaler = StandardScaler().fit(steady[:,14:])
            centers = KMeans(n_clusters=3,n_init=10,random_state=17).fit(scaler.transform(steady[:,14:]))
            condition = ConditionStandardScaler((14,),(0,)).fit(row_groups(initial,scaler,centers))
            def prepare(v):
                return np.column_stack((condition.transform(row_groups(v,scaler,centers))[:,:14],v[:,14:]))
            z,p = prepare(x),prepare(query)
        tree = CatBoostRegressor(iterations=100,depth=6,learning_rate=.05,l2_leaf_reg=1.,random_seed=17,
                                 thread_count=1,verbose=False,allow_writing_files=False).fit(z,y)
        rf,linear = fit_half(x,y)
        prediction = {"early":tree.predict(p,thread_count=1),"half":.5*(rf.predict(query)+linear.predict(query))}
        if schema=="official":
            prediction = {k:np.maximum(0,v) for k,v in prediction.items()}
        index = 0
        for u in held:
            truth = rows[u][1]
            stop = index+len(truth)
            records.append({"unit":u,"mse":{k:float(np.mean((v[index:stop]-truth)**2)) for k,v in prediction.items()},
                            "issued":{k:v[index:stop].tolist() for k,v in prediction.items()}})
            index=stop
    if len(records)!=len(all_units) or {r["unit"] for r in records}!=all_units:
        raise ValueError("源设备留出不完整")
    score = {k:float(np.sqrt(np.mean([r["mse"][k] for r in records]))) for k in ("half","early")}
    return {"folds":folds,"held":records,"score":score,"choice":min(score,key=score.get)}


class SourceSelectedEngine:
    """正式运行只读取源包；模型、早期参照、源侧取舍共同存入运行自己的模型仓库。"""
    def __init__(self,task,training,source,checksum,folder):
        x = np.asarray([r.values for r in training])
        y = np.asarray([r.label for r in training])
        self.repair = SourceCombinations(x)
        backend = SklearnBackend(execution_enabled=True)
        spec = {"policy":"source-tool-choice","package":checksum,"training":digest([r.sample_id for r in training]),
                "repair":self.repair.identity,"implementation": self.implementation_id(),"backend":backend.specification}

        def build():
            if task.target_unit!="cycles" or any(r.role!="fit" for r in source):
                raise ValueError("发动机工具选择需要源设备完整训练记录")
            groups,fit = defaultdict(list),defaultdict(list)
            for r in source:
                groups[r.unit_id].append(r)
            for r in training:
                fit[r.unit_id].append(r)
            if set(groups)!=set(fit) or any(len(rows)<8 for rows in groups.values()):
                raise ValueError("工况参照与训练设备不匹配")
            rows = {u:(np.array([r.values for r in v]),np.array([r.label for r in v])) for u,v in fit.items()}
            first = {u:np.array([r.values for r in sorted(v,key=lambda r:r.time_end)[:8]]) for u,v in groups.items()}
            choice = source_choice("official",rows,first,{u:v[0] for u,v in rows.items()},split_official(rows))
            if choice["choice"]=="early":
                scaler = ConditionStandardScaler((21,22,23),tuple(task.condition_decimals)).fit(np.vstack(list(first.values())))
                tree = CatBoostRegressor(iterations=100,depth=6,learning_rate=.05,l2_leaf_reg=1.,random_seed=17,
                                         thread_count=1,verbose=False,allow_writing_files=False).fit(scaler.transform(x),y)
                fitted = (scaler,tree)
            else:
                fitted = fit_half(x,y)
            return backend.dump({"decision":choice,"fitted":fitted})

        store = ModelArtifacts(folder)
        self.model_ref = store.create(spec,build)
        header,blob = store.read(self.model_ref)
        self.bundle = backend.load(blob)
        self.decision = self.bundle["decision"]
        self.identity = digest([spec,header["sha256"]])

    @staticmethod
    def implementation_id():
        from pathlib import Path
        return sha256(Path(__file__).read_bytes()).hexdigest()

    @staticmethod
    def predict(bundle,repair,x):
        mapped = repair.transform(x)
        left,right = bundle["fitted"]
        choice = bundle["decision"]["choice"]
        if choice=="early":
            return right.predict(left.transform(mapped),thread_count=1)
        if choice=="half":
            return .5*(left.predict(mapped)+right.predict(mapped))
        raise ValueError("未知的源设备选定模型")

    def advance_current(self,model,explore,confirm,serve):
        ids = [r["sample_id"] for r in serve]
        if not ids or len(set(ids))!=len(ids) or any(r["label"] is not None for r in serve):
            raise ValueError("服务清单含标签或重复测量")
        x = np.asarray([r["values"] for r in serve])
        pred = self.predict(self.bundle,self.repair,x)
        return (dict(zip(ids,pred.tolist(),strict=True)),dict(zip(ids,model.predict(x).tolist(),strict=True)),
                {"chosen":self.decision["choice"],"selection":"source-tool-choice","source_scores":self.decision["score"]})


