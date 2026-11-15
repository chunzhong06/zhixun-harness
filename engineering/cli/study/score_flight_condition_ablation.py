"""在完全相同的六处近邻测量上分开训练原读数和斜率换算，核算每项的增量。"""
import json
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from score_flight_condition_response import (BASE, DEST as PRIOR, PREP, SITES, acquire, reference_points,
    ConditionReadout, fixed_mix, receipt, issue, LocalStore, atomic_json, file_hash)

DEST = BASE / "flight-condition-ablation"
PRIMARY = "依工况估计是否值得换算"
PARTS = {"邻域原读数": np.r_[14:46], "仅换算读数": np.r_[0:14, 28:46]}
BASELINE = "工况分歧固定公式"
PAPER = "整趟序列"


def combine(values, w):
    return (values * w).sum(1) / w.sum(1)


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("原与换算对照已执行")
        prep = joblib.load(PREP).prep
        pts = reference_points(prep)
        train = [acquire(s,"dev",prep,pts) for s in SITES]
        features = np.concatenate([v[1] for v in train])
        life = np.concatenate([v[3] for v in train])
        models = {k:ConditionReadout().fit(features[:,:,idx],life) for k,idx in PARTS.items()}
        weights = DEST / "ablation.joblib"
        joblib.dump({"models":models,"points":pts,"prep":prep},weights)
        prior = json.loads((PRIOR / "scores.json").read_bytes())
        full = PRIOR / "condition.joblib"
        if file_hash(full) != prior["权重SHA256"]:
            raise ValueError("完整工具身份改变")
        joint = joblib.load(full)["readout"]
        records = {}
        for s,item in prior["sites"].items():
            ids,x,dist,truth = acquire(s,"test",prep,pts)
            old = receipt(PRIOR,s,item)
            if old["ids"] != [list(k) for k in ids]:
                raise ValueError("消融使用的航班不同")
            pred = dict(old["predictions"])
            readings, sigmas = [], []
            for name, idx in PARTS.items():
                val, sigma = models[name].readings(x[:,:,idx])
                readings.append(val)
                sigmas.append(sigma)
                pred[name+"等权"] = val.mean(1)
                pred[name+"分歧加权"] = combine(val,(1+sigma)**-2)
            both, spread = joint.readings(x)
            np.testing.assert_allclose(combine(both,(1+spread)**-2),pred[BASELINE],atol=1e-8)
            # 确保每种读法对相同的六个工况给出结果。局部规则在相应工况直接比较两个模型的误差。
            vals = np.stack(readings,axis=2)
            sig = np.stack(sigmas,axis=2)
            all_sigma = (1+sig) ** -2
            pred["两读法十二票固定分歧"] = combine(vals.reshape(len(ids),-1),all_sigma.reshape(len(ids),-1))
            pred["邻近信换算远处信原量"] = combine((vals[:,:,1] + dist*vals[:,:,0])/(1+dist), (1+spread)**-2)
            mix = np.column_stack([pred[k] for k in ("邻域原读数分歧加权","仅换算读数分歧加权",BASELINE,PAPER)])
            c = np.stack((dist,sig[:,:,0],sig[:,:,1],vals[:,:,0],vals[:,:,1],abs(vals[:,:,0]-vals[:,:,1])),axis=2)
            records[s] = (ids,truth,pred,mix,c,vals,spread)
        scores = {}
        for s,(ids,truth,pred,mix,c,vals,spread) in records.items():
            source = {k:v for k,v in records.items() if k!=s}
            y = np.concatenate([v[1] for v in source.values()])
            m = np.vstack([v[3] for v in source.values()])
            tags = np.concatenate([np.repeat(k,len(v[1])) for k,v in source.items()])
            fixed = fixed_mix(m,y,tags)
            pred["工况消融与论文固定组合"] = mix @ fixed
            X = np.vstack([v[4].reshape(-1,6) for v in source.values()])
            # 所有六点共享同一航班的寿命，机队与航班权重不因点数而改变。
            err = np.vstack([abs(v[5]-v[1][:,None,None]).reshape(-1,2) for v in source.values()])
            sw = np.repeat([len(y)/(len(source)*len(source[k][1])) for k in tags],6)
            gate = ExtraTreesRegressor(n_estimators=80,max_depth=4,min_samples_leaf=40,random_state=17,n_jobs=1)
            gate.fit(X,err,sample_weight=sw)
            def parts(context):
                estimate = gate.predict(context).clip(1) ** -2
                return estimate / estimate.sum(1)[:,None]
            w = parts(c.reshape(-1,6)).reshape(len(c),6,2)
            p = combine((vals*w).sum(2),(1+spread)**-2)
            pred[PRIMARY] = (p + pred[BASELINE])/2
            constant = np.average(parts(X),axis=0,weights=sw)
            pred["消融分量冻结为均值"] = (combine(vals@constant,(1+spread)**-2)+pred[BASELINE])/2
            scores[s] = issue(DEST,s,ids,pred,truth)
        means = {k:float(np.mean([v["scores"][k]["rmse"] for v in scores.values()])) for k in pred}
        controls = [prior["decision"]["固定"],"邻域原读数分歧加权","仅换算读数分歧加权","邻域原读数等权","仅换算读数等权",
                    "两读法十二票固定分歧","邻近信换算远处信原量","工况消融与论文固定组合","消融分量冻结为均值"]
        fixed = min(controls,key=means.get)
        paper = min((prior["decision"]["同工具论文"],"工况消融与论文固定组合",PAPER),key=means.get)
        gain = 1 - means[PRIMARY]/min(means[fixed],means[paper])
        decision = {"主候选": PRIMARY, "固定":fixed,"同工具论文":paper,"降幅":gain,"进独立":gain>=.01}
        atomic_json(DEST/"scores.json", {"sites":scores,"means":means,"decision":decision,"模型SHA256":file_hash(weights)})
        print(json.dumps({"比较":{k:means[k] for k in [PRIMARY,*controls]}, "判定":decision},ensure_ascii=False))


if __name__=="__main__":
    run()
