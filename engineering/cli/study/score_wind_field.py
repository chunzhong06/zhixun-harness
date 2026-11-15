"""先从已冻结的源机训练，再逐六小时交付开发风机的功率预测并回收限量标签。"""
import json
from hashlib import file_digest
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/"engineering/src"),str(ROOT/"engineering/cli/data")]
from prepare_wind_field import prepare, DATA
from harness.contracts import digest
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore,atomic_json
from harness.models.wind_power import WindRelease,WindMaintenance,PRIMARY

DEST = ROOT/"results/experiments/local-defaults/whole-maintenance/wind-field"
FIELDS = ["wind_speed","rotor_speed","ambient_temp"]
START = int(pd.Timestamp("2016-09-01",tz="UTC").timestamp())


def stream(site, clock, raw, truth, model, receiver, state=None):
    if (not len(clock) or np.any(np.diff(clock)<=0) or not np.isfinite(raw).all() or not np.isfinite(truth).all()
            or len(np.unique(clock))!=len(raw)):
        raise ValueError("开发流的输入身份、标签或时间轴异常")
    # 源模型只做逐行推理，批量计算不利用后来的输入。维护流程只读已发批次。
    bases = model.predict(raw)
    state = WindMaintenance() if state is None else state
    bounds = np.r_[0,np.flatnonzero(np.diff(clock//21600))+1,len(clock)]
    keys = []
    count = 0
    for a,b in zip(bounds[:-1],bounds[1:],strict=True):
        if hasattr(state,"predict_timed"):
            p = state.predict_timed(clock[a:b],raw[a:b],bases[a:b])
        else:
            p = state.predict(raw[a:b], bases[a:b])
        packet = {"设备":site,"时刻":clock[a:b].tolist(),"已回标":count,"预测":{n:v.tolist() for n,v in p.items()}}
        key = digest(packet)
        if not receiver.submit(key,packet)["accepted"]:
            raise ValueError("发放失败")
        keys.append(key)
        if b<len(clock) and (clock[b-1]+600)%21600==0:
            state.update(float(truth[b-1]))
            count += 1
    errors = {}
    cursor = 0
    for key in keys:
        p = json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
        end = cursor+len(p["时刻"])
        if digest(p)!=key or p["时刻"]!=clock[cursor:end].tolist():
            raise ValueError("实际发放和评分时刻不符")
        for n,vals in p["预测"].items():
            errors[n] = errors.get(n,0.) + float(np.sum((np.asarray(vals)-truth[cursor:end])**2))
        cursor = end
    if cursor!=len(clock):
        raise ValueError("存在未发放的计分时刻")
    return {"scores":{n:float(np.sqrt(v/cursor)) for n,v in errors.items()},"回标":count,"实发":cursor,"回执":keys}


def run(stage=DEST, train_fn=None, controller=WindMaintenance, primary=PRIMARY, previous=None):
    prepare()
    frame = pd.read_csv(DATA/"kelmarsh.csv.gz",usecols=["turbine_id","timestamp","power_pu",*FIELDS])
    frame["clock"] = pd.to_datetime(frame["timestamp"],utc=True).to_numpy(dtype="datetime64[s]").astype("int64")
    train = frame[frame.turbine_id.isin([f"Kelmarsh_{i}" for i in (1,2,3)]) & (frame.clock<START)]
    if train.empty or not np.isfinite(train[[*FIELDS,"power_pu"]]).all().all():
        raise ValueError("源训练范围不完整")
    model = WindRelease().fit(train[FIELDS].to_numpy(),train.power_pu.to_numpy()) if train_fn is None else train_fn(train)
    path = stage/"kelmarsh.joblib"
    joblib.dump(model,path,compress=3)
    with path.open("rb") as f:
        identity = file_digest(f,"sha256").hexdigest()
    model = joblib.load(path)
    receiver = LocalAcceptance()
    receiver.output = stage/"accepted"
    by_site = {}
    for unit in (4,5,6):
        site = f"Kelmarsh_{unit}"
        device = frame[(frame.turbine_id==site) & (frame.clock>=START)].sort_values("clock")
        state = model.controller() if controller is None else controller()
        by_site[site] = stream(site,device.clock.to_numpy(),device[FIELDS].to_numpy(),device.power_pu.to_numpy(),model,receiver,state)
        if previous:
            earlier = json.loads((previous/"scores.json").read_bytes())
            old = earlier["sites"][site]
            if by_site[site]["实发"]!=old["实发"] or by_site[site]["回标"]!=old["回标"]:
                raise ValueError("开发有效时刻或已返功率数变化")
            for a,b in zip(old["回执"],by_site[site]["回执"],strict=True):
                p = json.loads((previous/"accepted"/(a+".json")).read_bytes())["payload"]
                now = json.loads((receiver.output/(b+".json")).read_bytes())["payload"]
                if digest(p)!=a or digest(now)!=b or p["时刻"]!=now["时刻"] or p["已回标"]!=now["已回标"]:
                    raise ValueError("先前发布的输入身份发生变化")
                for role in ("现场","固定","论文"):
                    n = earlier["decision"][role]
                    if not np.allclose(p["预测"][n],now["预测"][n],rtol=0,atol=1e-7):
                        raise ValueError("旧对照未逐条复现")
    means = {n:float(np.mean([v["scores"][n] for v in by_site.values()])) for n in by_site[site]["scores"]}
    fixed = min((n for n in means if n not in (PRIMARY,primary)), key=means.get)
    paper = min((n for n in means if "论文" in n),key=means.get)
    worst = max(v["scores"][primary] for v in by_site.values())
    worst_fixed = max(v["scores"][fixed] for v in by_site.values())
    gain = 1-means[primary]/min(means[fixed],means[paper])
    decision = {"现场":primary,"固定":fixed,"论文":paper,"现场误差":means[primary],"固定误差":means[fixed],
                "论文误差":means[paper],"降幅":gain,"现场最差":worst,"固定最差":worst_fixed,
                "进入独立":gain>=.02 and worst<=1.01*worst_fixed}
    atomic_json(stage/"scores.json",{"decision":decision,"means":means,"sites":by_site,"权重SHA256":identity,"源样本":len(train)})
    print(json.dumps({"盲机正常功率开发":decision,"源训练":len(train),"逐机":{n:{"回标":v["回标"],"实发":v["实发"]} for n,v in by_site.items()}},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("风机开发流已评分")
        run()
