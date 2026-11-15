"""同组 707 个终点一次评价已交付的末版；先封存预测、权重和输入身份，再解锁答案。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import sys

from engine_endpoint_check import ROOT, RAW, PREFLIGHT, OUTPUT, ModelArtifacts, LocalStore, atomic_json, inspect_inputs, np, test_answers

STUDY = ROOT/"results/experiments/production-selection"
DEST = STUDY/"engine-transfer"
ARMS = ("original", "fixed", "project", "simple", "mature")


def read(path):
    return json.loads(path.read_bytes())


def evaluate():
    receipt, tracks, anchors, archive = inspect_inputs()
    if receipt != read(PREFLIGHT):
        raise ValueError("与既有 707 台终点的冻结输入不符")
    units = [t for group in tracks.values() for t in group]
    if len(units)!=707 or len({t.unit_id for t in units})!=707:
        raise ValueError("终点列表不完整")
    x = np.array([(*t.values[-1][3:], *t.values[-1][:3]) for t in units])
    weights = {"balanced":anchors["balanced"]}
    for arm in ARMS:
        folder = STUDY/"engine"/arm
        rounds = read(folder/"run.json")["rounds"]
        if len(rounds)!=10:
            raise ValueError("缺少十轮交接")
        last = rounds[-1]
        if read(folder/last["submitted"])["payload"]["model_id"] != last["selected"]:
            raise ValueError("末版与真实交接不符")
        _, weights[arm] = ModelArtifacts(folder/"models").read(last["model_artifact"])
    if weights["original"]!=anchors["original"] or weights["fixed"]!=weights["project"]:
        raise ValueError("起点或已确认的相同末版发生变化")
    source = {"preflight":sha256(PREFLIGHT.read_bytes()).hexdigest(),
              "weights":{arm:sha256(blob).hexdigest() for arm,blob in weights.items()}}
    sealed = DEST/"issued.json"
    if sealed.is_file():
        saved = read(sealed)
        if saved["source"]!=source:
            raise ValueError("已交预测与本次权重或输入身份不同")
        predictions = saved["predictions"]
    else:
        sys.path.insert(0,str(archive))
        from baselines.model import Model
        predictions = {}
        with TemporaryDirectory(prefix="frozen-engine-") as tmp:
            for arm,blob in weights.items():
                path = Path(tmp)/(arm+".npz")
                path.write_bytes(blob)
                p = Model.load(path).predict(x)
                if p.shape!=(len(units),) or not np.isfinite(p).all():
                    raise ValueError("末版预测无效")
                predictions[arm] = {t.unit_id:float(v) for t,v in zip(units,p,strict=True)}
        atomic_json(sealed,{"source":source,"predictions":predictions})
    old = read(OUTPUT/"issued.json")["predictions"]
    if any(set(predictions[arm])!=set(old[arm]) or
           any(abs(v-old[arm][i])>1e-8 for i,v in predictions[arm].items()) for arm in ("original","balanced")):
        raise ValueError("未复现以前封存的两组参照")
    truth = {s:test_answers(RAW,s,group) for s,group in tracks.items()}
    truth["pooled"] = {i:v for subset in truth.values() for i,v in subset.items()}
    if set(predictions)!=set(weights) or any(set(p)!=set(truth["pooled"]) for p in predictions.values()):
        raise ValueError("预测与答案标识不齐")
    scores = {arm:{s:float(np.sqrt(np.mean([(p[i]-v)**2 for i,v in answers.items()])))
                   for s,answers in truth.items()} for arm,p in predictions.items()}
    fixed, simple = scores["fixed"], scores["simple"]
    cleared = (simple["pooled"]<=0.99*fixed["pooled"] and simple["pooled"]<scores["balanced"]["pooled"]
               and all(simple[s]<=fixed[s] for s in tracks))
    atomic_json(DEST/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),
                "devices":{s:len(a) for s,a in truth.items()},"rmse":scores,"simple_cleared":cleared})
    print({a:round(v["pooled"],3) for a,v in scores.items()},cleared)


if __name__=="__main__":
    with LocalStore(DEST).exclusive():
        evaluate()
