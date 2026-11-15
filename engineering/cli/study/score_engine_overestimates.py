"""有其他设备的寿命回标后，检验原模型高于固定树时是否应下调本次交付。"""
from collections import defaultdict
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from score_engine_full_condition_forward import STUDY, ROOT, released, read, digest, _trainer
from score_completed_residual_transfer import engines, official, summary
from harness.models.early_engine_reference import EarlyEngineReference
from harness.models.full_condition_engine import FullConditionEngine
from harness.models.training_tools import SklearnBackend
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

DEST = STUDY.parent/"engine-overestimates"
BASE = STUDY.parent/"engine-full-condition-integrated/engine"


@contextmanager
def predictors():
    report,repair,blob = released(BASE,"native_artifact")
    _,old_repair,old_blob = released(STUDY.parent/"engine-nonnegative-integrated/engine","tree_artifact")
    small = SklearnBackend(execution_enabled=True).load(old_blob)
    archive, = (ROOT/"data/official/engine").glob("*.zip")
    if sha256(archive.read_bytes()).hexdigest()!=read(STUDY/"training.json")["source"] or repair.identity!=old_repair.identity:
        raise ValueError("工况输入修复与封存实验不同")
    with TemporaryDirectory(dir=DEST) as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"C") as trainer:
        cond,center,model,weight = FullConditionEngine.load_bundle(blob,trainer)
        if weight!=report["native_sha256"]:
            raise ValueError("续训权重与发放版不同")
        def predict(x):
            x = np.asarray(x)
            p = np.maximum(0,model.predict(FullConditionEngine.inputs(cond,center,model,repair.transform(x))))
            tree = np.maximum(0,EarlyEngineReference.predict(small,repair,x))
            return p,tree
        yield predict


def public_stream():
    _,_,batches = official("engine")
    for b in batches:
        roles = {}
        for role in ("explore","adopt","serve"):
            rows = [r for r in b.samples if r.role==role and r.available_at==b.ordinal]
            if role=="serve":
                if any(r.label is not None for r in rows):
                    raise ValueError("公开服务样本含答案")
                roles[role] = [(r.sample_id,r.unit_id,r.values) for r in rows]
            else:
                roles[role] = [(r.unit_id,r.sample_id,r.values,r.label) for r in rows]
        yield b.ordinal,roles["explore"],roles["adopt"],roles["serve"]


def failure_stream(data):
    ends,history,service,_,_ = data
    groups,rounds = defaultdict(list),defaultdict(list)
    for row in history:
        groups[row[0]].append(row)
    for key,u,clock,x in service:
        rounds[clock].append((key,u,x))
    ordered = sorted(ends,key=lambda u:(ends[u],u))
    pos = 0
    for clock,target in sorted(rounds.items()):
        fit,check = [],[]
        while pos<len(ordered) and ends[ordered[pos]]<clock:
            device = ordered[pos]
            (fit if pos%2==0 else check).extend(groups[device])
            pos+=1
        yield clock,fit,check,target


def update(rows,bank,predict):
    if not rows:
        return
    p,t = predict([r[2] for r in rows])
    for (u,k,_,y),value,tree in zip(rows,p,t,strict=True):
        if y is None or k in bank[u]:
            raise ValueError("可用回标缺答案或重复")
        gap = max(0.,value-tree)
        bank[u][k] = (float(gap*(value-y)),float(gap*gap))


def moments(bank):
    if not bank:
        return 0.,0.
    # 每台机器只占一份权重，防止长轨迹替代了多台设备的独立回标。
    return np.mean([np.mean(list(rows.values()),axis=0) for rows in bank.values()],axis=0)


def run(site,stream,truth,predict,expected):
    fit,confirm = defaultdict(dict),defaultdict(dict)
    receiver = LocalAcceptance()
    receiver.output = DEST/site/"accepted"
    owners,requests,events = {},[],[]
    weight = proposed = 0.
    for clock,learn,check,target in stream:
        update(learn,fit,predict)
        update(check,confirm,predict)
        if learn or check:
            num,den = moments(fit)
            proposed = float(np.clip(num/den,0,1)) if den>0 else 0.
            cv_num,cv_den = moments(confirm)
            weight = proposed if proposed*proposed*cv_den-2*proposed*cv_num<0 else 0.
        keys = [r[0] for r in target]
        if len(set(keys))!=len(keys) or owners.keys() & set(keys):
            raise ValueError("目标行重复")
        owners.update((k,u) for k,u,x in target)
        p,t = predict([x for k,u,x in target])
        np.testing.assert_allclose(p,[expected[k] for k in keys],rtol=0,atol=1e-8)
        gap = np.maximum(0,p-t)
        arms = {"native":p,"small-tree":t,"midpoint":.5*(p+t),"fixed-half":p-.5*gap,"fixed-all":p-gap,
                "fit":p-proposed*gap,"confirmed":p-weight*gap}
        event = {"clock":clock,"fit":[r[1] for r in learn],"check":[r[1] for r in check],
                 "proposed":proposed,"weight":weight}
        payload = {"event":event,"ids":keys,"predictions":{k:v.tolist() for k,v in arms.items()}}
        req = digest([site,payload])
        ack = receiver.submit(req,payload)
        if ack.get("request_id")!=req or not ack.get("accepted"):
            raise RuntimeError("新预测未交接")
        requests.append(req)
        events.append(event)
    if set(fit)&set(confirm) or set(owners.values())&(set(fit)|set(confirm)) or set(expected)!=set(owners):
        raise ValueError("学习、确认、服务设备交叉或覆盖不完整")
    issued = defaultdict(dict)
    for req in requests:
        payload = read(receiver.output/(req+".json"))["payload"]
        if digest([site,payload])!=req:
            raise ValueError("已封存预测身份改变")
        for arm,v in payload["predictions"].items():
            issued[arm].update(zip(payload["ids"],v,strict=True))
    y = truth()
    metrics = summary("engine",y,{"parent":issued["native"],**issued},owners)
    del metrics["parent"]
    keys = list(owners)
    ids = [k for k in keys if issued["confirmed"][k]!=issued["native"][k]]
    changed = {"total":len(ids),"improved":sum(abs(issued["confirmed"][k]-y[k])<abs(issued["native"][k]-y[k]) for k in ids)}
    # 两个可行动作的事后上限单列，只用来决定是否继续研究这条分歧。
    oracle = {k:min((issued["fixed-all"][k],issued["native"][k]),key=lambda v:abs(v-y[k])) for k in keys}
    ideal = float(np.sqrt(np.mean([(oracle[k]-y[k])**2 for k in keys])))
    print(json.dumps({site:{**{k:{"误差":v["score"],"最差台":v["worst_group"]} for k,v in metrics.items()},
                            "实际改变":changed,"逐条事后两端最优":ideal}},ensure_ascii=False),flush=True)
    return {"requests":requests,"scores":metrics,"changes":changed,"oracle":ideal,"events":events}


def main():
    with LocalStore(DEST/"lock").exclusive(),predictors() as pred:
        if (DEST/"scores.json").exists():
            raise ValueError("本次高估比较已有结果")
        priors = read(STUDY/"scores.json")
        pkg = official("engine")[0]
        data = engines()
        result = {}
        for site,stream,truth in (("engine",public_stream(),lambda:{r.sample_id:r.label for r in pkg.validation}),
                                  ("phm08",failure_stream(data),data[3])):
            req = priors[site]["request"]
            prior = read(STUDY/"accepted"/(req+".json"))["payload"]
            if digest([site,prior])!=req:
                raise ValueError("原交付身份不符")
            result[site] = run(site,stream,truth,pred,prior["predictions"]["full-early-native"])
        atomic_json(DEST/"scores.json",result)


if __name__=="__main__":
    main()
