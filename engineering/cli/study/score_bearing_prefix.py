"""重放十台轴承的实际工况、时间和已交付的对手预测，仅在上一台失效后收回其误差。"""
import json
from pathlib import Path
import numpy as np
from score_paderborn_forward import (ARCHIVE_SIZE, DATA, OUT, ROOT, Tools, atomic_json, devices, digest,
                                     fingerprint, frozen, LocalAcceptance, read)
from harness.runtime.storage import LocalStore
from harness.models.bearing_prefix import PrefixWeight, prefix_condition, fixed_prefix_predictions

DEST = OUT.parent/"bearing-prefix"
SPEC = ROOT/"engineering/configs/experiments/bearing-prefix.json"
LEFT = "static/conditional-life"
RIGHT = "update/phase-tree"
PRIMARY = "随振动变化学习配比"
GLOBAL = "按既往误差学全局配比"


def source_trials(initial,clock,template):
    result=[]
    for unit in initial:
        train=[u for u in initial if u.name!=unit.name]
        tools=Tools(train,clock,template)
        age=unit.ages[clock]
        pred=tools.predict(unit.inputs(clock),age)
        result.append((prefix_condition(unit.spectral),pred["conditional-life"],pred["phase-tree"],unit.durations[clock]-age))
    return result


def describe(cols, truth):
    scores={}
    for k,blocks in cols.items():
        errors=[(p-y)**2 for p,y in zip(blocks,truth,strict=True)]
        rmse=[float(np.sqrt(err.mean())) for err in errors]
        scores[k]={"RMSE":float(np.sqrt(np.concatenate(errors).mean())),"最差整台":max(rmse),
                   "等设备平均":float(np.mean(rmse)),"逐台":rmse}
    return scores


def previous_delivery(unit, clock, event, summary):
    p=read(OUT/clock/"accepted"/(event["request"]+".json"))["payload"]
    if (digest([summary["protocol_sha256"],clock,p])!=event["request"] or p["ids"]!=unit.ids
            or unit.name!=p["unit"] or summary["manifest_sha256"]!=fingerprint(DATA/"manifest.json")):
        raise ValueError("既有轴承回执或原始输入身份改变")
    return {k:np.array(v) for k,v in p["predictions"].items()}


def run(units, clock, template):
    hist=source_trials(units[:ARCHIVE_SIZE],clock,template)
    original=read(OUT/clock/"scores.json")
    released={}
    pending={}
    events=[]
    receiver=LocalAcceptance()
    receiver.output=DEST/clock/"accepted"
    version=fingerprint(SPEC)
    choices=[]
    for unit,event in zip(units[ARCHIVE_SIZE:],original["events"],strict=True):
        for past in units[ARCHIVE_SIZE:]:
            if past.name in pending and past.name not in released and past.end<unit.start:
                cond,l,r=pending[past.name]
                released[past.name]=(cond,l,r,past.durations[clock]-past.ages[clock])
        old=previous_delivery(unit,clock,event,original)
        cond=prefix_condition(unit.spectral)
        for n in (1,min(12,len(cond)),len(cond)//2):
            if not np.array_equal(prefix_condition(unit.spectral[:n]),cond[:n]):
                raise ValueError("趋势依赖本台未来读数")
        left,right=old[LEFT],old[RIGHT]
        model=PrefixWeight().fit([*hist,*released.values()])
        col={**old,**fixed_prefix_predictions(cond,left,right)}
        col[PRIMARY]=model.predict(cond,left,right)
        col[GLOBAL]=left+model.global_weight*(right-left)
        payload={"ids":unit.ids,"基准回执":event["request"],"已回标":list(released),"配比":model.weights.tolist(),
                 "预测":{k:v.tolist() for k,v in col.items()}}
        request=digest([version,clock,payload])
        if not receiver.submit(request,payload)["accepted"]:
            raise ValueError("前缀预测交接失败")
        events.append(request)
        choices.append(model.weights.tolist())
        pending[unit.name]=(cond,left,right)
    truth,cols=[],{}
    for unit,key in zip(units[ARCHIVE_SIZE:],events,strict=True):
        sent=read(receiver.output/(key+".json"))["payload"]
        if digest([version,clock,sent])!=key:
            raise ValueError("交付回执更改")
        truth.append(unit.durations[clock]-unit.ages[clock])
        for k,v in sent["预测"].items():cols.setdefault(k,[]).append(np.array(v))
    scores=describe(cols,truth)
    fixed=min((k for k in scores if k.startswith(("固定","static/","update/"))),key=lambda k:scores[k]["RMSE"])
    paper=min(("static/phme-feature","update/phme-feature"),key=lambda k:scores[k]["RMSE"])
    if abs(scores[original["best_fixed"]]["RMSE"]-original["scores"][original["best_fixed"]]["rmse"])>1e-8:
        raise ValueError("基准分数复核不符")
    rating=scores[PRIMARY]
    decision={"主方案":PRIMARY,"最佳固定":fixed,"论文":paper,"现场误差":rating["RMSE"],"固定误差":scores[fixed]["RMSE"],
              "论文误差":scores[paper]["RMSE"],"最差整台":rating["最差整台"],
              "开发通过":rating["RMSE"]<=.98*min(scores[k]["RMSE"] for k in (fixed,paper,"once",GLOBAL))
                    and rating["最差整台"]<=1.01*scores[fixed]["最差整台"]}
    return {"scores":scores,"decision":decision,"requests":events,"设备":[u.name for u in units[ARCHIVE_SIZE:]],"配比":choices}


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("相同协议已有结果")
        units=devices()
        template=frozen("bearing")[0]["catboost"]
        reports={c:run(units,c,template) for c in ("calendar","capped")}
        atomic_json(DEST/"scores.json",reports)
        print(json.dumps({k:v["decision"] for k,v in reports.items()},ensure_ascii=False))
