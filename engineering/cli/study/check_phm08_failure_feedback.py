"""按共同启动时钟重放完整 PHM08 轨迹，只允许已失效的反馈设备揭晓标签。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import old_step
from compare_current_neighbors import DEST
from compare_group_objectives import ROOT, _trainer, atomic_json, digest, metric, official, read, remaining_cycles, summary
from harness.models.feedback_neighbors import CurrentNeighbors, fit_weight
from unseen_sources import PHM08_TRAIN_SHA, RESULT, track_rows


def main(trees=False,native=False,experts=False):
    pkg,checksum,batches = official("engine")
    ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
    train = [r for r in pkg.train if r.sample_id in ids]
    fit = np.asarray([r.values for r in train]),np.asarray([r.label for r in train])
    source = ROOT/"data/public/phm08/train.txt"
    study = read(RESULT/"issued/engine/study.json")
    if sha256(source.read_bytes()).hexdigest()!=PHM08_TRAIN_SHA or study["source"]["sha256"]!=PHM08_TRAIN_SHA:
        raise ValueError("源轨迹与原比较不同")
    tracks = {f"PHM08/{u}":x for u,x in track_rows(source) if f"PHM08/{u}" in study["roles"]}
    released, history, units, window = set(),[],{},{}
    streams = {arm:{} for arm in ("frozen","frozen-half","regular","regular-half","feedback-weight")}
    if trees:
        from compare_fixed_tree_references import OUT,frozen,tree_values
        from harness.models.life_reference import choose
        fitted,receipts=frozen("engine")
        tree=fitted["catboost"]
        extra={k:{} for k in ("catboost-only","catboost-half","tree-confirm")}
    archive, = (ROOT/"data/official/engine").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"C") as trainer:
        p = Path(tmp)/"root.npz"
        p.write_bytes(pkg.checkpoint)
        root = model = trainer.Model.load(p)
        if native:
            from compare_fixed_tree_references import OUT
            from engineering.cli.workflow.serve_life_checkpoint import confirm_candidate
            bank=OUT/"tree-hidden/engine/models"
            candidate=trainer.Model.load(bank/"whole-teacher.npz")
            if sha256((bank/"whole-teacher.npz").read_bytes()).hexdigest()!=read(bank/"training.json")["arms"]["whole-teacher"]["sha256"]:
                raise ValueError("冻结赛方候选身份发生变化")
            native_predictions={"native-only":{},"native-confirm":{}}
            if experts:
                from score_life_experts import LifeExperts
                expert=LifeExperts()
                expert_predictions={}
        lookup = CurrentNeighbors(None,checksum,train)
        for r in range(10):
            cutoff = 20*r
            groups = {k:[] for k in ("explore","adopt","serve")}
            issued_units = []
            for unit,x in tracks.items():
                role = study["roles"][unit]
                if role=="serve":
                    end = len(x) if r==9 else min(len(x),cutoff+20)
                    indices = range(cutoff,end)
                elif len(x)<=cutoff and unit not in released:
                    indices = np.linspace(0,len(x)-1,min(8,len(x)),dtype=int)
                    released.add(unit)
                    issued_units.append(unit)
                else:
                    continue
                for j in indices:
                    row = x[j]
                    groups[role].append({"sample_id":f"{unit}/{j+1}","unit_id":unit,
                        "values":[*row[3:],*row[:3]],"label":None if role=="serve" else len(x)-j-1})
            explore,adopt,serve = (groups[k] for k in ("explore","adopt","serve"))
            start = perf_counter()
            model,_,event,base_plain = old_step("engine","fixed",model,fit,explore,adopt,serve,None)
            rows,base,ref = lookup.observations(model,serve)
            keys = [s["sample_id"] for s in rows]
            if keys!=list(base_plain) or any(abs(v-base_plain[i])>1e-8 for i,v in zip(keys,base,strict=True)):
                raise ValueError("普通维护的交接未复现")
            w = []
            if explore and adopt:
                for part in (explore,adopt):
                    records,b,n = lookup.observations(model,part)
                    w.append(fit_weight(b,n-b,np.array([s["label"] for s in records])))
            weight = min(w,default=0.)
            original = root.predict(np.asarray([s["values"] for s in serve]))
            values = (original,.5*(original+ref),base,.5*(base+ref),base+weight*(ref-base))
            if units.keys()&set(keys) or any(s["label"] is not None for s in serve):
                raise ValueError("服务输入重复或得到过答案")
            for a,v in zip(streams,values,strict=True):
                streams[a].update(zip(keys,map(float,v),strict=True))
            if trees:
                def choices(rows):
                    if not rows:
                        return {}
                    x=np.array([s["values"] for s in rows])
                    return {"current":model.predict(x),"catboost-only":tree_values(tree,"engine",x)}
                decision=choose(choices(explore),choices(adopt),[s["label"] for s in explore],
                                [s["label"] for s in adopt],"tree-confirm")
                cv=tree_values(tree,"engine",np.array([s["values"] for s in serve]))
                for key,output in (("catboost-only",cv),("catboost-half",.5*(cv+base)),
                                   ("tree-confirm",cv if decision["chosen"]=="catboost-only" else base)):
                    extra[key].update(zip(keys,output.tolist(),strict=True))
            units.update({s["sample_id"]:s["unit_id"] for s in serve})
            window.update(dict.fromkeys(keys,r))
            history.append({"cutoff":cutoff,"new_feedback_devices":sorted(issued_units),
                            "explore_rows":len(explore),"confirm_rows":len(adopt),"service_rows":len(serve),
                            "trained":event["physical_fit"],"adopted":event["adopted"],"weight":weight,
                            "shared_seconds":perf_counter()-start})
            if trees:
                history[-1]["tree_choice"]=decision
            if native:
                allowed,checks=confirm_candidate(model,candidate,explore,adopt)
                forecast=candidate.predict(np.array([s["values"] for s in rows]))
                native_predictions["native-only"].update(zip(keys,forecast.tolist(),strict=True))
                native_predictions["native-confirm"].update(zip(keys,(forecast if allowed else base).tolist(),strict=True))
                history[-1]["native_choice"]={"candidate":bool(allowed),"checks":checks}
                if experts:
                    proposals,weights=expert.advance(model,candidate,explore+adopt,rows)
                    for arm,v in proposals.items():
                        expert_predictions.setdefault("expert-"+arm,{}).update(zip(keys,v.tolist(),strict=True))
                    history[-1]["expert_weights"]=weights
    if digest(sorted(units))!=study["service_ids_sha256"] or len(units)!=study["service_rows"]:
        raise ValueError("新时钟改变了服务清单")
    out = DEST/"live"/"failure-feedback"
    if trees:
        prior=read(out/"issued.json")
        if any(max(abs(p[i]-prior["predictions"][arm][i]) for i in p)>1e-7 for arm,p in streams.items()):
            raise ValueError("同一失效后回标的普通参照没有复现")
        out=OUT/"routing"/"failure-feedback"
        streams.update(extra)
    if native:
        prior=read(out/"issued.json")["predictions"]
        if any(max(abs(v-prior[k][i]) for i,v in p.items())>1e-7 for k,p in streams.items()):
            raise ValueError("候选交付的原时钟与成熟对照有变化")
        streams.update(native_predictions)
        out=OUT/"tree-hidden/engine/live/failure-feedback"
        if experts:
            streams.update(expert_predictions)
            out=OUT/"tree-hidden/engine/expert-weights/failure-feedback"
    path = out/"issued.json"
    atomic_json(path,{"source":PHM08_TRAIN_SHA,"bank":lookup.identity,"fit":digest(sorted(ids)),
                      "history":[{k:v for k,v in h.items() if k!="shared_seconds"} for h in history],"predictions":streams})
    saved = read(path)["predictions"]
    answers = remaining_cycles()
    table = summary("engine",answers,{"parent":saved["regular"],**saved},units)
    for arm,p in saved.items():
        table[arm]["rounds"]=[metric("engine",answers,{k:v for k,v in p.items() if window[k]==r}) for r in range(10)]
    atomic_json(out/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table,
                                  "shared_seconds":sum(h["shared_seconds"] for h in history),
                                  "training_calls":sum(h["trained"] for h in history),
                                  "rows":len(units),"devices":len(set(units.values()))})
    print([(h["cutoff"],h["explore_rows"],h["confirm_rows"],h["weight"]) for h in history])
    print({a:(round(table[a]["score"],3),round(table[a]["worst_group"],3)) for a in saved})


if __name__=="__main__":
    from argparse import ArgumentParser
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--trees",action="store_true")
    parser.add_argument("--native",action="store_true",help="同一失效回标时钟上的赛方单权重候选")
    parser.add_argument("--experts",action="store_true",help="评估延迟回标和当窗回标的专家融合")
    args=parser.parse_args()
    main(args.trees or args.native or args.experts,args.native or args.experts,args.experts)
