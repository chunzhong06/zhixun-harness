"""在参考轴承失效后才训练阶段树和原包权重；同一候选分别强制发布与由独立设备确认。"""
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from sklearn.base import clone

from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, official, read, summary
from harness.models.model_artifacts import ModelArtifacts
from harness.models.training_tools import SklearnBackend
from score_completed_choice import PHASE, femto, fingerprint, xjtu
from score_unseen_xjtu_devices import LOCAL


DEST = OUT / "completed-refit"
PROTOCOL = ROOT / "engineering/configs/experiments/bearing-completed-refit.json"


def equal_device_weights(units, mass):
    _, inverse, counts = np.unique(units, return_inverse=True, return_counts=True)
    return mass / (len(counts) * counts[inverse])


def evaluate(name, stream, source, trainer, original_path, direct):
    ends, history, service, answers, manifests = stream
    folder = DEST / name
    models = ModelArtifacts(folder / "phase-models")
    native_models = ModelArtifacts(folder / "native-models")
    backend = SklearnBackend(execution_enabled=True)
    report = read(PHASE/"run.json")
    manifest, blob = ModelArtifacts(PHASE/"references").read(report["tree_artifact"])
    template = backend.load(blob)
    scale = float(report["scale_minutes"])
    x_source = np.asarray([r.values for r in source])
    y_source = np.asarray([r.label for r in source])
    life = np.asarray([r.label+r.time_end-1 for r in source])
    source_weight = equal_device_weights([r.unit_id for r in source], len(source))
    if len(source)!=72 or len(set(source_weight))!=1 or len(service)!=len({k for k,*_ in service}):
        raise ValueError("冻结的训练输入或服务身份不符")
    observations = {u:[r for r in history if r[0]==u] for u in ends}
    if (set(observations) & {r[1] for r in service} or any(len(v)!=5 for v in observations.values())
            or {r.unit_id for r in source} & (set(observations) | {r[1] for r in service})):
        raise ValueError("参考、服务或训练设备角色不符合协议")
    events, fit, confirm, released, costs = [], [], [], set(), []
    def phase_predict(model, x):
        return np.clip(tree_values(model,"bearing",x),0.,1.)*scale
    def prediction(family, model, x):
        return phase_predict(model,x) if family=="phase" else model.predict(x)
    original = trainer.Model.load(original_path)
    trained = {"phase":[template], "native":[original]}
    active = {"phase":0,"native":0}
    issued, owners, clocks = defaultdict(dict), {}, {}
    service_order = sorted(service,key=lambda r:(r[2],r[0]))
    with TemporaryDirectory() as work:
        for time, batch in groupby(service_order,key=lambda r:r[2]):
            ready = sorted((u for u in ends if ends[u]<time and u not in released),key=lambda u:(ends[u],u))
            new_training = []
            for u in ready:
                if len(released)%2==0:
                    fit.append(u)
                    new_training.append(u)
                else:
                    confirm.append(u)
                released.add(u)
            event = {"time":time,"received":ready,"fit":list(fit),"confirm":list(confirm)}
            if new_training:
                rows = [r for u in fit for r in observations[u]]
                local_x = np.array([r[2] for r in rows])
                local_y = np.array([r[3] for r in rows])
                target_phase = local_y / np.array([ends[r[0]] for r in rows])
                info = {"source_ids":[r.sample_id for r in source],"target_ids":[r[1] for r in rows],
                        "protocol": fingerprint(PROTOCOL),"site":name}
                start = perf_counter()
                model = clone(template)
                model.fit(np.vstack((x_source,local_x)),np.concatenate((y_source/life,target_phase)),
                          predictor__sample_weight=np.concatenate((source_weight,equal_device_weights([r[0] for r in rows], len(source)))))
                cost_phase = perf_counter()-start
                ref = models.create({**info,"base":manifest["sha256"]},lambda:backend.dump(model))
                trained["phase"].append(backend.load(models.read(ref)[1]))
                # 克隆上个训练候选。即使其未发布，always 和 confirmed 两组在本轮也比较同一权重。
                parent_path = Path(work)/f"parent-{len(fit)}.npz"
                checkpoint = Path(work)/f"candidate-{len(fit)}.npz"
                trained["native"][-1].save(parent_path)
                candidate = trainer.Model.load(parent_path)
                rng = np.random.default_rng(np.random.SeedSequence([17,len(fit)]))
                draw = [j%len(fit)*5 + int(rng.integers(5)) for j in range(len(source))]
                start = perf_counter()
                candidate.fit(np.vstack((x_source,local_x[draw])),np.concatenate((y_source,local_y[draw])),1)
                cost_native = perf_counter()-start
                candidate.save(checkpoint)
                native_ref = native_models.create({**info,"draw":draw,"starting_npz":fingerprint(original_path)},
                                                  checkpoint.read_bytes)
                checkpoint.write_bytes(native_models.read(native_ref)[1])
                trained["native"].append(trainer.Model.load(checkpoint))
                costs.append({"phase_seconds":cost_phase,"native_seconds":cost_native})
                event.update(phase_ref=ref,native_ref=native_ref)
            if ready:
                for family, versions in trained.items():
                    errors = None
                    latest = len(versions)-1
                    if confirm:
                        losses = []
                        for version in (active[family],latest):
                            per_device = []
                            for u in confirm:
                                records = observations[u]
                                v = prediction(family,versions[version],np.array([r[2] for r in records]))
                                per_device.append(float(np.mean((v-np.array([r[3] for r in records]))**2)))
                            losses.append(float(np.mean(per_device)))
                        errors = losses
                        if losses[1]<losses[0]:
                            active[family]=latest
                    event[family]={"latest":latest,"deployed":active[family],"confirm_mse":errors}
                events.append(event)
            batch=list(batch)
            x=np.asarray([r[3] for r in batch])
            ids=[r[0] for r in batch]
            forecasts = {"direct":tree_values(direct,"bearing",x),"median":np.full(len(x),np.median(y_source))}
            for family, versions in trained.items():
                for mode, version in (("frozen",0),("mixed",len(versions)-1),("confirmed",active[family])):
                    forecasts[family+"-"+mode] = prediction(family,versions[version],x)
            for a,v in forecasts.items():
                issued[a].update(zip(ids,v.tolist(),strict=True))
            owners.update((r[0],r[1]) for r in batch)
            clocks.update(dict.fromkeys(ids,time))
    path=folder/"issued.json"
    atomic_json(path,{"protocol_sha256":fingerprint(PROTOCOL),"initial_phase":manifest["sha256"],"source_scale":scale,
                      "initial_native":fingerprint(original_path),"source_files":{p.name:fingerprint(p) for p in manifests},
                      "events":events,"device":owners,"service_minutes":clocks,"predictions":issued})
    p=read(path)["predictions"]
    y=answers()
    if set(y)!=owners.keys():
        raise ValueError("独立评分行数不符")
    table=summary("bearing",y,{"parent":p["phase-frozen"],**p},owners)
    del table["parent"]
    for a,row in table.items():
        row["mae"]=float(np.mean([abs(v-y[k]) for k,v in p[a].items()]))
    frozen=table["phase-frozen"]
    fixed_best=min(table[a]["score"] for a in ("phase-frozen","native-frozen","direct","median"))
    gates={}
    for f in trained:
        mixed,auto = table[f+"-mixed"],table[f+"-confirmed"]
        gates[f]={"mixed":mixed["score"]<=.99*frozen["score"] and mixed["worst_group"]<=1.01*frozen["worst_group"],
                  "decision":auto["score"]<=.99*min(mixed["score"],fixed_best) and auto["worst_group"]<=1.01*frozen["worst_group"]}
    result={"issued_sha256":fingerprint(path),"arms":table,"fit_events":len(costs),"cost_per_update":costs,
            "feedback_rows":sum(len(observations[u]) for u in released),"rows":len(service),"gate":gates}
    atomic_json(folder/"scores.json",result)
    print(name,{a:round(v["score"],3) for a,v in table.items()},gates,flush=True)


if __name__=="__main__":
    package,_,batches=official("bearing")
    source=[r for r in batches[0].samples if r.role=="fit"]
    direct=frozen("bearing")[0]["catboost"]
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B") as trainer:
        original=Path(tmp)/"original.npz"
        original.write_bytes(package.checkpoint)
        for name,stream in (("femto",femto(48)),("xjtu",xjtu(package,LOCAL/"fifth-offset6.json"))):
            evaluate(name,stream,source,trainer,original,direct)
