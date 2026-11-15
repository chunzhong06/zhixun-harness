"""同一份树参照只从合法 fit 训练一次，将输出与各轮实际发布的工业模型配对。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time, perf_counter
from zipfile import ZipFile

import numpy as np

from compare_calibration_expiry import source_window
from compare_group_objectives import BASE, ROOT, PROFILES, _trainer, atomic_json, digest, delivered, metric, official, read, reference_folders, remaining_cycles, remaining_minutes, summary
from harness.contracts import ActionPlan
from harness.models.gas_views import pair_model
from harness.models.life_reference import MODES,alternatives,choose
from harness.models.model_artifacts import ModelArtifacts
from harness.models.pipeline_recipe import PipelineRecipe
from harness.models.training_tools import SklearnBackend

OUT = BASE/"fixed-tree-references"
FAMILIES = ("forest","catboost")


def tree_values(pipeline, task, x):
    predictor = pipeline.named_steps["predictor"]
    opts = {"thread_count":1} if type(predictor).__module__.startswith("catboost") else {}
    if task=="gas":
        if list(pipeline.classes_)!=list(range(1,7)):
            raise ValueError("树分类器的概率列不匹配竞赛类别")
        values = pipeline.predict_proba(x,**opts)
    else:
        values = pipeline.predict(x,**opts)
    if not np.isfinite(values).all():
        raise ValueError("树给出了非有限输出")
    return values


def classes(task,values):
    return values.argmax(axis=1)+1 if task=="gas" else values


def trained(task,pkg,rows,checksum):
    backend = SklearnBackend(execution_enabled=True)
    models, records = {}, {}
    folder = OUT/"trained"/task
    folder.mkdir(parents=True,exist_ok=True)
    for name in FAMILIES:
        recipe = PipelineRecipe(predictor=name,processor="none",max_depth=6,
                                class_weight="balanced" if task=="gas" else "none")
        start = process_time()
        model = backend.fit(pkg.task,"source",ActionPlan("balanced",pipeline=recipe),rows)
        seconds = process_time()-start
        blob = backend.dump(model)
        path = folder/(name+".joblib")
        path.write_bytes(blob)
        models[name] = backend.load(path.read_bytes())
        records[name] = {"package":checksum,"recipe":recipe.key,"format":"trusted-local-joblib",
                         "sha256":sha256(blob).hexdigest(),"cpu_seconds":seconds,"bytes":len(blob),
                         "rows":len(rows),"fit":digest([r.sample_id for r in rows]),
                         "environment":backend.specification}
    atomic_json(folder/"receipt.json",records)
    return models,records


def frozen(task):
    folder=OUT/"trained"/task
    receipts=read(folder/"receipt.json")
    backend=SklearnBackend(execution_enabled=True)
    models={}
    for name in FAMILIES:
        blob=(folder/(name+".joblib")).read_bytes()
        if sha256(blob).hexdigest()!=receipts[name]["sha256"]:
            raise ValueError("固定树内容发生变化")
        models[name]=backend.load(blob)
    return models,receipts


def paths(cohort,task):
    fixed,paper = reference_folders(cohort,"mature")
    old = BASE/"selected"/task if cohort==task else fixed.parent/"project"
    if task=="engine":
        current = BASE/"live-first"/("public" if task==cohort else "")/cohort
    else:
        current = old
    options = {"fixed":fixed,"paper":paper}
    if task!=cohort:
        options["simple"]=fixed.parent/"simple"
    if task=="bearing":
        options["stump"]=BASE/"residual-trees/live"/("public" if cohort==task else "")/cohort
    return current,options


def evaluate(cohort,task,models,receipts,pkg,median=None, *, phase_scale=None):
    folder,refs = paths(cohort,task)
    state = read(folder/"state.json")
    run = read(folder/"run.json")["rounds"]
    store = ModelArtifacts(folder/"models")
    controls = {arm:delivered(path) for arm,path in refs.items()}
    if task=="engine":
        controls["five-devices"] = read(BASE/"engine-neighbor-updates/switch/live"/cohort/"fixed-half.json")["predictions"]
    output = {"current":{},**{f"{name}-{mode}":{} for name in models for mode in ("only","half","confirm")}}
    if phase_scale is not None:
        output.update({"median":{},"half-life":{}})
        ref_path=OUT/"routing"/cohort/"issued.json"
        prior=read(ref_path)["predictions"]
        controls.update({a:prior[a] for a in ("median-confirm","tree-confirm")})
    elif median is not None:
        output.update({k:{} for k in ("median",*MODES)})
    history = []
    archive, = (ROOT/"data/official"/task).glob("*.zip")
    times = dict.fromkeys(models,0.)
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),PROFILES[task]) as trainer:
        head = pair_model(trainer.Model) if task=="gas" else trainer.Model
        checkpoint = Path(tmp)/"current.npz"
        for r,record in enumerate(run):
            header,blob = store.read(record["model_artifact"])
            checkpoint.write_bytes(blob)
            model = head.load(checkpoint)
            groups = source_window(state["samples"],r)
            ids = [[s["sample_id"] for s in rows] for rows in groups]
            x = np.array([s["values"] for rows in groups for s in rows])
            labels = [{s["sample_id"]:s["label"] for s in rows} for rows in groups[:2]]
            if len(set(i for group in ids for i in group))!=len(x) or any(s["label"] is not None for s in groups[2]):
                raise ValueError("当轮服务分片与已揭晓反馈重合")
            base = model.predict(x)
            indexes = np.cumsum([len(k) for k in ids])[:2]
            parts = np.split(classes(task,base),indexes)
            pred = dict(zip(ids[2],parts[2].tolist(),strict=True))
            actual = read(folder/record["submitted"])["payload"]
            if actual["model_id"]!=record["selected"] or pred.keys()!=actual["predictions"].keys() or any(abs(v-actual["predictions"][i])>1e-7 for i,v in pred.items()):
                raise ValueError("实际工业权重的服务预测未复现")
            output["current"].update(pred)
            event={"batch":record["batch"],"weights":header["sha256"],"arms":{}}
            for name,pipeline in models.items():
                start = perf_counter()
                tree = tree_values(pipeline,task,x)
                if name=="phase":
                    tree=np.clip(tree,0.,1.)*phase_scale
                times[name] += perf_counter()-start
                blend = np.split(classes(task,.5*(tree+base)),indexes)
                passed = True
                checks=[]
                for truth,b,n,keys in zip(labels,parts[:2],blend[:2],ids[:2],strict=True):
                    baseline,child = [metric(task,truth,dict(zip(keys,v.tolist(),strict=True))) for v in (b,n)]
                    checks.append([baseline,child])
                    passed &= child>baseline+1e-12 if task=="gas" else child<baseline-1e-12
                for mode,v in (("only",classes(task,tree)[indexes[-1]:]),("half",blend[2]),("confirm",blend[2] if passed else parts[2])):
                    output[f"{name}-{mode}"].update(zip(ids[2],v.tolist(),strict=True))
                event["arms"][name]={"checks":checks,"using_half":bool(passed)}
                if phase_scale is not None:
                    output["median"].update(dict.fromkeys(ids[2],median))
                    output["half-life"].update(dict.fromkeys(ids[2],phase_scale/2))
                elif name=="catboost" and median is not None:
                    versions=alternatives(base,tree,median)
                    groups_out=[{k:v[ix] for k,v in versions.items()} for ix in
                                (slice(0,indexes[0]),slice(indexes[0],indexes[1]),slice(indexes[1],None))]
                    output["median"].update(zip(ids[2],groups_out[2]["median"].tolist(),strict=True))
                    event["choices"]={}
                    for mode in MODES:
                        decision=choose(groups_out[0],groups_out[1],list(labels[0].values()),list(labels[1].values()),mode)
                        event["choices"][mode]=decision
                        output[mode].update(zip(ids[2],groups_out[2][decision["chosen"]].tolist(),strict=True))
            history.append(event)
    owner=OUT/"phase" if phase_scale is not None else (OUT/"routing" if median is not None else OUT)
    sealed=owner/cohort/"issued.json"
    atomic_json(sealed,{"trees":{k:v["sha256"] for k,v in receipts.items()},"history":history,"predictions":{**controls,**output}})
    saved = read(sealed)["predictions"]
    truth= {r.sample_id:r.label for r in pkg.validation}
    if cohort=="phm08":
        truth=remaining_cycles()
    elif cohort=="pronostia":
        truth=remaining_minutes("full-test-manifest.json")
    devices={k:state["samples"][k]["unit_id"] for k in output["current"]}
    results = summary(task,truth,{"parent":saved["current"],**saved},devices)
    for arm,p in saved.items():
        results[arm]["rounds"]=[metric(task,truth,{i:v for i,v in p.items() if state["samples"][i]["available_at"]==r}) for r in range(10)]
    atomic_json(sealed.with_name("scores.json"),{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":results,
                                               "reference_prediction_seconds":times})
    print(cohort,{k:(round(results[k]["score"],5),round(results[k]["worst_group"],4)) for k in saved},flush=True)


def main(task,routing):
    pkg,checksum,batches = official(task)
    fit_ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
    rows = [r for r in pkg.train if r.sample_id in fit_ids]
    if len(rows)!=len(fit_ids) or {r.sample_id for r in pkg.validation}&fit_ids:
        raise ValueError("训练行不属于预定独立训练库")
    if routing and task=="gas":
        raise ValueError("只有回归任务使用寿命参照选择")
    models,receipts = frozen(task) if routing else trained(task,pkg,rows,checksum)
    if any(r["fit"]!=digest([s.sample_id for s in rows]) or r["package"]!=checksum for r in receipts.values()):
        raise ValueError("冻结参照的训练清单不同")
    cohorts = {"gas":["gas"],"engine":["engine","phm08"],"bearing":["bearing","pronostia"]}[task]
    for cohort in cohorts:
        evaluate(cohort,task,models,receipts,pkg,float(np.median([r.label for r in rows])) if routing else None)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=PROFILES)
    parser.add_argument("--routing",action="store_true")
    args=parser.parse_args()
    main(args.task,args.routing)
