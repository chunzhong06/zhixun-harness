"""把冻结的成熟树组件迁移到源库之外。新轴承波形在方法确定后按固定时间间隔取得。"""
from argparse import ArgumentParser
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT,FAMILIES,paths,tree_values
from compare_error_slice import RAW,SUBSETS,read_cmapss,test_answers
from compare_group_objectives import BASE,ROOT,_trainer,atomic_json,digest,metric,official,read,summary
from harness.models.bearing_stumps import stump_model
from harness.models.model_artifacts import ModelArtifacts
from harness.models.training_tools import SklearnBackend
from score_ims_bearings import ims_inputs
from score_unseen_bearing_windows import window_inputs,window_answers


def suites(task, routing):
    if task=="bearing":
        for offset in ((40,) if routing else (24,40)):
            rounds,metadata,ends = window_inputs(offset)
            yield f"new{offset}",rounds,{k:v[0] for k,v in metadata.items()},lambda metadata=metadata,ends=ends:window_answers(metadata,ends)
        _,stamps,keys,idx,groups,x = ims_inputs()
        def ims_truth():
            times=[datetime.strptime(Path(f).name,"%Y.%m.%d.%H.%M.%S") for f in stamps]
            return {k:(times[-1]-times[idx[k]]).total_seconds()/60 for k in keys}
        yield "ims",{9:list(zip(keys,x,strict=True))},groups,ims_truth
        raw = read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"]
        rows = [("KAIST/"+r["file"],r["features"]) for r in raw if "features" in r]
        def kaist_truth():
            end=datetime.fromisoformat(raw[-1]["time"])
            return {"KAIST/"+r["file"]:(end-datetime.fromisoformat(r["time"])).total_seconds()/60 for r in raw if "features" in r}
        yield "kaist",{9:rows},dict.fromkeys((k for k,_ in rows),"KAIST"),kaist_truth
    else:
        tracks = {s:read_cmapss(RAW,subset=s,partition="test") for s in SUBSETS}
        for terminal in (False,True):
            rounds,meta,groups = {},{},{}
            for s,units in tracks.items():
                for unit in units:
                    cycles = [(9,len(unit.values))] if terminal else [(r,16+20*r) for r in range(10) if s!="FD001" and 16+20*r<len(unit.values)]
                    for r,c in cycles:
                        k = unit.unit_id if terminal else f"{s}/{unit.unit}/{c}"
                        row = unit.values[c-1]
                        rounds.setdefault(r,[]).append((k,[*row[3:],*row[:3]]))
                        groups[k] = unit.unit_id
                        meta[k] = (unit.unit_id,len(unit.values)-c)
            def truth(meta=meta):
                ends={k:v for s,units in tracks.items() for k,v in test_answers(RAW,s,units).items()}
                return {k:ends[u]+gap for k,(u,gap) in meta.items()}
            yield "terminal" if terminal else "cycles16",rounds,groups,truth


def evaluate(task,routing,phase=False):
    cohort = "pronostia" if task=="bearing" else "phm08"
    pkg,checksum,batches=official(task)
    fit_ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    labels=np.asarray([r.label for r in pkg.train if r.sample_id in fit_ids])
    current,folders=paths(cohort,task)
    folders={"current":current,**folders}
    if task=="bearing":
        folders["bias"]=BASE/"readout-live"/cohort
    trained=OUT/"trained"/task
    receipts=read(trained/"receipt.json")
    models={}
    backend=SklearnBackend(execution_enabled=True)
    for name in FAMILIES:
        blob=(trained/(name+".joblib")).read_bytes()
        if sha256(blob).hexdigest()!=receipts[name]["sha256"] or receipts[name]["package"]!=checksum:
            raise ValueError("冻结训练来源或模型已变化")
        models[name]=backend.load(blob)
    if phase:
        directory=OUT/"phase/trained"/task
        phase_receipt=read(directory/"receipt.json")
        blob=(directory/"phase.joblib").read_bytes()
        if sha256(blob).hexdigest()!=phase_receipt["sha256"]:
            raise ValueError("已冻结比例模型变更")
        models={"catboost":models["catboost"],"phase":backend.load(blob)}
        receipts={"catboost":receipts["catboost"],"phase":phase_receipt}
    owner=OUT/"phase" if phase else (OUT/"routing" if routing else OUT)
    source=read(owner/cohort/"issued.json")
    if source["trees"]!={k:v["sha256"] for k,v in receipts.items()}:
        raise ValueError("冻结模型身份与已选方案不同")
    weights={arm:read(f/"run.json")["rounds"] for arm,f in folders.items()}
    archive,=(ROOT/"data/official"/task).glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"B" if task=="bearing" else "C") as trainer:
        p=Path(tmp)/"frozen.npz"
        for suite,windows,groups,truth in suites(task,routing):
            preds,units,versions={}, {},{}
            for r,rows in sorted(windows.items()):
                keys,values=zip(*rows,strict=True)
                x=np.asarray(values)
                options={}
                for arm,folder in folders.items():
                    record=weights[arm][r]
                    header,blob=ModelArtifacts(folder/"models").read(record["model_artifact"])
                    if record["selected"]!=read(folder/record["submitted"])["payload"]["model_id"]:
                        raise ValueError("源轮次没有交接该权重")
                    versions.setdefault(arm,{})[r]=header["sha256"]
                    p.write_bytes(blob)
                    cls=stump_model(trainer.Model) if arm=="stump" else trainer.Model
                    options[arm]=cls.load(p).predict(x)
                options["mean"]=np.full(len(x),labels.mean())
                options["median"]=np.full(len(x),np.median(labels))
                for name,model in models.items():
                    v=tree_values(model,task,x)
                    if name=="phase":
                        scale=phase_receipt["scale"]
                        v=np.clip(v,0.,1.)*scale
                        options["half-life"]=np.full(len(x),scale/2)
                    half=.5*(options["current"]+v)
                    options.update({name+"-only":v,name+"-half":half,
                                    name+"-source-confirm":half if source["history"][r]["arms"][name]["using_half"] else options["current"]})
                if routing:
                    for mode,decision in source["history"][r]["choices"].items():
                        available = suite=="new40"
                        options[mode]=options[decision["chosen"] if available else "current"]
                for arm,v in options.items():
                    preds.setdefault(arm,{}).update(zip(keys,v.tolist(),strict=True))
                units.update(dict.fromkeys(keys,r))
            if task=="engine":
                prior=read(BASE/"engine-neighbor-updates/switch/live/transfer/issued.json")["predictions"]
                if any(abs(v-prior["current"][i])>1e-8 for i,v in preds["current"].items()):
                    raise ValueError("常规预测未复现")
                preds["five-devices"]={i:prior["source-weight-from-one"][i] for i in preds["current"]}
            path=owner/"transfer"/suite/"issued.json"
            atomic_json(path,{"inputs":digest(windows),"trees":source["trees"],"source_checkpoints":versions,"predictions":preds})
            frozen=read(path)["predictions"]
            answers=truth()
            table=summary(task,answers,{"parent":frozen["current"],**frozen},groups)
            for arm,v in frozen.items():
                table[arm]["rounds"]={r:metric(task,answers,{i:y for i,y in v.items() if units[i]==r}) for r in windows}
                table[arm]["range"]=[min(v.values()),max(v.values())]
            atomic_json(path.with_name("scores.json"),{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table})
            print(suite,{k:(round(table[k]["score"],2),round(table[k]["worst_group"],2)) for k in preds},flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("engine","bearing"))
    comparison=parser.add_mutually_exclusive_group()
    comparison.add_argument("--routing",action="store_true")
    comparison.add_argument("--phase",action="store_true")
    args=parser.parse_args()
    evaluate(args.task,args.routing,args.phase)
