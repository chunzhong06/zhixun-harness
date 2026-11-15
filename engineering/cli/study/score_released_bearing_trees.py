"""按真实采用轮次冻结并移用轴承树，核对原流、另一批未见波形和异台架的预测。"""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_error_slice import BASE,ROOT,ModelArtifacts,_trainer,read,reference_folders,released_checkpoints
from compare_group_objectives import atomic_json,digest,metric,official,summary
from compare_residual_trees import OUT
from score_unseen_bearing_windows import window_inputs,window_answers


def tree_prediction(tree,x):
    # sklearn 决策树在预测入口将特征转为 float32，阈值保留 double。
    values=np.asarray(x,dtype=np.float32)
    nodes=np.zeros(len(values),dtype=int)
    left,right,features=(np.asarray(tree[k],dtype=int) for k in ("children_left","children_right","feature"))
    thresholds=np.asarray(tree["threshold"])
    while np.any(left[nodes]>=0):
        rows=np.flatnonzero(left[nodes]>=0)
        n=nodes[rows]
        use_left=values[rows,features[n]].astype(np.float64)<=thresholds[n]
        nodes[rows]=np.where(use_left,left[n],right[n])
    return np.asarray(tree["value"],dtype=float)[nodes,0,0]


def predict_tree_path(model,source,arm,rounds):
    events=source["events"][arm]
    history=source["trees"][arm]
    predictions={}
    for r,items in sorted(rounds.items()):
        ids,x=zip(*items)
        x=np.asarray(x)
        values=model.predict(x)
        used=sum(e["adopted"] for e in events if e["round"]<=r+1)
        for tree in history[:used]:
            values=np.maximum(0,values+.5*tree_prediction(tree,x))
        predictions.update(zip(ids,map(float,values),strict=True))
    return predictions


def main():
    cohort="pronostia"
    source=read(OUT/cohort/"issued.json")
    package,checksum,_=official("bearing")
    if source["source"]!=checksum or source["initial_model"]!=sha256(package.checkpoint).hexdigest():
        raise ValueError("源树或起点不同")
    fixed,paper=reference_folders(cohort,"mature")
    paths={"old":fixed,"mature":paper,"simple":fixed.parent/"simple","bias":BASE/"readout-live"/cohort}
    rounds,metadata,ends=window_inputs(8)
    prev,prev_meta,prev_ends=window_inputs(16)
    fixed_data=read(fixed/"state.json")["samples"]
    original={r:[(i,s["values"]) for i,s in fixed_data.items() if s["available_at"]==r and s["role"]=="serve"] for r in range(10)}
    raw=read(ROOT/"data/public/kaist-bearing-v6/hourly-windows.json")["rows"]
    kaist={9:[("KAIST/"+s["file"],s["features"]) for s in raw if "features" in s]}
    suites={"unseen8":rounds,"previous16":prev,"kaist":kaist}
    pools={name:{} for name in suites}
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z,_trainer(z,Path(tmp),"B") as trainer:
        p=Path(tmp)/"model.npz"
        p.write_bytes(package.checkpoint)
        root=trainer.Model.load(p)
        for arm in ("stump","tree3"):
            reproduced=predict_tree_path(root,source,arm,original)
            original_predictions=source["predictions"][arm]
            if reproduced.keys()!=original_predictions.keys() or any(abs(v-original_predictions[k])>1e-8 for k,v in reproduced.items()):
                raise ValueError("保存的树路径未复现真实开发预测")
            for name,batches in suites.items():
                pools[name][arm]=predict_tree_path(root,source,arm,batches)
        weights={}
        for arm,folder in paths.items():
            releases=released_checkpoints(folder,arm,True)
            store=ModelArtifacts(folder/"models")
            weights[arm]={}
            for name,batches in suites.items():
                preds={}
                for r,records in sorted(batches.items()):
                    head,blob=store.read(releases[arm if r==9 else f"{arm}@{r+1:02}"])
                    weights[arm][r]=head["sha256"]
                    p.write_bytes(blob)
                    ids,x=zip(*records)
                    pred=trainer.Model.load(p).predict(np.array(x))
                    preds.update(zip(ids,map(float,pred),strict=True))
                pools[name][arm]=preds
    out=OUT/cohort/"transfer"
    issued=out/"issued.json"
    atomic_json(issued,{"source_sha256":sha256((OUT/cohort/"issued.json").read_bytes()).hexdigest(),"predictions":pools,
                        "input_digests":{n:digest(r) for n,r in suites.items()},"reference_weights":weights})
    pools=read(issued)["predictions"]
    answers={"unseen8":window_answers(metadata,ends),"previous16":window_answers(prev_meta,prev_ends)}
    end=datetime.fromisoformat(raw[-1]["time"])
    answers["kaist"]={"KAIST/"+s["file"]:(end-datetime.fromisoformat(s["time"])).total_seconds()/60 for s in raw if "features" in s}
    stats={}
    for name,streams in pools.items():
        groups={i:("KAIST" if name=="kaist" else i.split("/")[1]) for i in answers[name]}
        stats[name]=summary("bearing",answers[name],{"parent":streams["old"],**streams},groups)
        for arm,s in stats[name].items():
            pred=streams["old" if arm=="parent" else arm]
            s["rounds"]={r:metric("bearing",answers[name],{i:pred[i] for i,_ in rows}) for r,rows in suites[name].items()}
        print(name,{a:round(v["score"],2) for a,v in stats[name].items()},flush=True)
    atomic_json(out/"scores.json",{"issued_sha256":sha256(issued.read_bytes()).hexdigest(),"suites":stats})


if __name__=="__main__":
    main()
