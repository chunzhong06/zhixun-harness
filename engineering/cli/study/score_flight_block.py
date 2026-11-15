"""由原始航班重新生成不同起点和块均值序列，同时给成熟双向序列模型重训该新读法。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import h5py
import joblib
import numpy as np
from score_flight_component import DEST as FEATURES,SPLIT,BASE,other_masks,issue,receipt,file_hash,LocalStore,atomic_json,fixed_mix,PAPER,SENSORS,SETTINGS
from score_flight_trajectory_split import dataset_path,batch_sequence,call_paper
from score_flight_inverse import DEST as PRIOR
from harness.models.flight_block import samples,component_view,BlockReaders,PRIMARY,MEAN,AUGMENT

DEST=BASE/"flight-block"
NEW_PAPER="整趟论文按原始块均值重训"
MIX="多起点与双序列论文固定组合"


def extract(s,part,units,extractor):
    keys=[];xx=[];seq=[];ys=[]
    with h5py.File(dataset_path(s)) as f:
        names=lambda k:tuple(v.decode().strip() for v in f[k][:].ravel())
        if names("X_s_var")!=SENSORS or names("W_var")!=SETTINGS:raise ValueError("测站字段顺序变化")
        aux=f["A_"+part][:,:2].astype(int)
        bounds=np.r_[0,np.flatnonzero(np.any(np.diff(aux,axis=0),axis=1))+1,len(aux)]
        for start,end in zip(bounds[:-1],bounds[1:],strict=True):
            uid=tuple(map(int,aux[start]))
            if units is not None and uid[0] not in units:continue
            raw=np.column_stack((f["X_s_"+part][start:end],f["W_"+part][start:end]))
            rows=samples(raw)
            keys.append(uid);xx.append(np.stack([component_view(extractor,r) for r in rows]))
            seq.append(batch_sequence(rows[-1]))
            ys.append(float(f["Y_"+part][start,0]))
    order=sorted(range(len(keys)),key=keys.__getitem__)
    return [keys[i] for i in order],np.stack(xx)[order],np.stack(seq)[order],np.array(ys)[order]


def main():
    with LocalStore(DEST/"lock").exclusive(),TemporaryDirectory(dir=DEST) as temp:
        if (DEST/"scores.json").exists():raise ValueError("相同读法已有结果")
        split=json.loads((SPLIT/"scores.json").read_bytes());prev=json.loads((PRIOR/"scores.json").read_bytes())
        meta=json.loads((FEATURES/"scores.json").read_bytes())
        if file_hash(FEATURES/"release.joblib")!=meta["发布权重SHA256"] or file_hash(FEATURES/"profiles.npz")!=meta["缓存SHA256"]:
            raise ValueError("部件模型或分组身份不同")
        trained=joblib.load(FEATURES/"release.joblib");extractor=trained["extractor"]
        X=[];ys=[];sequence=[];early=[];valid=[]
        for s in sorted(split["训练设备"],key=lambda k:(k=="DS02",k)):
            units=split["训练设备"][s]
            if not units:continue
            ids,x,seq,y=extract(s,"dev",set(units),extractor)
            X.append(x);ys.append(y);sequence.append(seq)
            held=set(sorted(units)[::5])
            starts={u:min(c for v,c in ids if u==v) for u in units}
            early.extend(c-starts[u]<8 for u,c in ids);valid.extend(u in held for u,_ in ids)
        model=BlockReaders().fit(np.concatenate(X),np.concatenate(ys),trained["model"].readouts[-1])
        folder=Path(temp);pp=DEST/"paper.pt"
        np.savez(folder/"train.npz",x=np.concatenate(sequence),y=np.concatenate(ys),early=early,valid=valid)
        call_paper(["train",str(folder/"train.npz"),str(pp)])
        groups={};data={};sequence=[]
        with np.load(FEATURES/"profiles.npz") as z:
            for s,r in prev["sites"].items():
                ids,x,seq,y=extract(s,"test",None,extractor)
                old=receipt(PRIOR,s,r)
                if old["ids"]!=[list(k) for k in ids] or old["ids"]!=z[s+"_ids"].tolist():raise ValueError("原始序列错位")
                out=model.predict(x)
                np.testing.assert_allclose(out["第一个取样起点"],old["predictions"]["合用原量与部件余差"],atol=1e-8,rtol=0)
                data[s]=(ids,y,{**old["predictions"],**out});groups[s]=z[s+"_group"]
                sequence.append(seq)
        np.save(folder/"test.npy",np.concatenate(sequence))
        call_paper(["predict",str(folder/"test.npy"),str(folder/"output.npy"),str(pp)])
        pseq=np.load(folder/"output.npy");i=0;bank={}
        fixed_new=[MEAN,AUGMENT,"第一个取样起点","原模型在四个起点等权","原模型直接用块均值","同三工具固定平均","联合部件与块均值固定各半"]
        for s,(ids,y,p) in data.items():
            p[NEW_PAPER]=pseq[i:i+len(y)];i+=len(y)
            bank[s]=np.column_stack([p[k] for k in [*fixed_new,PAPER,NEW_PAPER]])
        scores={}
        for s,(ids,y,p) in data.items():
            masks=other_masks(groups,s)
            b=np.vstack([bank[t][m] for t,m in masks.items()]);labels=np.concatenate([data[t][1][m] for t,m in masks.items()])
            teams=np.concatenate([np.repeat(t,m.sum()) for t,m in masks.items()])
            p[MIX]=bank[s]@fixed_mix(b,labels,teams)
            scores[s]=issue(DEST,s,ids,p,y)
        means={k:float(np.mean([r["scores"][k]["rmse"] for r in scores.values()])) for k in p}
        fixed=min([prev["decision"]["最强固定"],MIX,*fixed_new],key=means.get)
        paper=min([prev["decision"]["同工具论文"],MIX,NEW_PAPER,PAPER],key=means.get)
        gain=1-means[PRIMARY]/min(means[fixed],means[paper])
        worst=max(r["scores"][PRIMARY]["worst"] for r in scores.values())
        old_worst=max(r["scores"]["第一个取样起点"]["worst"] for r in scores.values())
        d={"现场":PRIMARY,"最强固定":fixed,"同工具论文":paper,"现场误差":means[PRIMARY],"固定误差":means[fixed],"论文误差":means[paper],
           "降幅":gain,"现场最差":worst,"改前最差":old_worst,"进独立":gain>=.01 and worst<=1.01*old_worst}
        weights=DEST/"release.joblib";joblib.dump({"extractor":extractor,"readers":model},weights)
        atomic_json(DEST/"scores.json",{"sites":scores,"means":means,"decision":d,"发布权重SHA256":file_hash(weights),"论文权重SHA256":file_hash(pp)})
        print(json.dumps({"开发交付":d,"块均值重训":means[MEAN],"多起点训练":means[AUGMENT],"重训序列":means[NEW_PAPER]},ensure_ascii=False))


if __name__=="__main__":main()
