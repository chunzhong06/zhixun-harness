"""以完整实验重复留出，在同步参考可用时比较目标行单独训练、直接标准化和现场选读法。"""
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.contracts import digest
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json

SOURCE = ROOT/"data/public/dsps-transfer"
DEST = ROOT/"results/experiments/local-defaults/whole-maintenance/dsps-reference"
FIXED = ("原器件直接用","目标器件逻辑回归","目标器件核分类","跨器件联合分类","逐通道标准化","直接标准化","分段直接标准化")


def fit_models(x, y):
    old, spare = x[:,0,:], x[:,-1,:]
    source_model = make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000)).fit(old,y)
    target_model = make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000)).fit(spare,y)
    target_svm = make_pipeline(StandardScaler(),SVC(C=4,gamma="scale")).fit(spare,y)
    pooled = make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000)).fit(
        np.vstack((old,spare)),np.tile(y,2))
    sc = StandardScaler().fit(spare)
    sc_old = StandardScaler().fit(old)
    z, zo = sc.transform(spare),sc_old.transform(old)
    full = Ridge(alpha=1.).fit(z,zo)
    diagonal = (z*zo).sum(axis=0)/((z*z).sum(axis=0)+1)
    blocks = [Ridge(alpha=1.).fit(z[:,k:k+10],zo[:,k:k+10]) for k in range(0,80,10)]
    def predict(new):
        zn = sc.transform(new)
        mapped = (zn*diagonal,full.predict(zn),
                  np.column_stack([m.predict(zn[:,i*10:(i+1)*10]) for i,m in enumerate(blocks)]))
        return np.column_stack([source_model.predict(new),target_model.predict(new),target_svm.predict(new),
                                pooled.predict(new),*(source_model.predict(sc_old.inverse_transform(v)) for v in mapped)])
    return predict,sc


def cross_fit(x, y, repeats):
    """用其余安装批的真实留出预测学习选法；不使用本批任何标签。"""
    predictions = np.empty((len(y),len(FIXED)),dtype=int)
    for rep in np.unique(repeats):
        take = repeats==rep
        model,_ = fit_models(x[~take],y[~take])
        predictions[take] = model(x[take,-1,:])
    return predictions


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("已有安装开发结果")
        meta = json.loads((SOURCE/"source.json").read_bytes())
        archive = SOURCE/"development.npz"
        if sha256(archive.read_bytes()).hexdigest()!=meta["特征SHA256"]:
            raise ValueError("开发特征身份变化")
        with np.load(archive,allow_pickle=False) as data:
            x,y,groups,ids=(data[k] for k in ("x","y","repetition","ids"))
        if x.shape!=(54,12,80) or set(groups)!={1,2,3} or len(set(ids))!=len(ids):
            raise ValueError("开发暴露或同时采样的阵列排数不对")
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        all_truth,all_output,requests=[],{},[]
        for rep in np.unique(groups):
            mask = groups==rep
            source, labels = x[~mask],y[~mask]
            clf,scale = fit_models(source,labels)
            held = cross_fit(source,labels,groups[~mask])
            correct = held==labels[:,None]
            # 直接目标器件读法是默认平分优先项；固定对手也能使用相同的安装标签。
            priority = np.array([1,2,3,5,6,4,0])
            score = correct.mean(axis=0)
            global_choice = priority[score[priority].argmax()]
            target = x[mask,-1,:]
            pred = clf(target)
            z,zref = scale.transform(target),scale.transform(source[:,-1,:])
            ds = ((z[:,None,:]-zref[None,:,:])**2).mean(axis=2)
            ix = np.argsort(ds,axis=1,kind="stable")[:,:8]
            w = 1/(1+np.take_along_axis(ds,ix,axis=1))
            support = np.einsum("nk,nkm->nm",w,correct[ix]) + 4*score
            local = priority[np.argmax(support[:,priority],axis=1)]
            out = {k:pred[:,i].tolist() for i,k in enumerate(FIXED)}
            out["源端留整组选定"] = pred[:,global_choice].tolist()
            out["当前读数邻近选择"] = pred[np.arange(len(pred)),local].tolist()
            payload={"ids":ids[mask].tolist(),"实际分类":out,"训练重复":np.unique(groups[~mask]).tolist(),
                     "源端选定":FIXED[global_choice],"现场选用":[FIXED[i] for i in local]}
            key=digest([meta["特征SHA256"],int(rep),payload])
            if not receiver.submit(key,payload)["accepted"]:
                raise ValueError("安装候选未交接")
            requests.append(key)
        for rep,key in zip(np.unique(groups),requests,strict=True):
            p=json.loads((receiver.output/(key+".json")).read_bytes())["payload"]
            if digest([meta["特征SHA256"],int(rep),p])!=key or p["ids"]!=ids[groups==rep].tolist():
                raise ValueError("服务曝光身份不同")
            all_truth.extend(y[groups==rep].tolist())
            for k,v in p["实际分类"].items():
                all_output.setdefault(k,[]).extend(v)
        table={k:{"F1":float(f1_score(all_truth,v,labels=[1,2,3],average="macro")),
                  "最差召回":float(recall_score(all_truth,v,labels=[1,2,3],average=None).min()),
                  "正确":int(np.sum(np.array(v)==all_truth))} for k,v in all_output.items()}
        fixed=max(FIXED,key=lambda k:table[k]["F1"])
        best=max(("当前读数邻近选择","源端留整组选定"),key=lambda k:table[k]["F1"])
        decision={"最强固定":fixed,"候选":best,"对固定增量":table[best]["F1"]-table[fixed]["F1"],
                  "进入独立验证":table[best]["F1"]>=table[fixed]["F1"]+.01 and table[best]["最差召回"]>=table[fixed]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"requests":requests,"scores":table,"decision":decision,"development_only":True})
        print(json.dumps({"开发逐条成绩":table,"采用判定":decision},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
