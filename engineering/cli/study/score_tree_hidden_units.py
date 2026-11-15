"""把源训练树的分裂位置迁入赛方隐藏层；只从源预测残差决定保留哪些位置。"""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT,frozen,tree_values
from compare_group_objectives import ROOT,_trainer,atomic_json,official
from score_life_training_weights import cohorts,score


def choose_borders(z,target,tree,model,width):
    feature,threshold,positions,orders=[],[],[],{}
    for k,borders in tree.get_borders().items():
        k=int(k)
        if not borders:
            continue
        order=np.argsort(z[:,k],kind="stable")
        sorted_z=z[order,k]
        start=len(feature)
        for border in borders:
            t=(border-model.mean[k])/model.scale[k]
            p=int(np.searchsorted(sorted_z,t,side="right"))
            if min(p,len(z)-p)<.01*len(z) or not -10<t<10:
                continue
            feature.append(k)
            threshold.append(float(t))
            positions.append(p)
        if len(feature)>start:
            orders[k]=(order,slice(start,len(feature)))
    f=np.asarray(feature)
    cut=np.asarray(threshold)
    left=np.asarray(positions)
    right=len(z)-left
    used=np.zeros(len(left),bool)
    residual=target-target.mean()
    selected=[]
    for _ in range(min(width,len(left))):
        sums=np.empty(len(left))
        total=residual.sum()
        for order,span in orders.values():
            sums[span]=np.cumsum(residual[order])[left[span]-1]
        # 两个叶子的误差下降量，已选择切分不可再次分配到隐藏层。
        gains=sums*sums/left+(total-sums)**2/right-total**2/len(z)
        gains[used]=-np.inf
        best=int(np.argmax(gains))
        selected.append(best)
        used[best]=True
        residual=residual-np.where(z[:,f[best]]<=cut[best],sums[best]/left[best],(total-sums[best])/right[best])
    rng=np.random.default_rng(17)
    random=rng.choice(len(left),size=len(selected),replace=False)
    return [[(int(f[j]),float(cut[j])) for j in indexes] for indexes in (selected,random)]


def prepare(task,pkg,trainer,root,teacher):
    x=np.asarray([r.values for r in pkg.train])
    y=np.asarray([r.label for r in pkg.train])
    if any(r.role!="fit" for r in pkg.train):
        raise ValueError("隐藏层选择仅使用源训练观测")
    source=root/"original.npz"
    source.write_bytes(pkg.checkpoint)
    model=trainer.Model.load(source)
    z=model.normalized(x)
    h,_=model.forward(z)
    order=np.argsort(h.std(axis=0)*abs(model.params["w2"][:,0]),kind="stable")
    target=tree_values(teacher,task,x)
    start=perf_counter()
    fitted,random=choose_borders(z,target,teacher.named_steps["predictor"],model,len(order))
    seconds=perf_counter()-start
    if len(fitted)!=len(order):
        raise ValueError("源数据的有效树阈值不够")
    options={"frozen-teacher":([],target),"half-teacher":(fitted[:len(order)//2],target),
             "whole-teacher":(fitted,target),"random-half":(random[:len(order)//2],target),
             "half-truth":(fitted[:len(order)//2],y),"frozen-truth":([],y)}
    models={}
    details={}
    for name,(splits,objective) in options.items():
        tic=perf_counter()
        model=trainer.Model.load(source)
        for unit,(sensor,border) in zip(order,splits):
            model.params["w1"][:,unit]=0.
            model.params["w1"][sensor,unit]=4.
            model.params["b1"][unit]=-4.*border
        a=np.column_stack((model.forward(z)[0],np.ones(len(z))))
        gram=a.T@a+.001*len(a)*np.eye(a.shape[1])
        normalized=(objective-model.meta["target_mean"])/model.meta["target_scale"]
        w=np.linalg.solve(gram,a.T@normalized)
        model.params["w2"][:,0]=w[:-1]
        model.params["b2"][0]=w[-1]
        model.meta["steps"]+=1
        model.meta["epochs"]+=1
        path=root/(name+".npz")
        model.save(path)
        models[name]=trainer.Model.load(path)
        pred=models[name].predict(x)
        details[name]={"sha256":sha256(path.read_bytes()).hexdigest(),"hidden_units_replaced":len(splits),
            "seconds":perf_counter()-tic,"source_teacher_rmse":float(np.sqrt(np.mean((target-pred)**2))),
            "source_truth_rmse":float(np.sqrt(np.mean((y-pred)**2)))}
    atomic_json(root/"training.json",{"arms":details,"ranking_seconds":seconds,"chosen_borders":fitted,"random_borders":random,
                                     "source_rows":len(x),"parent":pkg.checkpoint_sha})
    return models


def main(task):
    pkg,_,_=official(task)
    trees,_=frozen(task)
    teacher=trees["catboost"]
    out=OUT/"tree-hidden"
    models_dir=out/task/"models"
    models_dir.mkdir(parents=True,exist_ok=True)
    archive,=(ROOT/"data/official"/task).glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B" if task=="bearing" else "C") as trainer:
        models=prepare(task,pkg,trainer,models_dir,teacher)
        def predict(x):
            return {"direct-teacher":tree_values(teacher,task,x),**{k:m.predict(x) for k,m in models.items()}}
        results={c[0]:score(task,c,predict,out,"direct-teacher") for c in cohorts(task,pkg)}
    atomic_json(out/task/"scores.json",results)
    for name,t in results.items():
        print(name,{k:round(t[k]["score"],2) for k in ("current","teacher-ridge","frozen-teacher","half-teacher","whole-teacher","random-half","half-truth")})


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("bearing","engine"))
    main(parser.parse_args().task)
