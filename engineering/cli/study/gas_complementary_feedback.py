"""从冻结的源批训练分类器出发，在按时回标的旧批上比较不同候选选择规则。"""
from hashlib import sha256
from time import perf_counter

import numpy as np

from compare_fixed_tree_references import frozen,tree_values
from compare_gas_replicates import MODELS
from compare_group_objectives import BASE,ROOT,atomic_json,digest,metric,official,read
from harness.data.gas_input import REGISTERED_SHA256,read_gas_batches
from harness.models.gas_views import represent
from harness.models.gas_feedback_bank import fit_discriminant,discriminate


OUT=BASE/"gas-complementary-feedback"


def lda(bank,query,y):
    return discriminate(fit_discriminant(bank,y),query)


def checked(folder):
    path=folder/"issued.json"
    if sha256(path.read_bytes()).hexdigest()!=read(folder/"scores.json")["issued_sha256"]:
        raise ValueError("冻结预测记录身份变更")
    return read(path)


def build():
    pkg,checksum,batches=official("gas")
    fit_ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    fit=[r for r in pkg.train if r.sample_id in fit_ids]
    y=np.array([r.label for r in fit])
    train=np.asarray([r.values for r in fit])
    original=checked(BASE/"gas-concentration")["predictions"]
    svm=checked(BASE/"gas-sensor-ranks")["outputs"]
    batches=[b for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256) if b.batch_number>=3]
    groups=[[r.sample_id for r in b.records] for b in batches]
    keys=[i for group in groups for i in group]
    x=np.asarray([r.values for b in batches for r in b.records])
    if len(keys)!=len(set(keys)) or len(fit_ids)!=len(fit) or len(fit)!=445:
        raise ValueError("新候选训练或服务清单不一致")
    outputs={k:np.array([original[name][i] for i in keys]) for k,name in (("current","project"),("paper","paper"))}
    timings={}
    tic=perf_counter()
    scale=np.maximum(np.median(np.abs(train),axis=0),1e-9)
    scaled=lambda v: np.sign(v)*np.log1p(np.abs(v)/scale)
    bank,query=scaled(train),scaled(x)
    timings["transform"]=perf_counter()-tic
    columns=np.arange(128).reshape(16,8)
    views={"all":columns.ravel(),**{f"type-{j+1}":columns[indices].ravel() for j,indices in enumerate(MODELS)}}
    for name,col in views.items():
        tic=perf_counter()
        outputs[name]=lda(bank[:,col],query[:,col],y)
        timings[name]=perf_counter()-tic
    tic=perf_counter()
    outputs["relative"]=lda(represent(train,"shape",None,None),represent(x,"shape",None,None),y)
    timings["relative"]=perf_counter()-tic
    trees,signatures=frozen("gas")
    fit_hash=digest([r.sample_id for r in fit])
    for arm,model in trees.items():
        if signatures[arm]["fit"]!=fit_hash or signatures[arm]["package"]!=checksum:
            raise ValueError("核验的树模型训练来源不同")
        tic=perf_counter()
        outputs[arm]=tree_values(model,"gas",x).argmax(axis=1)+1
        timings[arm]=perf_counter()-tic
    for k in ("shape-full","shape-gated"):
        outputs[k]=np.array([svm[k]["candidate"][i] for i in keys])
    path=OUT/"fixed.json"
    atomic_json(path,{"package":checksum,"source":REGISTERED_SHA256,"fit":fit_hash,
                      "keys":groups,"predictions":{k:v.tolist() for k,v in outputs.items()},
                      "tree_sha256":{k:v["sha256"] for k,v in signatures.items()}})
    atomic_json(OUT/"cost.json",{"fixed_sha256":sha256(path.read_bytes()).hexdigest(),"fit_or_predict_seconds":timings})
    return batches


def decisions(pred,truth,indices,mode):
    names=list(pred)
    if not len(indices):
        return ["current"]*6
    y=truth[indices]
    base=pred["current"][indices]
    if mode=="global":
        k=max(names,key=lambda a:metric("gas",dict(enumerate(y)),dict(enumerate(pred[a][indices]))))
        return [k]*6
    choice=[]
    for label in range(1,7):
        used="current"
        best=1
        for k,p in pred.items():
            q=p[indices]
            mask=(base==label)&(base!=q)
            if mask.sum()<4:
                continue
            improvement=int((q[mask]==y[mask]).sum())-int((base[mask]==y[mask]).sum())
            if improvement>best:
                used,best=k,improvement
        choice.append(used)
    return choice


def assess(values,truth,blocks):
    def score(p,y):
        return metric("gas",dict(enumerate(y)),dict(enumerate(p)))
    by_batch=[score(values[ix],truth[ix]) for ix in blocks]
    recall=[float(np.mean(values[truth==g]==g)) for g in range(1,7)]
    return {"pooled":score(values,truth),"equal_batches":float(np.mean(by_batch)),
            "worst_batch":min(by_batch),"recall":recall,"batches":by_batch}


def evaluate():
    batches=build()
    fixed=read(OUT/"fixed.json")
    names=list(fixed["predictions"])
    pred={k:np.array(v) for k,v in fixed["predictions"].items()}
    keys=[i for b in fixed["keys"] for i in b]
    truth=np.array([r.label for b in batches for r in b.records])
    blocks=np.split(np.arange(len(keys)),np.cumsum([len(b) for b in fixed["keys"]])[:-1])
    base=pred["current"]
    wrong=base!=truth
    repaired=np.any([p==truth for p in pred.values()],axis=0)&wrong
    stats={k:assess(p,truth,blocks) for k,p in pred.items()}
    oracle=[max(names,key=lambda a:stats[a]["batches"][j]) for j in range(len(blocks))]
    diagnostic={"wrong":int(wrong.sum()),"recoverable":int(repaired.sum()),"oracle_choices":oracle,
                "oracle_batch_mean":float(np.mean([stats[a]["batches"][j] for j,a in enumerate(oracle)])),
                "recoverable_by_class":[int(np.sum(repaired&(truth==g))) for g in range(1,7)],
                "errors_by_class":[int(np.sum(wrong&(truth==g))) for g in range(1,7)]}
    traces, streams={},{}
    if repaired.sum()>.10*wrong.sum():
        for seed in (17,47,89):
            sampled=[np.array(sorted(b,key=lambda j:digest([seed,keys[j]]))[:32]) for b in blocks]
            for age in ("last","cumulative"):
                for mode in ("global","local"):
                    arm=f"{age}-{mode}-{seed}"
                    forecast=base.copy()
                    records=[]
                    for j,block in enumerate(blocks):
                        previous=sampled[max(0,j-1) if age=="last" else 0:j]
                        visible=np.concatenate(previous) if previous else np.array([],int)
                        use=decisions(pred,truth,visible,mode)
                        for g,k in enumerate(use,1):
                            ix=block[base[block]==g]
                            forecast[ix]=pred[k][ix]
                        records.append({"batch":j+3,"choice":use,"feedback_seen":len(visible),
                                        "returned_after_service":[keys[i] for i in sampled[j]]})
                    traces[arm],streams[arm]=records,forecast.tolist()
    path=OUT/"issued.json"
    atomic_json(path,{"fixed_sha256":sha256((OUT/"fixed.json").read_bytes()).hexdigest(),"traces":traces,"predictions":streams})
    for k,v in read(path)["predictions"].items():
        stats[k]=assess(np.array(v),truth,blocks)
    atomic_json(OUT/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"fixed":names,
                                  "diagnostic":diagnostic,"arms":stats})
    print(diagnostic)
    for k,s in stats.items():
        print(k,*(round(s[v],5) for v in ("pooled","equal_batches","worst_batch")),round(s["recall"][3],4))


if __name__=="__main__":
    evaluate()
