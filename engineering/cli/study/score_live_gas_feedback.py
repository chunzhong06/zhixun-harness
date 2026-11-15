"""核对逐批真实入口回执，比较规则在额外抽样种子、固定一次和多次选择上的收益。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,BASE,ROOT,REGISTERED_SHA256,read_gas_batches,read,atomic_json,digest,official
from harness.models.feedback_routing import FeedbackRouter,apply_choices
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.gas_views import pair_model
from harness.models.official_rul_cpu import _trainer
from harness.models.model_artifacts import ModelArtifacts


def early_router(names,fit,batches):
    folder=BASE/"selected/gas"
    r=read(folder/"run.json")["rounds"][3]
    _,blob=ModelArtifacts(folder/"models").read(r["model_artifact"])
    source=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip"
    observed=[s for b in batches for s in b.samples if s.role in ("explore","adopt") and s.available_at==b.ordinal]
    if len({s.sample_id for s in observed})!=len(observed):
        raise ValueError("来源反馈重复")
    with TemporaryDirectory() as tmp,ZipFile(source) as pack,_trainer(pack,Path(tmp),"A") as trainer:
        p=Path(tmp)/"current.npz"
        p.write_bytes(blob)
        model=pair_model(trainer.Model).load(p)
        candidates=GasFeedbackBank(fit).predict(model,np.asarray([s.values for s in observed]))
    router=FeedbackRouter(names,6)
    router.update([s.label for s in observed],candidates)
    return router


def main():
    live=OUT/"live"
    source=read(OUT/"fixed.json")
    keys=[i for group in source["keys"] for i in group]
    blocks=np.split(np.arange(len(keys)),np.cumsum([len(g) for g in source["keys"]])[:-1])
    batches=[b for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256) if b.batch_number>=3]
    if [r.sample_id for b in batches for r in b.records]!=keys:
        raise ValueError("目标来源改变")
    y=np.asarray([r.label for b in batches for r in b.records])
    pred={k:np.asarray(source["predictions"][k]) for k in GasFeedbackBank.names}
    pkg,_,public=official("gas")
    fitted={s.sample_id for s in public[0].samples if s.role=="fit"}
    prepared=early_router(list(pred),[s for s in pkg.train if s.sample_id in fitted],public)
    earlier=read(OUT/"issued.json")["predictions"]
    results={}
    for mode in ("local","global"):
        results["official-feedback-fixed-"+mode]=apply_choices(pred,prepared.choices(mode))
    vote=np.stack([sum(a==g for a in pred.values()) for g in range(1,7)],axis=1)
    results["light-vote"]=np.where(vote[np.arange(len(keys)),pred["current"]-1]==vote.max(1),pred["current"],vote.argmax(1)+1)
    results.update(pred)
    for seed in (17,47,89,*range(101,111)):
        live_router=FeedbackRouter(list(pred),6)
        frozen=None
        for j,b in enumerate(blocks):
            parts={k:p[b] for k,p in pred.items()}
            for mode in ("local","global","relabel"):
                for suffix,router in (("rolling",live_router),("first",frozen or live_router)):
                    key=f"{suffix}-{mode}-{seed}"
                    results.setdefault(key,np.zeros(len(y),int))[b]=apply_choices(parts,router.choices(mode))
            ix=np.asarray(sorted(b,key=lambda i:digest([seed,keys[i]]))[:32],int)
            live_router.update(y[ix],{k:v[ix] for k,v in pred.items()})
            if j==0:
                frozen=FeedbackRouter(list(pred),6,live_router.counts.copy())
    verification={}
    for code,expected in [(str(s),results[f"rolling-local-{s}"]) for s in (17,47,89)]+[("zero",pred["current"])]:
        folder=live/code
        report=read(folder/"run.json")
        stream=[]
        returned=0
        for group,event in zip(source["keys"],report["accepted"],strict=True):
            request=event["request"]
            rec=read(folder/"accepted"/(request+".json"))
            p=rec["payload"]
            if (p["feedback_seen"]!=returned or rec["request_id"]!=request or digest([report["identity"],p])!=request
                    or set(p["predictions"])!=set(group) or event["returned"]!=sorted(group,key=lambda i:digest([report["seed"],i]))[:report["feedback_per_batch"]]):
                raise ValueError("交接身份、反馈时机或样本身份不符")
            stream.extend(p["predictions"][i] for i in group)
            returned+=len(event["returned"])
        if not np.array_equal(stream,expected):
            raise ValueError("现场预测不同于轻量版独立重算")
        verification[code]={"full_bank_differences":0 if code=="zero" else int(np.count_nonzero(np.array(stream)!=earlier[f"cumulative-local-{code}"])),
                            "seen_at_final_issue":report["accepted"][-1]["feedback_seen"],"setup_seconds":report["setup_seconds"],
                            "batch_seconds":sum(v["seconds"] for v in report["accepted"]),"run_sha256":sha256((folder/"run.json").read_bytes()).hexdigest()}
    path=live/"comparison.json"
    atomic_json(path,{"fixed_sha256":sha256((OUT/"fixed.json").read_bytes()).hexdigest(),"verification":verification,
                      "predictions":{k:v.tolist() for k,v in results.items()}})
    table={k:scores(y,np.asarray(v),blocks,6) for k,v in read(path)["predictions"].items()}
    atomic_json(live/"scores.json",{"comparison_sha256":sha256(path.read_bytes()).hexdigest(),"verification":verification,"arms":table})
    print(verification)
    for k in ("official-feedback-fixed-local","official-feedback-fixed-global","light-vote","current","first-local-17","first-local-47","first-local-89"):
        print(k,round(table[k]["equal_batches"],5))
    for prefix in ("rolling-local","first-local","rolling-relabel","rolling-global"):
        v=[table[f"{prefix}-{s}"]["equal_batches"] for s in range(101,111)]
        print(prefix,"101..110",np.round([min(v),np.mean(v),max(v)],5))


if __name__=="__main__":
    main()
