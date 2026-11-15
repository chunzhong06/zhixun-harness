"""还原官方十轮现场权重，检验启用投票的时机，并与后期仪器回放及真实入口互核。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from check_feedback_controls import scores
from compare_calibration_expiry import source_window
from compare_fixed_tree_references import paths,delivered
from gas_complementary_feedback import BASE,OUT,ROOT,atomic_json,digest,official,read
from harness.models.feedback_routing import FeedbackRouter,apply_choices
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.gas_vote_guard import GuardedVote
from harness.models.gas_views import pair_model
from harness.models.model_artifacts import ModelArtifacts
from harness.models.official_rul_cpu import _trainer
from score_new_electronic_nose import load_rows
from gas_complementary_feedback import REGISTERED_SHA256,read_gas_batches


SEEDS=(17,47,89,*range(101,111))


def public_stream():
    folder,refs=paths("gas","gas")
    samples=read(folder/"state.json")["samples"]
    pkg,sha,batches=official("gas")
    fit_ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    bank=GasFeedbackBank([r for r in pkg.train if r.sample_id in fit_ids])
    archive=ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip"
    if sha256(archive.read_bytes()).hexdigest()!=sha:
        raise ValueError("公开包身份已变")
    store=ModelArtifacts(folder/"models")
    stream=[]
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"A") as trainer:
        path=Path(tmp)/"public.npz"
        for j,r in enumerate(read(folder/"run.json")["rounds"]):
            _,blob=store.read(r["model_artifact"])
            path.write_bytes(blob)
            model=pair_model(trainer.Model).load(path)
            explore,adopt,serve=source_window(samples,j)
            feedback=explore+adopt
            def predict(rows):
                return bank.predict(model,np.asarray([s["values"] for s in rows]))
            p=predict(serve)
            ids=[s["sample_id"] for s in serve]
            if dict(zip(ids,p["current"].tolist(),strict=True))!=read(folder/r["submitted"])["payload"]["predictions"]:
                raise ValueError("当轮原模型交付未复现")
            if any(s["label"] is not None for s in serve) or set(ids)&{s["sample_id"] for s in feedback}:
                raise ValueError("回标与当轮服务重复")
            stream.append((ids,p,[s["sample_id"] for s in feedback],predict(feedback),[r["label"] for r in feedback]))
    truth={r.sample_id:r.label for r in pkg.validation}
    return stream,truth,{name:delivered(loc) for name,loc in refs.items()}


def simulate(stream,truth,folder,budget,classes,controls=None,include_fixed=False):
    forecasts={}
    choices={}
    keys=[i for ids,*_ in stream for i in ids]
    blocks=np.split(np.arange(len(keys)),np.cumsum([len(s[0]) for s in stream])[:-1])
    if len(set(keys))!=len(keys):
        raise ValueError("服务编号重复")
    for seed in SEEDS:
        guard=GuardedVote(tuple(stream[0][1]),classes)
        old=FeedbackRouter(tuple(stream[0][1]),classes) if include_fixed else None
        for b,(ids,p,returned,fp,answers) in enumerate(stream):
            bank=guard.bank(p)
            predictions={"current":p["current"],"vote":bank["current"]}
            if old is not None:
                predictions.update(p)
                predictions[f"local-{seed}"]=apply_choices(p,old.choices("local"))
            for mode in ("vote-local","vote-confirm","guarded-vote"):
                selected=guard.choices(mode) if mode=="guarded-vote" else guard.router.choices(mode)
                name=f"{mode}-{seed}"
                predictions[name]=apply_choices(bank,selected)
                choices.setdefault(name,[]).append(selected)
            for name,v in predictions.items():
                forecasts.setdefault(name,np.empty(len(keys),int))[blocks[b]]=v
            ix=np.asarray(sorted(range(len(returned)),key=lambda n:digest([seed,returned[n]]))[:budget],int)
            y=[answers[i] for i in ix]
            sampled={n:v[ix] for n,v in fp.items()}
            guard.update(y,sampled)
            if old is not None:
                old.update(y,sampled)
    for name,p in (controls or {}).items():
        forecasts[name]=np.asarray([p[i] for i in keys])
    path=folder/"issued.json"
    atomic_json(path,{"keys":keys,"choices":choices,"predictions":{k:v.tolist() for k,v in forecasts.items()}})
    sealed=read(path)
    y=np.array([truth[k] for k in keys])
    table={k:scores(y,np.asarray(v),blocks,classes) for k,v in sealed["predictions"].items()}
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table,"budget":budget,"samples":len(keys)})
    return table


def external(path,truth,fixed):
    source=read(path)
    stream=[]
    pos=0
    for ids in source["keys"]:
        p={k:np.array(source["predictions"][k][pos:pos+len(ids)]) for k in fixed}
        stream.append((ids,p,ids,p,[truth[k] for k in ids]))
        pos+=len(ids)
    return stream


def verify_live():
    source=read(OUT/"guarded/late/issued.json")
    verified={}
    for mode in ("vote","vote-local","vote-confirm","guarded-vote"):
        for seed in ((17,) if mode=="vote" else (17,47,89)):
            name=f"{mode}-{seed}"
            folder=OUT/"live"/name
            report=read(folder/"run.json")
            prediction={}
            seen=0
            for evt in report["accepted"]:
                envelope=read(folder/"accepted"/(evt["request"]+".json"))
                payload=envelope["payload"]
                if (digest([report["identity"],payload])!=evt["request"] or payload["feedback_seen"]!=seen or
                    set(prediction)&set(payload["predictions"]) or evt["returned"]!=sorted(payload["predictions"],key=lambda k:digest([seed,k]))[:report["feedback_per_batch"]]):
                    raise ValueError("真实预测交接与反馈时点不一致")
                prediction.update(payload["predictions"])
                seen+=len(evt["returned"])
            expected=source["predictions"]["vote" if mode=="vote" else name]
            if set(prediction)!=set(source["keys"]) or [prediction[k] for k in source["keys"]]!=expected:
                raise ValueError("真实工作流与独立评分逐行不符")
            verified[name]={"accepted_rows":len(prediction),"feedback_seen_before_last":report["accepted"][-1]["feedback_seen"],
                            "setup_seconds":report["setup_seconds"],"service_seconds":sum(a["seconds"] for a in report["accepted"])}
    atomic_json(OUT/"guarded/live-verification.json",verified)


def main():
    stream,truth,controls=public_stream()
    simulate(stream,truth,OUT/"guarded/public",32,6,controls)
    batches=read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
    truth={r.sample_id:r.label for b in batches if b.batch_number>=3 for r in b.records}
    simulate(external(OUT/"fixed.json",truth,GasFeedbackBank.names),truth,OUT/"guarded/late",32,6)
    y,labels,keys=load_rows()
    truth=dict(zip((k for row in keys for k in row),y,strict=True))
    folder=OUT/"new-instrument"
    stream=external(folder/"issued.json",truth,read(folder/"scores.json")["fixed"])
    simulate(stream,truth,OUT/"guarded/one-year",4,len(labels))
    verify_live()


if __name__=="__main__":
    main()
