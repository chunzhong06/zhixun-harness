"""把公开十轮真正留下的混淆计数带至后续采集，核对继续、重启和历史衰减。"""
from hashlib import sha256
from statistics import mean

import numpy as np

from check_feedback_controls import scores
from gas_complementary_feedback import OUT,ROOT,REGISTERED_SHA256,read_gas_batches,read,digest,atomic_json
from harness.models.gas_feedback_bank import GasFeedbackBank
from harness.models.observer_errors import ObserverErrors
from score_gas_source_confusions import predict
from score_guarded_gas import SEEDS,external


def main():
    root=OUT/"observer-errors"
    batches=read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
    truth={r.sample_id:r.label for b in batches if b.batch_number>=3 for r in b.records}
    stream=external(OUT/"fixed.json",truth,GasFeedbackBank.names)
    keys=[i for group,*_ in stream for i in group]
    blocks=np.split(np.arange(len(keys)),np.cumsum([len(r[0]) for r in stream])[:-1])
    out={}
    used={}
    for seed in SEEDS:
        pub=root/"live/public"/str(seed)
        late=root/"live/late"/str(seed)
        log=read(pub/"run.json")
        saved=read(pub/"checkpoint/state.json")
        if log["identity"]!=saved["identity"] or log["accepted"]!=saved["accepted"]:
            raise ValueError("公开服务的最终回标与运行不一致")
        old=ObserverErrors(GasFeedbackBank.names,6,saved["counts"])
        n=int(old.counts[0].sum())
        if n!=sum(len(e["returned"]) for e in log["accepted"]):
            raise ValueError("历史真实标签数不一致")
        last=read(pub/"accepted"/(log["accepted"][-1]["request"]+".json"))["payload"]
        first=read(late/"accepted"/(read(late/"run.json")["accepted"][0]["request"]+".json"))["payload"]
        if last["weights"]!=first["model_sha256"]:
            raise ValueError("现场切换的工业权重不同")
        current=ObserverErrors(GasFeedbackBank.names,6)
        used[str(seed)]=n
        for ids,p,eligible,outputs,y in stream:
            pred={"reset":current.predict(p,"uniform"),"carry":predict(old.counts,current.counts,p),
                  "shrink":predict(old.counts*32/n,current.counts,p),"frozen":old.predict(p,"uniform")}
            for k,v in pred.items():
                out.setdefault(f"{k}-{seed}",[]).extend(v.tolist())
            ix=np.asarray(sorted(range(len(eligible)),key=lambda j:digest([seed,eligible[j]]))[:32])
            current.update([y[j] for j in ix],{k:v[ix] for k,v in outputs.items()})
    dest=root/"site-handoff"
    path=dest/"issued.json"
    atomic_json(path,{"keys":keys,"past_labels":used,"predictions":out})
    y=np.asarray([truth[k] for k in keys])
    table={k:scores(y,np.asarray(v),blocks,6) for k,v in read(path)["predictions"].items()}
    old=read(root/"late/issued.json")["predictions"]
    if any(old[f"uniform-{s}"]!=out[f"reset-{s}"] for s in SEEDS):
        raise ValueError("现场重启与逐条实测不同")
    means={k:{m:mean(table[f"{k}-{s}"][m] for s in SEEDS) for m in ("pooled","equal_batches","worst_batch")}
           for k in ("reset","carry","shrink","frozen")}
    atomic_json(dest/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"arms":table,"means":means})
    print(means)


if __name__=="__main__":
    main()
