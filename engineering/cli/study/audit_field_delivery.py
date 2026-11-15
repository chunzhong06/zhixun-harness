"""只用已确认的输出计分；核对各策略的可见反馈、固定模型和外部服务身份。"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import f1_score, recall_score

from harness.contracts import digest
from harness.data.official_gas import read_official_gas
from harness.data.official_gas import prediction_label, prediction_request
from harness.models.field_classification import MODES, TOOLS
from harness.runtime.storage import atomic_json
from prepare_dynamic_mixtures import DEST as DATA, ROOT
from engineering.cli.workflow.gas_observer_predict import infer


OUT=ROOT/"results/experiments/local-defaults/whole-maintenance/field-delivery"
PROTOCOL=ROOT/"engineering/configs/experiments/field-policy-delivery.json"


def read(path):
    return json.loads(path.read_bytes())


def score(y,p,classes):
    return {"macro_f1":float(f1_score(y,p,labels=classes,average="macro",zero_division=0)),
            "min_recall":float(min(recall_score(y,p,labels=classes,average=None,zero_division=0)))}


def audit(site):
    public=site=="official"
    if public:
        package=read_official_gas(ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip")
        truth={r.sample_id:r.label for r in package.validation}
        classes=list(range(1,7))
    else:
        truth=read(DATA/site/"evaluation.json")
        classes=[1,2]
    common=None
    outputs={}
    choices={}
    costs={}
    block_ids=[]
    evidence={}
    for mode in ("ongoing","once","recent-backtest",*TOOLS):
        loc=OUT/site/mode
        if public:
            loc=OUT/"official-fixed"/mode if mode in TOOLS else loc/"gas"
        report=read(loc/"run.json")
        evidence[mode]=sha256((loc/"run.json").read_bytes()).hexdigest()
        costs[mode]=report["field_cost"] if public else report["cost"]
        deployed={}
        receipts=[]
        selections=[]
        seen=set()
        for event in report["accepted"]:
            rec=read(loc/"accepted"/(event["request"]+".json"))
            p=rec["payload"]
            if (digest([report["identity"],p])!=rec["request_id"] or rec["request_id"]!=event["request"]
                    or p["feedback_seen"]!=len(seen) or seen & set(event["returned"])):
                raise ValueError("发放或反馈历史不同")
            ids=p["candidate_ids"]
            selection=p["selected_tool"]
            if ([p["predictions"][k] for k in ids]!=p["candidate_predictions"][selection]
                    or deployed.keys() & p["predictions"].keys()):
                raise ValueError("实际提交未使用选中的模型或服务编号重复")
            if mode=="ongoing":
                block_ids.append(ids)
                for a,values in p["source_predictions"].items():
                    outputs.setdefault("source-"+a,{}).update(zip(ids,values,strict=True))
                if "vote" in p:
                    outputs.setdefault("vote",{}).update(zip(ids,p["vote"],strict=True))
                else:
                    from harness.models.vote_feedback import majority
                    voted=majority({k:np.asarray(v) for k,v in p["source_predictions"].items()},len(classes))
                    outputs.setdefault("vote",{}).update(zip(ids,voted.tolist(),strict=True))
            receipts.append(digest([ids,p["candidate_predictions"],p["source_predictions"],event["returned"]]))
            deployed.update(p["predictions"])
            selections.append(selection)
            seen.update(event["returned"])
        if common is None:
            common=receipts
        elif receipts!=common:
            raise ValueError(f"{site} 的 {mode} 组候选或所见资料不同")
        choices[mode]=selections
        outputs[mode]=deployed
    ids=[k for batch in block_ids for k in batch]
    if not public and set(ids)!=truth.keys():
        raise ValueError("新场地覆盖不完整")
    y=np.asarray([truth[k] for k in ids])
    vectors={a:np.asarray([v[k] for k in ids]) for a,v in outputs.items()}
    table={a:score(y,p,classes) for a,p in vectors.items()}
    if public:
        baseline=ROOT/"results/experiments/local-defaults/gas-complementary-feedback/feedback-refit/public"
        ref={"observer":{},"mixed":{}}
        for path in sorted(baseline.glob("batch-*.json")):
            row=read(path)
            for arm,reference in (("mixed","refit"),("observer","observer")):
                ref[arm].update(zip(row["ids"],row["predictions"][reference],strict=True))
        if any(ref[a]!=outputs[a] for a in ref):
            raise ValueError("交付未复现原有真实校准或固定混合训练")
    fixed=[a for a in vectors if a not in ("ongoing","once","recent-backtest")]
    strongest=max(fixed,key=lambda a:table[a]["macro_f1"])
    ongoing,once=vectors["ongoing"],vectors["once"]
    changed=ongoing!=once
    # 额外诊断：事后按同批真实准确率挑工具。只解释策略是否还有修错空间，不作可交付分数。
    block_oracle=[]
    for batch in block_ids:
        arm=max(TOOLS,key=lambda a:sum(outputs[a][k]==truth[k] for k in batch))
        block_oracle.extend(outputs[arm][k] for k in batch)
    result={"protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"streams":evidence,
            "scores":table,"costs":costs,"rows":len(ids),"batches":len(block_ids),"labels_used":len(seen),
            "strongest_fixed":strongest,"selection_counts":{a:dict(Counter(v)) for a,v in choices.items()},
            "ongoing_choices":choices["ongoing"],"once_choices":choices["once"],"changed_vs_once":int(changed.sum()),
            "rescued_vs_once":int(np.sum(changed & (ongoing==y))),"spoiled_vs_once":int(np.sum(changed & (once==y))),
            "hindsight_accuracy_selected":score(y,np.asarray(block_oracle),classes),
            "meets_gate":all(table["ongoing"]["macro_f1"]>=table[a]["macro_f1"]+.01 for a in (*fixed,"once"))
                        and table["ongoing"]["min_recall"]>=table[strongest]["min_recall"]-.02}
    atomic_json(OUT/site/"scores.json",result)
    print(site,len(ids),strongest,{a:round(table[a]["macro_f1"],5) for a in (*MODES,strongest)},
          result["rescued_vs_once"],result["spoiled_vs_once"],flush=True)
    return result


def check_single_rows():
    data_root=ROOT/"data/official"
    source=read_official_gas(data_root/"gas/cpu-a-guide-20261110-v1.zip")
    measured={r.sample_id:r.values for r in source.validation}
    counts={}
    for mode, folder in (("ongoing",OUT/"official/ongoing/gas"),("mixed",OUT/"integrated-fixed/gas")):
        run=read(folder/"run.json")
        count=0
        for e in run["accepted"]:
            p=read(folder/"accepted"/(e["request"]+".json"))["payload"]
            snapshot=folder/"snapshots"/(p["snapshot"]+".json")
            ids=p["candidate_ids"]
            requests=(prediction_request(list(measured[k])) for k in ids)
            actual=[prediction_label(r) for r in infer(snapshot,data_root,requests)]
            if actual != [p["predictions"][k] for k in ids]:
                raise ValueError("官方单条入口与已接受的维护预测不同")
            count+=len(ids)
        counts[mode]=count
    atomic_json(OUT/"official/single-row.json",counts)


if __name__=="__main__":
    atomic_json(OUT/"scores.json",{site:audit(site) for site in ("official","co","methane")})
    check_single_rows()
