"""从集成入口的正式回执计分，并从交付前的快照对同一批逐行复算。"""
import numpy as np
from sklearn.metrics import f1_score

from engineering.cli.workflow.gas_observer_predict import infer
from harness.data.official_gas import prediction_label, prediction_request
from score_whole_maintenance import integrated_predictions, official
from score_site_training import OUT, ROOT, atomic_json, read


def run():
    folder=OUT/"integrated"
    results={}
    for task in ("gas","bearing","engine"):
        loc=folder/task
        report=read(loc/"run.json")
        if "accepted" in report:
            files=[loc/"accepted"/(e["request"]+".json") for e in report["accepted"]]
        else:
            files=[loc/e["submitted"] for e in report["rounds"]]
        records=[read(p)["payload"] for p in files]
        pred={k:v for row in records for k,v in row["predictions"].items()}
        old=integrated_predictions(task)
        package,_,_=official(task)
        truth={r.sample_id:r.label for r in package.validation}
        if old.keys()!=pred.keys() or len(pred)!=sum(len(row["predictions"]) for row in records):
            raise ValueError("官方服务身份不同或重复")
        if task=="gas":
            values={r.sample_id:r.values for r in package.validation}
            reference=read(OUT/"official/scores.json")
            for row,request in zip(records,reference["requests"],strict=True):
                snapshot=loc/"snapshots"/(row["snapshot"]+".json")
                keys=row["candidate_ids"]
                queries=(prediction_request(list(values[k])) for k in keys)
                actual=[prediction_label(res) for res in infer(snapshot,ROOT/"data/official",queries)]
                target=read(OUT/"official/accepted"/(request+".json"))["payload"]
                if (dict(zip(target["ids"],target["predictions"]["ongoing"],strict=True))!=row["predictions"] or
                        actual!=[pred[k] for k in keys] or row["selected_tool"]!=target["chosen"]):
                    raise ValueError("配方选择、快照预测或实际回执不一致")
            score=float(f1_score([truth[k] for k in pred],list(pred.values()),average="macro"))
        else:
            score=float(np.sqrt(np.mean([(v-truth[k])**2 for k,v in pred.items()])))
            if any(abs(v-old[k])>1e-7 for k,v in pred.items()):
                raise ValueError("原有寿命任务发行行为发生变化")
        results[task]={"score":score,"rows":len(pred),"different_from_current":sum(abs(v-old[k])>1e-7 for k,v in pred.items())}
    atomic_json(folder/"scores.json",results)
    print(results,flush=True)


if __name__=="__main__":
    run()
