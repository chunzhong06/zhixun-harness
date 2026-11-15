"""逐轮核对本地队员入口与冻结对照所交出的寿命预测、决定、训练和费用。"""
from hashlib import sha256

from compare_fixed_tree_references import OUT
from compare_group_objectives import atomic_json,digest,official,read,remaining_cycles,remaining_minutes,summary
from score_local_defaults import cost


def main():
    table={}
    for task,cohort,mode in (("bearing","bearing","median-confirm"),("bearing","pronostia","median-confirm"),
                             ("engine","engine","tree-confirm"),("engine","phm08","tree-confirm")):
        pkg,checksum,batches=official(task)
        ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
        folder=OUT/"live"/cohort
        report=read(folder/"run.json")
        state=read(folder/"regular/state.json")
        prior=read(OUT/"routing"/cohort/"issued.json")
        out={}
        if report["policy"]!=mode or report["completed_rounds"]!=len(batches) or len(report["accepted"])!=len(batches):
            raise ValueError("交付未完成或策略有误")
        for receipt,expected in zip(report["accepted"],prior["history"],strict=True):
            submitted=read(folder/"accepted"/(receipt["request"]+".json"))
            payload=submitted["payload"]
            batch=payload["batch"]
            values=payload["predictions"]
            regular=state["completed"][batch]
            if (batch!=expected["batch"] or receipt["chosen"]!=payload["decision"]["chosen"]
                    or receipt["request"]!=submitted["request_id"] or receipt["request"]!=digest([report["identity"],payload])
                    or digest(values)!=receipt["outputs_sha256"] or payload["selected"]!=regular["model_id"]
                    or payload["input_digest"]!=regular["input_digest"] or values.keys()!=regular["predictions"].keys()
                    or values.keys()&out.keys() or payload["decision"]["chosen"]!=expected["choices"][mode]["chosen"]):
                raise ValueError("真实采用、交接或重放选择不同")
            out.update(values)
        reference=prior["predictions"][mode]
        delta=max(abs(y-reference[i]) for i,y in out.items())
        if delta>1e-7 or reference.keys()!=out.keys():
            raise ValueError(f"入口与冻结对照预测不同 {cohort} {delta}")
        truth={r.sample_id:r.label for r in pkg.validation}
        if cohort=="pronostia":
            truth=remaining_minutes("full-test-manifest.json")
        elif cohort=="phm08":
            truth=remaining_cycles()
        scores=summary(task,truth,{"parent":prior["predictions"]["current"],"live":out},
                       {k:state["samples"][k]["unit_id"] for k in out})
        table[cohort]={"policy":mode,"paired_predictions":len(out),"max_abs_difference":delta,"scores":scores,
                       "regular_cost":cost(folder/"regular",checksum,ids),"extra_window_seconds":sum(v["reference_seconds"] for v in report["accepted"]),
                       "tree_trained":report["tree_artifact"] is not None,"run_sha256":sha256((folder/"run.json").read_bytes()).hexdigest()}
    atomic_json(OUT/"live"/"scores.json",table)
    print({k:(round(v["scores"]["live"]["score"],3),v["paired_predictions"],round(v["extra_window_seconds"],3)) for k,v in table.items()})


if __name__=="__main__":
    main()
