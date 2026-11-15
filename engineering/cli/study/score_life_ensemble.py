"""对同一批已封存寿命预测做简单稳健集成，发出新预测后才读取服务答案。"""
from hashlib import sha256

import numpy as np

from compare_fixed_tree_references import OUT,paths
from compare_group_objectives import atomic_json,official,read,remaining_cycles,remaining_minutes,summary
from score_tree_reference_transfer import suites


def compose(folder,task,suite,groups,answers):
    path=folder/"issued.json"
    expected=read(folder/"scores.json")["issued_sha256"]
    if sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError("冻结预测与原评分身份不一致")
    source=read(path)["predictions"]
    retained={k:p for k,p in source.items() if k in
              ("current","forest-only","catboost-only","catboost-half","median","fixed","paper","simple")}
    keys=list(source["current"])
    if any(set(p)!=set(keys) for p in retained.values()) or any(k not in groups for k in keys):
        raise ValueError("候选的测量或设备不一致")
    values=np.stack([[source[name][k] for k in keys] for name in ("current","forest-only","catboost-only")])
    for name,outputs in (("three-median",np.median(values,axis=0)),("three-mean",values.mean(0)),
                         ("two-tree-mean",values[1:].mean(0))):
        retained[name]=dict(zip(keys,outputs.tolist(),strict=True))
    output=OUT/"ensemble"/suite
    sealed=output/"issued.json"
    atomic_json(sealed,{"source_sha256":expected,"predictions":retained})
    p=read(sealed)["predictions"]
    table=summary(task,answers(),{"parent":p["current"],**p},groups)
    atomic_json(output/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":table})
    return {k:{m:r[m] for m in ("score","worst_group","rows")} for k,r in table.items()}


def main():
    results={}
    for task,second in (("engine","phm08"),("bearing","pronostia")):
        pkg,_,_=official(task)
        for cohort in (task,second):
            current,_=paths(cohort,task)
            state=read(current/"state.json")["samples"]
            groups={k:v["unit_id"] for k,v in state.items()}
            def answers():
                if cohort==task:
                    return {r.sample_id:r.label for r in pkg.validation}
                return remaining_cycles() if task=="engine" else remaining_minutes("full-test-manifest.json")
            results[cohort]=compose(OUT/"routing"/cohort,task,cohort,groups,answers)
        for suite,_,groups,answers in suites(task,False):
            folder=OUT/"transfer"/suite
            if not (folder/"issued.json").is_file():
                folder=OUT/"routing/transfer"/suite
            results[suite]=compose(folder,task,suite,groups,answers)
    atomic_json(OUT/"ensemble/scores.json",results)
    for name,t in results.items():
        print(name,{k:tuple(round(t[k][m],2) for m in ("score","worst_group"))
                    for k in ("current","catboost-only","catboost-half","three-median","three-mean","two-tree-mean")})


if __name__=="__main__":
    main()
