"""按真实发放逐流计分，并将当前和上轮的固定方法及同工具论文放入同一比较。"""
import json
import numpy as np
from score_gas_returned_errors import BASE, STREAMS, DEST as PRIOR, read, digest, f1_score, recall_score, LocalStore, atomic_json
from engineering.cli.workflow.gas_audit_stream import run as serve, audit

DEST = PRIOR.parent/"gas-audit-feedback"


def scores(folder,run,truth,classes):
    ids,cols = [],{}
    for e in run["events"]:
        payload = read(folder/"accepted"/(e["request"]+".json"))["payload"]
        if digest([run["identity"],payload])!=e["request"]:
            raise ValueError("实际发出的预测记录已变")
        ids.extend(payload["ids"])
        for k,p in payload["预测"].items():
            cols.setdefault(k,[]).extend(p)
    if len(set(ids))!=len(ids) or set(ids)!=set(truth):
        raise ValueError("未覆盖该服务流")
    y = [truth[k] for k in ids]
    return {k:{"F1":float(f1_score(y,v,labels=classes,average="macro",zero_division=0)),
               "最差召回":float(recall_score(y,v,labels=classes,average=None,zero_division=0).min())} for k,v in cols.items()}


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("当前抽查选法已开发验收")
        streams = {}
        for site,path in STREAMS.items():
            src = BASE/path
            spec = read(src/"stream.json")
            streams[site] = {}
            for q in audit.POLICIES:
                out = DEST/site/q
                issued = serve(src/"stream.json",out,q)
                streams[site][q] = scores(out,issued,read(src/"evaluation.json"),list(range(1,spec["classes"]+1)))
        old = read(PRIOR/"scores.json")
        means = {k:dict(v) for k,v in old["means"].items()}
        for q in audit.POLICIES:
            means.setdefault(q,{})
            for k in next(iter(streams.values()))[q]:
                means[q][k] = {m:float(np.mean([s[q][k][m] for s in streams.values()])) for m in ("F1","最差召回")}
        main_arm = ("混合抽查",audit.PRIMARY)
        fixed = [old["decision"]["固定"],*((q,k) for q in audit.POLICIES if q!="BADGE" for k in audit.ARMS)]
        get = lambda a:means[a[0]][a[1]]["F1"]
        fixed = max(fixed,key=get)
        paper = max((("BADGE",k) for k in means["BADGE"] if k!="各流回看最强单通道"),key=get)
        diff = get(main_arm)-max(get(fixed),get(paper))
        decision = {"主候选":main_arm,"固定":fixed,"论文":paper,"现场F1":get(main_arm),"固定F1":get(fixed),
                    "论文F1":get(paper),"最小领先":diff,"进独立":diff>=.01 and means[main_arm[0]][main_arm[1]]["最差召回"]>=means[fixed[0]][fixed[1]]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"means":means,"streams":streams,"decision":decision})
        print(json.dumps({"正式开发比较":decision},ensure_ascii=False))


if __name__=="__main__":
    main()
