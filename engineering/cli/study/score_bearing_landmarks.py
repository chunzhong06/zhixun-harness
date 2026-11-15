"""按每个振动阶段的实际到达时刻训练寿命工具，已结束的整台才参与试用和下一台安装。"""
import json
import numpy as np
from score_bearing_prefix import (ARCHIVE_SIZE, DEST as PRIOR, ROOT, LocalAcceptance, LocalStore, atomic_json,
                                 describe, devices, digest, fingerprint, read)
from harness.models.bearing_landmarks import ADAPTIVE, LandmarkLife, LandmarkTrials, record

DEST = PRIOR.parent/"bearing-landmarks"
SPEC = ROOT/"engineering/configs/experiments/bearing-landmarks.json"


def main():
    units=devices()
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("相同协议已有结果")
        version=fingerprint(SPEC)
        previous=read(PRIOR/"scores.json")
        reports={}
        rec={u.name:record(u) for u in units}
        for clock in ("calendar","capped"):
            hist=[]
            installed=units[:ARCHIVE_SIZE]
            for target in installed:
                m=LandmarkLife().fit([rec[u.name] for u in installed if u!=target])
                row=rec[target.name]
                bank,st=m.predict(row["condition"],row["age"],target.ages[clock])
                hist.append((bank,st,target.durations[clock]-target.ages[clock]))
            issued,events=[],[]
            receiver=LocalAcceptance()
            receiver.output=DEST/clock/"accepted"
            choices=[]
            for u,parent in zip(units[ARCHIVE_SIZE:],previous[clock]["requests"],strict=True):
                known=[r for r in units if r.end<u.start]
                policy=LandmarkTrials().fit([*hist,*[(b,s,p.durations[clock]-p.ages[clock]) for p,b,s in issued if p.end<u.start]])
                m=LandmarkLife().fit([rec[r.name] for r in known])
                row=rec[u.name]
                bank,st=m.predict(row["condition"],row["age"],u.ages[clock])
                live=policy.predict(bank,st)
                for n in (1,7,len(st)//2):
                    sub,states=m.predict(row["condition"][:n],row["age"][:n],u.ages[clock][:n])
                    if not np.array_equal(policy.predict(sub,states),live[:n]):
                        raise ValueError("振动阶段读取了本台未来信息")
                prior=read(PRIOR/clock/"accepted"/(parent+".json"))["payload"]
                if digest([fingerprint(ROOT/"engineering/configs/experiments/bearing-prefix.json"),clock,prior])!=parent or prior["ids"]!=u.ids:
                    raise ValueError("前次交付错位")
                # 固定对照同样可以选用原固定趋势工具和新工具的固定配比。
                baseline=np.array(prior["预测"][previous[clock]["decision"]["最佳固定"]])
                combined={**prior["预测"],**{k:v.tolist() for k,v in bank.items()},ADAPTIVE:live.tolist()}
                for k,v in bank.items():
                    combined[f"旧固定与{k}等权"]=((v+baseline)/2).tolist()
                payload={"ids":u.ids,"预测":combined,"过去设备":[r.name for r in known],
                         "已达到阶段数":st.tolist(),"各阶段采用":policy.actions,"原回执":parent}
                key=digest([version,clock,payload])
                if not receiver.submit(key,payload)["accepted"]:raise ValueError("未完成交接")
                events.append(key);choices.append(policy.actions);issued.append((u,bank,st))
            truth,cols=[],{}
            for (u,_,_),key in zip(issued,events,strict=True):
                sent=read(receiver.output/(key+".json"))["payload"]
                if digest([version,clock,sent])!=key:raise ValueError("回执预测变化")
                truth.append(u.durations[clock]-u.ages[clock])
                for k,v in sent["预测"].items():cols.setdefault(k,[]).append(np.asarray(v))
            scores=describe(cols,truth)
            old=previous[clock]["decision"]
            fixed=min((k for k in scores if k.startswith(("固定","static/","update/","旧固定与","历史寿命","振动阶段"))),
                      key=lambda k:scores[k]["RMSE"])
            f,p,a=scores[fixed],scores[old["论文"]],scores[ADAPTIVE]
            decision={"最佳固定":fixed,"现场误差":a["RMSE"],"固定误差":f["RMSE"],"论文误差":p["RMSE"],
                      "最差整台":a["最差整台"],"开发通过":a["RMSE"]<=.98*min(f["RMSE"],p["RMSE"],scores["once"]["RMSE"]) and a["最差整台"]<=1.01*f["最差整台"]}
            reports[clock]={"scores":scores,"decision":decision,"requests":events,"设备":[u.name for u,_,_ in issued],"选择":choices}
        atomic_json(DEST/"scores.json",reports)
        print(json.dumps({k:v["decision"] for k,v in reports.items()},ensure_ascii=False))


if __name__=="__main__":
    main()
