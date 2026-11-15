"""用过去的振幅预测兑现误差选择速度窗，按十台原顺序交付全部寿命预测。"""
import json
import numpy as np
from score_bearing_prefix import ARCHIVE_SIZE, ROOT, LocalAcceptance, LocalStore, atomic_json, describe, devices, digest, fingerprint, read
from score_bearing_landmarks import DEST as PRIOR, SPEC as PRIOR_SPEC
from harness.models.bearing_wear_rate import PRIMARY, WearRate

DEST=PRIOR.parent/"bearing-wear-rate"
SPEC=ROOT/"engineering/configs/experiments/bearing-wear-rate.json"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():raise ValueError("相同协议已有结果")
        units=devices()
        old=read(PRIOR/"scores.json")
        version=fingerprint(SPEC)
        reports={}
        for clock in ("calendar","capped"):
            endpoint=LocalAcceptance()
            endpoint.output=DEST/clock/"accepted"
            prior=old[clock]
            events=[]
            for u,parent in zip(units[ARCHIVE_SIZE:],prior["requests"],strict=True):
                past=[r for r in units if r.end<u.start]
                m=WearRate().fit(past)
                raw,choices=m.predict(u.spectral,u.ages["capped"],clock=="calendar")
                for n in (1,6,len(choices)//2):
                    p,c=m.predict(u.spectral[:n],u.ages["capped"][:n],clock=="calendar")
                    if not np.array_equal(raw[PRIMARY][:n],p[PRIMARY]) or not np.array_equal(choices[:n],c):
                        raise ValueError("速度验证使用了尚未出现的振动")
                prev=read(PRIOR/clock/"accepted"/(parent+".json"))["payload"]
                if digest([fingerprint(PRIOR_SPEC),clock,prev])!=parent or prev["ids"]!=u.ids:
                    raise ValueError("原始交付身份改变")
                col=prev["预测"].copy()
                anchor=np.array(col[prior["decision"]["最佳固定"]])
                for name,v in raw.items():
                    col[name]=v.tolist()
                    col["与前期固定各半/"+name]=((anchor+v)/2).tolist()
                payload={"ids":u.ids,"预测":col,"历史设备":[p.name for p in past],
                         "已验证速度窗序号":choices.tolist(),"装机终点振幅":m.terminal,"装机时间上限尺度":m.scale,
                         "装机日历与运行倍率":m.ratio,"前次回执":parent}
                key=digest([version,clock,payload])
                if not endpoint.submit(key,payload)["accepted"]:raise ValueError("原始时序交接失败")
                events.append(key)
            cols,truth={},[]
            for u,key in zip(units[ARCHIVE_SIZE:],events,strict=True):
                d=read(endpoint.output/(key+".json"))["payload"]
                if digest([version,clock,d])!=key:raise ValueError("发放结果改变")
                for k,v in d["预测"].items():cols.setdefault(k,[]).append(np.asarray(v))
                truth.append(u.durations[clock]-u.ages[clock])
            scores=describe(cols,truth)
            fixed=min((k for k in scores if k.startswith(("固定","static/","update/","旧固定与","历史寿命","振动阶段","与前期固定各半/固定"))),key=lambda k:scores[k]["RMSE"])
            arm="与前期固定各半/"+PRIMARY
            a,f=scores[arm],scores[fixed]
            paper=prior["decision"]["论文误差"]
            decision={"主方案":arm,"最佳固定":fixed,"固定误差":f["RMSE"],"论文误差":paper,"现场误差":a["RMSE"],
                      "最差整台":a["最差整台"],"开发通过":a["RMSE"]<=.98*min(f["RMSE"],paper,scores["once"]["RMSE"]) and a["最差整台"]<=1.01*f["最差整台"]}
            reports[clock]={"scores":scores,"decision":decision,"requests":events,"设备":[u.name for u in units[ARCHIVE_SIZE:]]}
        atomic_json(DEST/"scores.json",reports)
        print(json.dumps({k:v["decision"] for k,v in reports.items()},ensure_ascii=False))


if __name__=="__main__":
    main()
