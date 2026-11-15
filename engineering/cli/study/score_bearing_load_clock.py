"""按原时间次序用已完成台架训练，比较运行累积、固定历史校正和现场工况选择。"""
import json
import joblib
import numpy as np

from score_bearing_wear_rate import DEST as PRIOR, SPEC as PRIOR_SPEC
from score_bearing_prefix import ARCHIVE_SIZE,ROOT,LocalStore,LocalAcceptance,atomic_json,describe,devices,digest,fingerprint,read
from harness.models.bearing_load_clock import LoadLife,LoadTrials,PRIMARY

DEST=PRIOR.parent/"bearing-load-clock"
SPEC=ROOT/"engineering/configs/experiments/bearing-load-clock.json"
ARM="与原强工具各半/"+PRIMARY


def evaluate(units,clock):
    previous=read(PRIOR/"scores.json")[clock]
    fixed_old=previous["decision"]["最佳固定"]
    endpoint=LocalAcceptance();endpoint.output=DEST/clock/"accepted"
    events=[];cost_history={}
    for u,key in zip(units[ARCHIVE_SIZE:],previous["requests"],strict=True):
        old=read(PRIOR/clock/"accepted"/(key+".json"))["payload"]
        if digest([fingerprint(PRIOR_SPEC),clock,old])!=key or old["ids"]!=u.ids:
            raise ValueError("前次轴承预测身份不符")
        past=[r for r in units if r.end<u.start]
        # 按同一历史集合留掉一整台，以检验训练器是否能迁移到下一台。
        trials=[]
        for h in past:
            archive=[r for r in past if r.name!=h.name]
            key_hist=(h.name,tuple(r.name for r in archive))
            if key_hist not in cost_history:
                fit=LoadLife().fit(archive)
                active,calendar,c=fit.predict(h.settings,h.ages["capped"],h.ages["calendar"])
                y=h.durations[clock]-h.ages[clock]
                b=calendar if clock=="calendar" else active
                # 每台单独按其时长缩放，防止一个长停机轨迹决定整库的选择。
                cost_history[key_hist]=(c,((b-y[:,None])/max(h.durations[clock],1))**2)
            trials.append(cost_history[key_hist])
        model=LoadLife().fit(past);policy=LoadTrials().fit(trials)
        path=DEST/clock/(u.name+".joblib")
        path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump((model,policy),path,compress=3);model,policy=joblib.load(path)
        active,calendar,c=model.predict(u.settings,u.ages["capped"],u.ages["calendar"])
        bank=calendar if clock=="calendar" else active
        outputs=policy.predict(c,bank)
        # 截断输入不应改变已经交付的头半段，检验现场选择同样满足这个性质。
        half=max(1,len(c)//2)
        a,b,sub=model.predict(u.settings[:half],u.ages["capped"][:half],u.ages["calendar"][:half])
        other=policy.predict(sub,b if clock=="calendar" else a)
        np.testing.assert_allclose(other[PRIMARY],outputs[PRIMARY][:half],rtol=0,atol=1e-8)
        anchor=np.array(old["预测"][fixed_old])
        paper_names=["static/phme-feature","update/phme-feature"]
        columns={n:old["预测"][n] for n in [fixed_old,"once",*paper_names]}
        for name,val in outputs.items():
            columns[name]=val.tolist()
            columns["与原强工具各半/"+name]=((val+anchor)/2).tolist()
            for peer in paper_names:
                columns[peer+"及载荷各半/"+name]=((val+old["预测"][peer])/2).tolist()
        pkt={"ids":u.ids,"预测":columns,"已结束":[h.name for h in past],"源权重":fingerprint(path),"旧回执":key}
        req=digest([fingerprint(SPEC),clock,pkt])
        if not endpoint.submit(req,pkt)["accepted"]:
            raise ValueError("累计载荷交接失败")
        events.append(req)
    ys=[];cols={}
    for u,key in zip(units[ARCHIVE_SIZE:],events,strict=True):
        pkt=read(endpoint.output/(key+".json"))["payload"]
        if digest([fingerprint(SPEC),clock,pkt])!=key:
            raise ValueError("累计载荷在评分时变化")
        ys.append(u.durations[clock]-u.ages[clock])
        for n,p in pkt["预测"].items():
            cols.setdefault(n,[]).append(np.array(p))
    scores=describe(cols,ys)
    fixed=min([n for n in cols if PRIMARY not in n],key=lambda k:scores[k]["RMSE"])
    paper=min([n for n in cols if "phme-" in n],key=lambda k:scores[k]["RMSE"])
    f=scores[fixed];p=scores[paper];a=scores[ARM]
    d={"主方案":ARM,"固定":fixed,"论文":paper,"现场误差":a["RMSE"],"固定误差":f["RMSE"],"论文误差":p["RMSE"],
       "最差整台":a["最差整台"],"固定最差":f["最差整台"],"开发通过":a["RMSE"]<=.98*min(f["RMSE"],p["RMSE"],scores["once"]["RMSE"]) and a["最差整台"]<=1.01*f["最差整台"]}
    return {"decision":d,"scores":scores,"requests":events}


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("该累计运行量已发放")
        units=devices()
        reports={clock:evaluate(units,clock) for clock in ("calendar","capped")}
        atomic_json(DEST/"scores.json",reports)
        print(json.dumps({k:v["decision"] for k,v in reports.items()},ensure_ascii=False))
