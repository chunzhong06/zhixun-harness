"""把逐次试用与最强固定配比一起经已保存权重发放，再按接收的结果评分。"""
from hashlib import sha256
import json
import subprocess
import sys
import joblib
import numpy as np
from score_dsps_dose_contrast import DEST as PRIOR, acquire, read, LocalStore, LocalAcceptance, digest, atomic_json, tally
from engineering.cli.workflow.dsps_cycle_predict import predict, ROOT
from harness.models.dsps_dose_contrast import InstallationTrials, TRIAL, SIMPLE, EQUAL, PAPER

DEST = PRIOR.parent / "dsps-install-trials"


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("本次安装试用已结束")
        ids,y,groups,levels,a,b,records = acquire(with_records=True)
        store = LocalAcceptance()
        store.output = DEST/"accepted"
        cols,truth,events,detail = {},[],[],{}
        for group,parent in zip(np.unique(groups),read(PRIOR/"scores.json")["requests"],strict=True):
            test = groups==group
            model = InstallationTrials().fit(a[~test],b[~test],y[~test],groups[~test],levels[~test])
            weight = DEST/f"fold-{group}.joblib"
            joblib.dump(model,weight)
            sha = sha256(weight.read_bytes()).hexdigest()
            frozen = joblib.load(weight)
            predictions,_ = frozen.predict(b[test])
            old = read(PRIOR/"accepted"/(parent+".json"))["payload"]
            if (digest([int(group),old])!=parent or old["ids"]!=ids[test].tolist()
                or any(old["预测"][k]!=predictions[k].tolist() for k in (PAPER,EQUAL))):
                raise ValueError("同工具固定赢家未复现")
            ix = np.flatnonzero(test)
            for arm,mode in ((EQUAL,"dose-equal"),(TRIAL,"install-trials")):
                actual = [predict(frozen,records[j],arm)["outputs"]["class"] for j in ix]
                if actual!=predictions[arm].tolist():
                    raise ValueError("逐次发出与批量预测不同")
                command = [sys.executable,"-B",str(ROOT/"engineering/cli/workflow/dsps_cycle_predict.py"),
                           "--artifact",str(weight),"--sha256",sha,"--method",mode]
                sample = {k:v.tolist() for k,v in records[ix[0]]["inputs"].items()}
                sent = subprocess.run(command,input=json.dumps({"inputs":sample})+"\n",text=True,
                                      capture_output=True,check=True,cwd=ROOT)
                if json.loads(sent.stdout)["outputs"]["class"]!=actual[0]:
                    raise ValueError("命令行发放结果不一致")
            payload = {"ids":ids[test].tolist(),"预测":{k:v.tolist() for k,v in predictions.items()},
                       "权重":sha,"装机所选配比":frozen.best}
            key = digest([int(group),payload])
            if not store.submit(key,payload)["accepted"]:
                raise ValueError("接收端拒绝")
            accepted = read(store.output/(key+".json"))["payload"]
            if digest([int(group),accepted])!=key:
                raise ValueError("回执内容改变")
            events.append(key)
            truth.extend(y[test])
            for k,v in accepted["预测"].items():
                cols.setdefault(k,[]).extend(v)
            detail[str(group)] = {"试用条数":len(frozen.trial_truth),"试用等权正确":int(sum(frozen.trial_truth==frozen.trial_equal)),
                                  "改判数":int(sum(predictions[TRIAL]!=predictions[EQUAL])),"装机所选":frozen.best}
        scores = {k:tally(np.asarray(truth),np.asarray(v)) for k,v in cols.items()}
        fixed = max((k for k in cols if k not in (TRIAL,SIMPLE)),key=lambda k:scores[k]["F1"])
        gap = scores[TRIAL]["F1"] - scores[fixed]["F1"]
        decision = {"主方案":TRIAL,"固定":fixed,"固定F1":scores[fixed]["F1"],"论文F1":scores[PAPER]["F1"],
                    "现场F1":scores[TRIAL]["F1"],"差值":gap,
                    "进独立":gap>=.01 and scores[TRIAL]["最差召回"]>=scores[fixed]["最差召回"]-.02}
        # 同一装机集保存可部署权重，供复用和冻结；不会开启独立数据。
        model = InstallationTrials().fit(a,b,y,groups,levels)
        final = DEST/"release.joblib"
        joblib.dump(model,final)
        atomic_json(DEST/"scores.json",{"scores":scores,"decision":decision,"requests":events,
                    "装机试用":detail,"发布权重SHA256":sha256(final.read_bytes()).hexdigest()})
        print(json.dumps({"开发交付":decision,"固定等权":scores[EQUAL],"装机所选":scores[SIMPLE],
                          "外层开发交付条数":len(y),
                          "每折装机试用条数":[v["试用条数"] for v in detail.values()]},ensure_ascii=False))


if __name__=="__main__":
    main()
