"""以三个完整的开发重复比较识别气体时去除浓度影响的几种办法。"""
from hashlib import sha256
from pathlib import Path
import json
import joblib
import numpy as np
from scipy.io import loadmat

from verify_dsps_matched_entry import DEST as PRIOR, SOURCE, LocalAcceptance, LocalStore, atomic_json, digest, read, tally, target_features
from harness.models.dsps_dose_contrast import InstalledDoseRelease, PRIMARY, PAPER

DEST = PRIOR.parent / "dsps-dose-contrast"


def acquire(with_records=False):
    hashes = {r["成员"]: r["SHA256"] for r in read(SOURCE/"source.json")["已核验成员"]}
    with np.load(SOURCE/"development.npz") as z:
        ids, y, groups = (z[k] for k in ("ids", "y", "repetition"))
    levels = np.array([int(Path(k).stem[1:]) for k in ids])
    banks = {0: [], 11: []}
    records = []
    for uid in ids:
        p = SOURCE/"members"/uid
        if sha256(p.read_bytes()).hexdigest() != hashes[uid]:
            raise ValueError("装机原件身份变化")
        m = loadmat(p, squeeze_me=True, struct_as_record=False)[p.stem]
        for row, bank in banks.items():
            inputs = [getattr(m,k)[:,128*row:128*(row+1)].reshape(-1,16,8).transpose(0,2,1) for k in ("Bl","Tr","St")]
            bank.append(target_features(*inputs))
            if with_records and row == 11:
                records.append({"inputs":dict(zip(("Bl","Tr","St"),inputs,strict=True))})
    data = ids, y, groups, levels, np.stack(banks[0]), np.stack(banks[11])
    return (*data, records) if with_records else data


def main():
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("当前协议已有评分")
        ids,y,groups,levels,a,b = acquire()
        receiver = LocalAcceptance()
        receiver.output = DEST/"accepted"
        previous = read(PRIOR/"scores.json")
        events, truth, cols, choices = [], [], {}, {}
        for g, parent in zip(np.unique(groups), previous["requests"], strict=True):
            test = groups == g
            m = InstalledDoseRelease().fit(a[~test],b[~test],y[~test],groups[~test],levels[~test])
            path = DEST/f"fold-{g}.joblib"
            joblib.dump(m,path)
            predictions, estimated = joblib.load(path).predict(b[test])
            old = read(PRIOR/"accepted"/(parent+".json"))["payload"]
            if digest([int(g),old])!=parent or old["ids"]!=ids[test].tolist() or old["prediction"]!=predictions[PAPER].tolist():
                raise ValueError("原始论文工具未逐条复现")
            payload = {"ids":ids[test].tolist(),"预测":{k:v.tolist() for k,v in predictions.items()},
                       "估计浓度":estimated.tolist(),"安装选择":m.selections,"权重":sha256(path.read_bytes()).hexdigest()}
            key = digest([int(g),payload])
            if not receiver.submit(key,payload)["accepted"]:
                raise ValueError("交付未接收")
            sent = read(receiver.output/(key+".json"))["payload"]
            if digest([int(g),sent])!=key:
                raise ValueError("接收内容与发送不同")
            events.append(key)
            choices[str(g)] = sent["安装选择"]
            truth.extend(y[test])
            for k,v in sent["预测"].items():
                cols.setdefault(k,[]).extend(v)
        results = {k:tally(np.asarray(truth),np.asarray(v)) for k,v in cols.items()}
        fixed = max((k for k in results if k!=PRIMARY),key=lambda k:results[k]["F1"])
        gap = results[PRIMARY]["F1"] - results[fixed]["F1"]
        decision = {"主方案":PRIMARY,"固定":fixed,"论文":PAPER,"最小差值":gap,
                    "进独立":gap>=.01 and results[PRIMARY]["最差召回"]>=results[fixed]["最差召回"]-.02}
        atomic_json(DEST/"scores.json",{"scores":results,"decision":decision,"requests":events,"安装选择":choices})
        print(json.dumps({"开发交付":results,"取舍":decision},ensure_ascii=False))


if __name__ == "__main__":
    main()
