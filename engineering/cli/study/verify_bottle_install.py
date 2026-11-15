"""让真实命令自己提问，只交出已请求的本机标样；再由新进程接收未知配方读数。"""
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import f1_score, recall_score

from score_paired_design import DEST as PRIOR, ADAPT, CHOICE, LocalStore, LocalAcceptance, atomic_json,digest
from score_calibration_query import ROOT, DATA, NAME, chemistry
from score_year_gas_calibration import HASH

DEST = PRIOR.parent/"bottle-entry"
CLI = ROOT/"engineering/cli/workflow/gas_bottle_install.py"


def fit_by_request(source_file, raw, group, ids, size, gas, rule, method, weight, temp):
    args = [sys.executable,"-B",str(CLI),"install","--source",str(source_file),"--gas",gas,"--budget",str(size),
            "--rule",rule,"--readout",method,"--artifact",str(weight)]
    process = subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,encoding="utf-8",text=True)
    asked = []
    try:
        for _ in range(size):
            line = process.stdout.readline()
            if not line:
                raise ValueError("装机进程未提出足额测量请求")
            uid = json.loads(line)["request"]
            if uid in asked or uid not in ids[:120]:
                raise ValueError("请求重复或越过装机配方池")
            # 写出的文件仅含这一次已购买的真实周期，询问器没有整个目标池的文件路径。
            np.savez(temp,id=uid,cycles=raw[group==uid])
            process.stdin.write(json.dumps({"id":uid,"file":str(temp)})+"\n")
            process.stdin.flush()
            asked.append(uid)
        identity = json.loads(process.stdout.readline())
        process.stdin.close()
        if process.wait(timeout=30) or identity["obtained"]!=asked:raise ValueError("装机记录与实际返回不同")
        return asked,identity
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if not process.stdin.closed:process.stdin.close()
        process.stdout.close()


def run(folder, source_dir=PRIOR, destination=DEST, plan=None, expected_id=None):
    previous = json.loads((source_dir/"scores.json").read_bytes())
    with (DATA/NAME).open("rb") as f:
        if file_digest(f,"md5").hexdigest()!=HASH:raise ValueError("原始波形身份变化")
    raw = loadmat(DATA/NAME,variable_names=["target","sensor"],simplify_cells=True)
    t = raw["target"]
    valid = np.isfinite(t["states"]) & np.isfinite(t["acetone"]) & np.isfinite(t["toluene"])
    group = t["states"][valid].astype(int)
    from hashlib import sha256
    ids = sorted(np.unique(group).tolist(),key=lambda i:sha256(f"year-gas-v1/{i}".encode()).digest())
    source,*chips = sorted({n.rsplit("_",1)[0] for n in raw["sensor"]})
    def cycles(chip):
        return np.stack([raw["sensor"][chip+f"_sensor{i}"][valid] for i in range(4)],axis=1)
    take = np.isin(group,ids[:120])
    source_file = folder/"source.npz"
    np.savez(source_file,cycles=cycles(source)[take],group=group[take],ids=np.asarray(ids[:120]),composition=chemistry(ids[:120]))
    receiver = LocalAcceptance()
    receiver.output = destination/"accepted"
    service_file = folder/"service.npz"
    records = {}
    test_rows = np.concatenate([np.flatnonzero(group==uid) for uid in ids[120:]])
    for chip in chips:
        signal = cycles(chip)
        np.savez(service_file,cycles=signal[test_rows],group=group[test_rows])
        for gas in ("acetone","toluene"):
            for size in (20,80):
                key = f"{chip}-{gas}-{size}"
                req = previous["requests"][key]
                old = json.loads((source_dir/"accepted"/(req+".json")).read_bytes())["payload"]
                identity = digest(old) if expected_id is None else expected_id(key,old)
                if identity!=req or old["配方"]!=ids[120:]:raise ValueError("原预测身份变化")
                if plan is None:
                    variants = {"现场":(ADAPT,CHOICE)}
                    variants.update({name:tuple(p[gas+"-"+str(size)]) for name,p in previous["controls"].items()})
                    recipes = {role:(rule,method,old,method) for role,(rule,method) in variants.items()}
                else:
                    recipes = plan(key,previous,old)
                cache = {}
                info = {}
                for role,(rule,method,receipt,recorded_method) in recipes.items():
                    if (rule,method) not in cache:
                        weight = destination/(key+"-"+role+".joblib")
                        asked,identity = fit_by_request(source_file,signal,group,ids,size,gas,rule,method,weight,folder/"purchased.npz")
                        if asked!=receipt["装机"][rule]:raise ValueError("部署重新询问的标气次序改变")
                        proc = subprocess.run([sys.executable,"-B",str(CLI),"predict","--input",str(service_file),"--artifact",str(weight),
                                               "--sha256",identity["sha256"]],stdout=subprocess.PIPE,check=True,text=True,encoding="utf-8")
                        sent = json.loads(proc.stdout)
                        v = np.array(receipt["浓度"][rule][recorded_method])
                        if (sent["ids"]!=ids[120:] or not np.allclose(v,sent["concentration"],atol=1e-8,rtol=1e-10)
                                or (1+(v>old["阈值"])).tolist()!=sent["class"]):
                            raise ValueError("由原始读数重发后浓度或分类发生变化")
                        cache[rule,method] = {"测量":asked,"权重":weight.name,"摘要":identity["sha256"],"预测":sent}
                    info[role] = cache[rule,method]
                payload = {"阈值":old["阈值"],"上轮":req,"角色":info}
                release = digest([key,payload])
                if not receiver.submit(release,payload)["accepted"]:raise ValueError("未收到完整输出")
                records[key] = release
    score = {}
    for key,req in records.items():
        payload = json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
        if digest([key,payload])!=req:raise ValueError("已交付输出变化")
        gas = key.split("-")[1]
        yy = 1+(np.array([t[gas][valid][np.flatnonzero(group==uid)[0]] for uid in ids[120:]])>payload["阈值"])
        score[key] = {role:{"F1":float(f1_score(yy,v["预测"]["class"],average="macro")),
                            "弱类召回":float(recall_score(yy,v["预测"]["class"],average=None).min())} for role,v in payload["角色"].items()}
    means = {name:float(np.mean([s[name]["F1"] for s in score.values()])) for name in next(iter(score.values()))}
    summary = {name+"F1":m for name,m in means.items()}
    summary["每种策略交付次数"] = len(records)*len(ids[120:])
    atomic_json(destination/"scores.json",{"实发":summary,"scores":score,"requests":records})
    print(json.dumps({"装机询问原始入口":summary},ensure_ascii=False))


if __name__=="__main__":
    with LocalStore(DEST/"lock").exclusive(),TemporaryDirectory(dir=DEST) as temp:
        if (DEST/"scores.json").exists():raise ValueError("已完成本次部署复测")
        run(Path(temp))
