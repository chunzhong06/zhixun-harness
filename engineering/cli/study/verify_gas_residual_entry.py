"""真实运行四个服务流的逐批命令，核对每一次交付及下一批可用的八份反馈。"""
import json
import subprocess
import sys

import numpy as np
from score_gas_returned_errors import BASE, DEST as PRIOR, QUERY_RUNS, STREAMS, digest, LocalStore, atomic_json, f1_score, recall_score, read
from score_gas_joint_query import ROOT

DEST = PRIOR.parent / "gas-residual-entry"


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("正式入口已复测")
        results = {}
        for name, path in STREAMS.items():
            folder = DEST/name
            manifest = BASE/path/"stream.json"
            old = PRIOR/name/"均匀覆盖"
            prev = read(old/"run.json")
            qdir = QUERY_RUNS/name/"均匀覆盖"/"17"
            qrun = read(qdir/"run.json")
            args = [sys.executable, "-B", str(ROOT/"engineering/cli/workflow/gas_residual_stream.py"),
                    "--stream", str(manifest), "--output", str(folder), "--budget", "8", "--seed", "17"]
            if name == "换件":
                subprocess.run([*args,"--through","4"], check=True, cwd=ROOT)
            subprocess.run(args, check=True, cwd=ROOT)
            issued = read(folder/"run.json")
            truth = read(manifest.parent/"evaluation.json")
            seen, values = [], []
            for event,key,q in zip(issued["events"], prev["requests"], qrun["accepted"], strict=True):
                before = read(old/"accepted"/(key+".json"))["payload"]
                p = read(folder/"accepted"/(event["request"]+".json"))["payload"]
                if (digest([issued["identity"],p]) != event["request"] or digest([prev["identity"],before]) != key or
                        event["selected"] != q["selected"] or p["prediction"] != before["预测"]["相近读数残差"] or
                        p["ids"] != before["ids"] or p["已见回标"] != before["已见回标"]):
                    raise ValueError("实际逐批入口的交接、询问或纠错结果与研究记录不同")
                seen.extend(p["ids"])
                values.extend(p["prediction"])
            if len(set(seen)) != len(seen) or set(seen) != set(truth):
                raise ValueError("服务评测未覆盖最后一批或有重复")
            y = [truth[i] for i in seen]
            results[name] = {"F1":float(f1_score(y,values,average="macro")),
                             "最差召回":float(recall_score(y,values,average=None).min()),
                             "反馈":issued["returned"],"实际预测数":len(y),"批次":len(issued["events"])}
        total = float(np.mean([v["F1"] for v in results.values()]))
        atomic_json(DEST/"scores.json",{"streams":results,"F1":total})
        print(json.dumps({"命令实测":results,"F1":total},ensure_ascii=False))


if __name__ == "__main__":
    run()
