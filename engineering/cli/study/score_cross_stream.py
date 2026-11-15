"""维护 PHM08 时将当轮交付权重用于另一批发动机；保存全部预测后才读其寿命答案。"""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_error_slice import BASE, ROOT, ModelArtifacts, RAW, _trainer, read_cmapss, released_checkpoints, test_answers
from compare_proposal_frequency import OUTPUT, atomic_json, digest, metric, read, reference_folders
from engineering.cli.study.run_rul_maintenance import PACKAGES

DEST = BASE/"live-first"/"cross-stream"


def compare(paired=False,offset=1):
    if offset not in (1,6,11,16):
        raise ValueError("采样偏移未登记")
    fixed, paper = reference_folders("phm08", "mature")
    paths = {"fixed":fixed, "simple":fixed.parent/"simple", "mature":paper,
             "paper-first":OUTPUT/"phm08"/"mature", "error-first":BASE/"live-first"/"phm08"}
    out = DEST
    if paired:
        paths.update({k:BASE/"trajectory-training"/k/"phm08" for k in ("trajectory-order","trajectory-difference")})
        out = BASE/"trajectory-training"/"cross-stream"
    if offset!=1:
        out=out.with_name(out.name+f"-offset{offset}")
    tracks = {s:read_cmapss(RAW, subset=s, partition="test") for s in ("FD002","FD003","FD004")}
    groups, rounds = {}, defaultdict(list)
    for subset, units in tracks.items():
        for t in units:
            for r in range(10):
                cycle = offset+20*r
                if cycle < len(t.values):
                    key = f"{subset}/{t.unit}/{cycle}"
                    groups[key] = (subset,t.unit_id,r,len(t.values)-cycle)
                    row = t.values[cycle-1]
                    rounds[r].append((key, (*row[3:],*row[:3])))
    if offset!=1:
        earlier=read(DEST/"issued.json")["predictions"]["fixed"]
        if groups.keys() & earlier.keys():
            raise ValueError("新增测点进入过原计分清单")
        if offset==6 and groups.keys() & read(DEST.with_name("cross-stream-offset11")/"issued.json")["predictions"]["fixed"].keys():
            raise ValueError("与上一组错位测点重合")
        if offset==16:
            for prior in (6,11):
                if groups.keys() & read(DEST.with_name(f"cross-stream-offset{prior}")/"issued.json")["predictions"]["fixed"].keys():
                    raise ValueError("与此前冻结测点重合")
    fname, expected, _ = PACKAGES["engine"]
    source = ROOT/"data/official/engine"/fname
    if sha256(source.read_bytes()).hexdigest() != expected or set(rounds) != set(range(10)):
        raise ValueError("原包或预注册的发放轮数不符")
    streams, provenance = {}, {}
    with TemporaryDirectory() as tmp, ZipFile(source) as pack, _trainer(pack,Path(tmp),"C") as trainer:
        for arm,folder in paths.items():
            releases = released_checkpoints(folder, arm, True)
            store = ModelArtifacts(folder/"models")
            stream, weights = {}, []
            for r in range(10):
                ref = releases[arm if r==9 else f"{arm}@{r+1:02}"]
                info, blob = store.read(ref)
                weights.append(info["sha256"])
                checkpoint = Path(tmp)/"checkpoint.npz"
                checkpoint.write_bytes(blob)
                ids, data = zip(*rounds[r])
                pred = trainer.Model.load(checkpoint).predict(np.asarray(data))
                if pred.shape != (len(ids),) or not np.isfinite(pred).all():
                    raise ValueError("服务输出数值或长度不符")
                stream.update(zip(ids, map(float,pred), strict=True))
            if stream.keys() != groups.keys():
                raise ValueError("目标服务清单不符")
            streams[arm], provenance[arm] = stream, weights
    issued = out/"issued.json"
    atomic_json(issued, {"inputs":digest(rounds), "checkpoints":provenance, "predictions":streams})
    streams = read(issued)["predictions"]
    end = {s:test_answers(RAW,s,units) for s,units in tracks.items()}
    truth = {k:end[s][uid]+gap for k,(s,uid,_,gap) in groups.items()}
    metrics = {}
    for arm,stream in streams.items():
        subsets, devices, windows = defaultdict(dict), defaultdict(dict), defaultdict(dict)
        for k,v in stream.items():
            subset,unit,r,_ = groups[k]
            subsets[subset][k], devices[unit][k], windows[r][k] = v,v,v
        metrics[arm] = {"score":metric("engine",truth,stream),
                        "subsets":{s:metric("engine",truth,p) for s,p in subsets.items()},
                        "devices":{u:metric("engine",truth,p) for u,p in devices.items()},
                        "rounds":[metric("engine",truth,windows[r]) for r in range(10)]}
    atomic_json(out/"scores.json", {"issued_sha256":sha256(issued.read_bytes()).hexdigest(),
                "devices":len(set(v[1] for v in groups.values())), "observations":len(groups), "arms":metrics})
    print({a:round(s["score"],3) for a,s in metrics.items()},flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--offset", type=int, choices=(1,6,11,16), default=1)
    args=parser.parse_args()
    compare(args.paired,args.offset)
