"""对照作者的批量局部修补与冻结的气体融合模型，采用决定只使用共同确认资料。"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from zipfile import ZipFile

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
from check_gas_patching_start import ROOT, PACKAGE, SOURCE, compile_bridge, dependencies, export_weights, sha
from harness.data.gas_input import read_gas_batches, REGISTERED_SHA256
from harness.evaluation.metrics import macro_f1
from harness.models.gas_views import pair_model
from harness.runtime.storage import atomic_json

PROTOCOL = ROOT / "engineering/configs/experiments/gas-batchwise-patching.json"
SPLIT = ROOT / "results/experiments/official-gas/pilot-11/protocol.json"
FROZEN = ROOT / "engineering/configs/experiments/gas-frozen-future.json"


def author_predict(work, cp, weights, train, labels, inputs):
    """训练文件只收官方训练标签；另一个文件仅含未标记的确认与评价特征。"""
    labeled = work / "train.bin"
    with labeled.open("wb") as stream:
        stream.write(struct.pack(">I", len(train)))
        for row, label in zip(train, labels, strict=True):
            stream.write(np.asarray(row, dtype=">f8").tobytes())
            stream.write(bytes([label]))
    data, output = work / "features.bin", work / "classes.bin"
    data.write_bytes(struct.pack(">I", len(inputs)) + np.asarray(inputs, dtype=">f8").tobytes())
    # Weka 3.7.12 支持用完整类名直接创建学习器；旧版插件扫描依赖 Java 8 的类加载器。
    cmd = ["java", "-Xmx512m", "-XX:ActiveProcessorCount=1", "-Dweka.test.maventest=true",
           "-Dweka.packageManager.loadPackages=false",
           "-cp", cp, "OfficialGasBase",
           str(weights), str(work / "frozen.model"), str(data), str(output), str(labeled)]
    completed = subprocess.run(cmd, cwd=work, capture_output=True, timeout=240)
    log = completed.stdout.decode(errors="replace")
    # 上游某些错误会写 stderr 后继续执行，不能把静默退回原版计为一次有效拟合。
    if completed.returncode or completed.stderr.strip() or " succeeded." not in log:
        raise RuntimeError((completed.stderr.decode(errors="replace") + log)[-4000:])
    times = {k: int(v) for line in log.splitlines() if "_NANOSECONDS=" in line
             for k, v in [line.split("=", 1)]}
    if set(times) != {"FIT_NANOSECONDS", "PREDICT_NANOSECONDS"}:
        raise ValueError("作者入口没有完成一次训练与预测")
    pred = np.frombuffer(output.read_bytes(), dtype="uint8").astype(int) + 1
    if len(pred) != len(inputs) or np.any((pred < 1) | (pred > 6)):
        raise ValueError("作者预测与输入未对齐")
    return pred, times


def score(y, predicted):
    return macro_f1(zip(y, predicted, strict=True))


def run(folder):
    protocol = json.loads(PROTOCOL.read_bytes())
    jars = dependencies(SOURCE.parent.parent / "runtime-dependencies")
    frozen = ROOT / json.loads(FROZEN.read_bytes())["arms"]["double"]["path"]
    if sha(frozen) != protocol["frozen_project_sha256"] or sha(PACKAGE) != protocol["official_sha256"]:
        raise ValueError("参赛模型与冻结身份不符")
    if folder.exists():
        raise FileExistsError("为保护实测回执，输出路径须为新目录")
    sys.path.insert(0, str(PACKAGE))
    from baselines.model import Model
    split = json.loads(SPLIT.read_bytes())
    with ExitStack() as stack:
        work = Path(stack.enter_context(TemporaryDirectory(prefix="gas-patching-compare-")))
        archive = stack.enter_context(ZipFile(PACKAGE))
        cp = compile_bridge(work, jars)
        parent = work / "checkpoint.npz"
        parent.write_bytes(archive.read("checkpoint.npz"))
        base, ours = Model.load(parent), pair_model(Model).load(frozen)
        weights = work / "weights.bin"
        export_weights(parent, weights)
        classes = base.meta["classes"]
        train = [json.loads(line) for line in archive.read("public/train.jsonl").splitlines()]
        valid = {row["id"]: row for line in archive.read("public/validation.jsonl").splitlines()
                 for row in [json.loads(line)]}
        confirm = [sid for batch in split["rounds"].values() for sid in batch["adopt"]]
        if len(set(confirm)) != len(confirm) or len(confirm) != 412 or set(confirm) & set(split["holdout"]):
            raise ValueError("确认集身份或用途与原协议不符")
        raw = ROOT / "data/public/gas-drift/gas-drift.zip"
        selected = protocol["source_batches"]
        batches = {b.batch_number: b.records for b in read_gas_batches(raw, expected_sha256=REGISTERED_SHA256)
                   if b.batch_number in selected}
        features = np.asarray([valid[i]["inputs"]["features"] for i in confirm] +
                              [r.values for n in selected for r in batches[n]])
        author, times = author_predict(work, cp, weights, [r["inputs"]["features"] for r in train],
                                       [classes.index(r["answer"]) for r in train], features)
        preds = {"author": author}
        for key, model in (("static", base), ("project", ours)):
            start = perf_counter_ns()
            preds[key] = np.concatenate([model.predict(features[i:i+128]).argmax(axis=1) + 1
                                         for i in range(0, len(features), 128)])
            times[key + "_predict_ns"] = perf_counter_ns() - start
        nconfirm = len(confirm)
        current = [classes.index(valid[i]["answer"]) + 1 for i in confirm]
        scores = {k: score(current, v[:nconfirm]) for k,v in preds.items()}
        chosen = {k: (k if scores[k] > scores["static"] else "static") for k in ("author","project")}
        folder.mkdir(parents=True)
        issued = folder / "issued.jsonl"
        offset = nconfirm
        with issued.open("x", encoding="utf-8") as stream:
            for number in selected:
                for r in batches[number]:
                    record = {"batch": number, "line": r.source_line,
                              **{k: int(v[offset]) for k,v in preds.items()}}
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                    offset += 1
        if offset != len(features):
            raise ValueError("来源批对应的预测数量不一致")
        # 采用决定和逐条预测已经固定，此后才使用后续批次的答案。
        results = {}
        lines = [json.loads(line) for line in issued.read_bytes().splitlines()]
        for number in selected:
            local = [r for r in lines if r["batch"] == number]
            if [r["line"] for r in local] != [r.source_line for r in batches[number]]:
                raise ValueError("预测行顺序发生改变")
            y = np.array([r.label for r in batches[number]])
            base_ok = np.array([r["static"] for r in local]) == y
            summary = {}
            for key in preds:
                p = np.array([r[key] for r in local])
                summary[key] = {"macro_f1":score(y,p),"repaired":int(np.sum(~base_ok & (p==y))),
                                "broken":int(np.sum(base_ok & (p!=y)))}
            results[number] = {"rows":len(y), "raw":summary,
                               "served":{k:summary[v] for k,v in chosen.items()}}
        totals = {k: {"mean_macro_f1":float(np.mean([v["served"][k]["macro_f1"] for v in results.values()])),
                      "worst_macro_f1":min(v["served"][k]["macro_f1"] for v in results.values())}
                  for k in chosen}
        atomic_json(folder/"result.json", {"protocol_sha256":sha(PROTOCOL), "split_sha256":sha(SPLIT),
                    "source_sha256":REGISTERED_SHA256,"parent_sha256":sha(parent),"project_sha256":sha(frozen),
                    "issued_sha256":sha(issued),"confirmation":scores,"selected":chosen,"batches":results,
                    "aggregate":totals,"times":times,"predict_rows_including_confirmation":len(features),
                    "timing_scope":"Java逐条调用，Python按128条成批；不据该单次时间判速度优胜"})
        print(json.dumps({"selected":chosen,"aggregate":totals},ensure_ascii=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args().output)
