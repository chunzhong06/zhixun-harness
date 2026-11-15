"""对相同类别、相同浓度的真实测量复核批次退分，浓度只在封存预测后用于评分。"""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_group_objectives import BASE, ROOT, LocalStore, _trainer, atomic_json, delivered, digest, official, read, reference_folders
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches
from harness.models.gas_views import pair_model
from harness.models.model_artifacts import ModelArtifacts

OUT = BASE/"gas-concentration"
PUBLIC = ROOT/"data/public/gas-drift"
VOC = PUBLIC/"gas-drift-concentrations-uci270.zip"
VOC_SHA = "98fe3a30981a222dd4518fbcc3dddd45d5c0ce9b03ef6dc6fe5cf7a04cfbff5e"


def concentrations(batches):
    if sha256(VOC.read_bytes()).hexdigest() != VOC_SHA:
        raise ValueError("UCI 270 档案身份不同")
    amounts = {}
    with ZipFile(VOC) as new, ZipFile(PUBLIC/"gas-drift.zip") as old:
        for batch in batches:
            source = old.read(batch.member).splitlines()
            annotated = new.read(Path(batch.member).name).splitlines()
            for r, before, after in zip(batch.records, source, annotated, strict=True):
                label, vector = before.strip().split(None, 1)
                header, extended = after.strip().split(None, 1)
                gas, ppm = header.split(b";")
                if gas != label or r.label != int(gas) or vector != extended:
                    raise ValueError("批内行号相同但类别或 128 维原始测量不同")
                amount = float(ppm)
                if not np.isfinite(amount) or amount < 0 or r.sample_id in amounts:
                    raise ValueError("浓度无效或测量重复")
                amounts[r.sample_id] = amount
    return amounts


def scores(expected, predicted, weights, active):
    matrix = np.zeros((6,6))
    np.add.at(matrix, (expected-1,predicted-1), weights)
    tp = np.diag(matrix)
    total = matrix.sum(axis=1)
    recall = np.divide(tp,total,out=np.zeros(6),where=total>0)
    denom = total+matrix.sum(axis=0)
    f1 = np.divide(2*tp,denom,out=np.zeros(6),where=denom>0)
    return {"macro_f1":float(f1[active].mean()), "balanced_recall":float(recall[active].mean()),
            "recall_by_label":{str(i+1):float(recall[i]) for i in active},
            "confusion":matrix.tolist()}


def common(early, late, amounts, preds, minimum):
    def key(r):
        return r.label, amounts[r.sample_id]
    counts = [Counter(map(key,rows)) for rows in (early, late)]
    cells = sorted(k for k,v in counts[0].items() if v>=minimum and counts[1][k]>=minimum)
    by_label = Counter(label for label,_ in cells)
    report = {"cells":[{"label":k[0],"ppm":k[1],"early":counts[0][k],"late":counts[1][k]} for k in cells],
              "labels":sorted(by_label)}
    if not cells:
        return report
    active = np.array(sorted(by_label))-1
    for name, rows, count in zip(("early","late"),(early, late),counts,strict=True):
        samples = [r for r in rows if key(r) in cells]
        w = np.array([1/(len(active)*by_label[r.label]*count[key(r)]) for r in samples])
        if not np.isclose(w.sum(),1.):
            raise ValueError("标准化权重总和不符")
        y = np.array([r.label for r in samples])
        report[name] = {"rows":len(samples), "scores":{arm:scores(y,np.array([stream[r.sample_id] for r in samples]),w,active)
                                                     for arm,stream in preds.items()}}
    return report


def compare():
    package, official_sha, rounds = official("gas")
    original = BASE/"original/gas"
    old = BASE/"selected/gas"
    _, paper = reference_folders("gas","mature")
    folders = {"original":original,"project":old,"paper":paper}
    batches = list(read_gas_batches(PUBLIC/"gas-drift.zip",expected_sha256=REGISTERED_SHA256))
    official_rows = {r.sample_id:r for r in (*package.train,*package.validation)}
    assigned = {r.sample_id for b in rounds for r in b.samples if r.role=="serve"}
    early, by_source = [], {}
    for b in batches[:2]:
        for r in b.records:
            i = f"gas_b{b.batch_number:02d}_{r.source_line:06d}"
            row = official_rows[i]
            if (r.label, r.values) != (row.label, row.values):
                raise ValueError("官方原包和 UCI 输入或标签不同")
            if i in assigned:
                early.append(r)
                by_source[r.sample_id] = i
    if len(early)!=len(assigned):
        raise ValueError("官方十轮服务记录没有完全对上原始采集行")
    late = [r for b in batches[2:] for r in b.records]
    members = [*early,*late]
    ids = [r.sample_id for r in members]
    features = np.array([r.values for r in members])
    outputs, hashes, issued_checks = {}, {}, {}
    archive, = (ROOT/"data/official/gas").glob("*.zip")
    with LocalStore(OUT).exclusive(), TemporaryDirectory(prefix="gas-concentration-") as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),"A") as trainer:
        previous = BASE/"sensor-pairs/gas/issued.json"
        if sha256(previous.read_bytes()).hexdigest()!=read(previous.with_name("scores.json"))["issued_sha256"]:
            raise ValueError("先前的无反馈预测来源不符")
        compare_late = read(previous)["external_terminal"]["parent"]
        model_type = pair_model(trainer.Model)
        for arm,folder in folders.items():
            record = read(folder/"run.json")["rounds"][-1]
            header,blob = ModelArtifacts(folder/"models").read(record["model_artifact"])
            if read(folder/record["submitted"])["payload"]["model_id"]!=record["selected"]:
                raise ValueError("末版权重没有真实交接")
            path = Path(tmp)/(arm+".npz")
            path.write_bytes(blob)
            prediction = model_type.load(path).predict(features).argmax(axis=1)+1
            out = dict(zip(ids,prediction.tolist(),strict=True))
            live = delivered(folder)
            issued_checks[arm] = {"early_different_from_ten_rounds":sum(out[sid]!=live[oid] for sid,oid in by_source.items()),
                                  "later_different_from_archived_project":sum(out[r.sample_id]!=compare_late[r.sample_id] for r in late)}
            hashes[arm] = header["sha256"]
            outputs[arm] = out
        sealed = OUT/"issued.json"
        receipt = {"package_sha256":official_sha,"source_sha256":REGISTERED_SHA256,"weights":hashes,
                   "input_sha256":sha256(features.astype("<f8").tobytes()).hexdigest(),"predictions":outputs}
        if sealed.exists() and receipt!=read(sealed):
            raise ValueError("既有封存与重新预测不同")
        atomic_json(sealed,receipt)
        amounts = concentrations(batches)
        report = {}
        for name,rows in (("early",early),("late",late),*((str(b.batch_number),b.records) for b in batches[2:])):
            y = np.array([r.label for r in rows])
            by_arm = {}
            for arm,stream in outputs.items():
                pred = np.array([stream[r.sample_id] for r in rows])
                active = np.union1d(y,pred)-1
                by_arm[arm] = scores(y,pred,np.ones(len(y)),active)
            report[name] = {"rows":len(rows),"scores":by_arm}
        main = {str(n):common(early,late,amounts,outputs,n) for n in (1,3,5)}
        by_batch = {str(b.batch_number):common(early,b.records,amounts,outputs,3) for b in batches[2:]}
        atomic_json(OUT/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(), "concentration_sha256":VOC_SHA,
            "matched_archive_rows":len(amounts),"overlap_official_rows":len(official_rows),"raw":report,
            "standardized":main,"individual_later_batches":by_batch,"issued_checks":issued_checks})
        for name in ("early","late"):
            print(name, {a:round(v["macro_f1"],5) for a,v in report[name]["scores"].items()})
        for n,row in main.items():
            print(n,len(row["cells"]),row["labels"],[(row[k]["rows"],round(row[k]["scores"]["project"]["macro_f1"],5)) for k in ("early","late")])


if __name__ == "__main__":
    compare()
