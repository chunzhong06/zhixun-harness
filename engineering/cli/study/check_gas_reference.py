"""用 445 条官方训练测量建立电阻参考，只在封存候选预测之后计算跨批成绩。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches
from harness.data.official_gas import read_official_gas
from harness.evaluation.metrics import macro_f1
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.runtime.storage import atomic_json

BASE = ROOT/"results/experiments/official-gas"
OUT = BASE/"resistance-reference"


def read(path):
    return json.loads(path.read_bytes())


def denominators(x):
    d, q = x[:,:,0], x[:,:,1]-1
    ref = np.divide(d,q,out=np.full_like(d,np.nan),where=np.abs(q)>1e-6)
    return np.where((ref>0)&np.isfinite(ref),ref,np.nan)


def correct(x, reference):
    raw = x.reshape(-1,16,8)
    denom = denominators(raw)
    scale = np.divide(reference,denom,out=np.ones_like(denom),where=np.isfinite(denom))
    transformed = raw.copy()
    transformed[:,:,0] *= scale
    transformed[:,:,2:] *= scale[:,:,None]
    if not np.isfinite(transformed).all():
        raise ValueError("尺度校正产生无效测量")
    return transformed.reshape(-1,128)


def history(folder, sha_field):
    raw = (folder/"issued.jsonl").read_bytes()
    if sha256(raw).hexdigest()!=read(folder/"result.json")[sha_field]:
        raise ValueError("已交预测与实测回执不符")
    records = [r for r in map(json.loads,raw.splitlines()) if r["batch"]!="next"]
    result = {(int(r["batch"]),r.get("line",r.get("id"))):r for r in records}
    if len(result)!=len(records):
        raise ValueError("评价测量编号重复")
    return result


def metrics(y, p):
    classes = set(y)|set(p)
    official = macro_f1(zip(y,p,strict=True))
    return {"official":official,"six":official*len(classes)/6}


def main():
    archive = ROOT/"data/official/gas/cpu-a-guide-20261110-v1.zip"
    if sha256(archive.read_bytes()).hexdigest()!=PACKAGE_SHA:
        raise ValueError("气体起点发生变化")
    package = read_official_gas(archive)
    split = read(BASE/"pilot-11/protocol.json")
    ids = [i for r in split["rounds"].values() for i in r["adopt"]]
    valid = {r.sample_id:r for r in package.validation}
    if len(set(ids))!=412 or set(ids)&set(split["holdout"]):
        raise ValueError("确认集用途或数量不符")
    x = np.array([r.values for r in package.train]).reshape(-1,16,8)
    refs = np.nanmedian(denominators(x),axis=0)
    if not np.isfinite(refs).all():
        raise ValueError("训练资料缺少有效参考电阻")
    sources = [b for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
               if 3<=b.batch_number<=8]
    observations = [r for b in sources for r in b.records]
    sys.path.insert(0,str(archive))
    from baselines.model import Model
    with TemporaryDirectory(prefix="gas-reference-") as folder:
        checkpoint = Path(folder)/"parent.npz"
        checkpoint.write_bytes(package.checkpoint)
        model = Model.load(checkpoint)
        def predict(data):
            return np.concatenate([model.predict(data[j:j+128]).argmax(axis=1)+1 for j in range(0,len(data),128)])
        xc = np.array([valid[i].values for i in ids])
        yc = [valid[i].label for i in ids]
        confirm = {"base":metrics(yc,predict(xc)),"candidate":metrics(yc,predict(correct(xc,refs)))}
        adopted = confirm["candidate"]["official"]>confirm["base"]["official"]
        inputs = np.asarray([r.values for r in observations])
        start = process_time()
        original = predict(inputs).tolist()
        base_cpu = process_time()-start
        start = process_time()
        proposal = predict(correct(inputs,refs)).tolist()
        candidate_cpu = process_time()-start
    keys = [(b.batch_number,r.source_line) for b in sources for r in b.records]
    issued = {"source":REGISTERED_SHA256,"checkpoint":package.checkpoint_sha256,"reference":refs.tolist(),
              "confirmation":confirm,"adopted":adopted,"keys":keys,"original":original,"candidate":proposal}
    path = OUT/"issued.json"
    if path.exists() and read(path)!=json.loads(json.dumps(issued)):
        raise ValueError("重复运行的来源、候选或发布判定发生变化")
    atomic_json(path,issued)
    baseline = history(BASE/"patching-comparison/run-36","issued_sha256")
    libraries = [history(BASE/"subset-direction/measured","issued"),history(BASE/"module-smoke/fusion","issued_sha256")]
    if any(set(h)!=set(keys) for h in [baseline,*libraries]):
        raise ValueError("对照不是同批同测量")
    groups = {}
    counts = dict(shared=0,shared_fixed=0,repaired=0,broken=0)
    offset = 0
    for b in sources:
        y = [r.label for r in b.records]
        n = len(y)
        pred = proposal[offset:offset+n]
        comparison = [baseline[k] for k in keys[offset:offset+n]]
        base = [r["static"] for r in comparison]
        if base!=original[offset:offset+n]:
            raise ValueError("重跑原版与冻结预测不符")
        for j,(truth,old,new) in enumerate(zip(y,base,pred,strict=True)):
            if truth==old:
                counts["broken"]+=int(new!=truth)
                continue
            counts["repaired"]+=int(new==truth)
            k = keys[offset+j]
            options = [comparison[j]["author"],comparison[j]["project"]]
            options.extend(v for h in libraries for name,v in h[k].items() if name not in ("id","batch","static"))
            if truth not in options:
                counts["shared"]+=1
                counts["shared_fixed"]+=int(new==truth)
        groups[b.batch_number] = {name:metrics(y,p) for name,p in {
            "base":base,"candidate":pred,"served":pred if adopted else base,
            "author":[r["author"] for r in comparison],"frozen":[r["project"] for r in comparison]}.items()}
        offset+=n
    if counts["shared"]!=3499 or offset!=8151:
        raise ValueError("共同错误与已核验的资料范围不同")
    aggregate = {name:{metric:{"mean":float(np.mean([g[name][metric] for g in groups.values()])),
                               "worst":min(g[name][metric] for g in groups.values())} for metric in ("official","six")}
                 for name in next(iter(groups.values()))}
    atomic_json(OUT/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"confirmation":confirm,
                "adopted":adopted,"aggregate":aggregate,"source_batches":groups,"candidate_changes":counts,
                "cpu_prediction_seconds":{"base":base_cpu,"candidate_including_correction":candidate_cpu}})
    print(confirm,adopted,counts,aggregate)


if __name__=="__main__":
    main()
