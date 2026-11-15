"""把已交付的轴承末版本移到同一新台架，交齐输出后按两种时间口径评分。"""
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_group_objectives import BASE, ROOT, _trainer, atomic_json, digest, metric, official, read, reference_folders, summary
from harness.data.official_bearing import MEASURES
from harness.models.bearing_stumps import stump_model
from harness.models.model_artifacts import ModelArtifacts
from unseen_sources import channel


def ims_inputs():
    source = ROOT/"data/public/ims-bearings"
    manifest = read(source/"sampled.json")
    stamps = manifest["timestamps"]
    selected = sorted(set(range(0,len(stamps),60))|{len(stamps)-1})
    if len(stamps)!=2156 or sorted(manifest["files"])!=[stamps[j] for j in selected]:
        raise ValueError("预定时间抽样发生变化")
    keys, indices, devices, inputs = [], {}, {}, []
    for j in selected:
        name = stamps[j]
        file = source/name
        if sha256(file.read_bytes()).hexdigest()!=manifest["files"][name]:
            raise ValueError("波形文件身份变化")
        a = np.loadtxt(file)
        if a.shape!=(20480,8) or not np.isfinite(a).all():
            raise ValueError("IMS 第一次试验的通道或长度不符")
        for unit in (3,4):
            values = []
            for col in range(2*(unit-1),2*unit):
                v = list(channel(a[:,col],sample_rate=20000))
                v[MEASURES.index("spectral_rms")] *= np.sqrt(10241/16385)
                values.extend(v)
            key = f"IMS/{unit}/{Path(name).name}"
            keys.append(key)
            indices[key],devices[key] = j,str(unit)
            inputs.append(values)
    return manifest,stamps,keys,indices,devices,inputs


def main():
    manifest,stamps,keys,indices,devices,inputs = ims_inputs()
    fixed,paper = reference_folders("pronostia","mature")
    folders = {"current":fixed.parent/"project","fixed":fixed,"mature":paper,"simple":fixed.parent/"simple",
               "bias":BASE/"readout-live/pronostia","stump":BASE/"residual-trees/live/pronostia"}
    package,checksum,_ = official("bearing")
    training_max = max(r.label for r in package.train)
    archive, = (ROOT/"data/official/bearing").glob("*.zip")
    predictions, weights = {}, {}
    out = BASE/"residual-trees/pronostia/ims"
    with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"B") as trainer:
        p = Path(tmp)/"last.npz"
        for arm,folder in folders.items():
            rounds = read(folder/"run.json")["rounds"]
            r = rounds[-1]
            if len(rounds)!=10 or read(folder/r["submitted"])["payload"]["model_id"]!=r["selected"]:
                raise ValueError("末版权重不是源流实际交接的在用版")
            header,blob = ModelArtifacts(folder/"models").read(r["model_artifact"])
            p.write_bytes(blob)
            cls = stump_model(trainer.Model) if arm=="stump" else trainer.Model
            values = cls.load(p).predict(np.asarray(inputs))
            if not np.isfinite(values).all():
                raise ValueError("预测含非有限值")
            predictions[arm] = dict(zip(keys,map(float,values),strict=True))
            weights[arm] = header["sha256"]
    path = out/"issued.json"
    atomic_json(path,{"source":manifest["archive_sha256"],"official":checksum,"weights":weights,
                      "inputs":digest([keys,inputs]),"predictions":predictions})
    saved = read(path)["predictions"]
    ts = [datetime.strptime(Path(p).name,"%Y.%m.%d.%H.%M.%S") for p in stamps]
    elapsed = np.asarray([(b-a).total_seconds()/60 for a,b in zip(ts,ts[1:])])
    # 日历为主结果；扣除大间断时最多按原采集说明计五或十分钟，是运行时长的近似。
    nominal = np.where(np.arange(len(elapsed))<42,5.,10.)
    active = np.r_[np.cumsum(np.minimum(elapsed,nominal)[::-1])[::-1],0.]
    if np.any(elapsed<=0):
        raise ValueError("时间戳没有递增")
    targets = {"calendar":{k:(ts[-1]-ts[j]).total_seconds()/60 for k,j in indices.items()},
               "interval-capped":{k:float(active[j]) for k,j in indices.items()}}
    tables = {}
    for mode,answers in targets.items():
        table = summary("bearing",answers,{"parent":saved["current"],**saved},devices)
        for arm,p in saved.items():
            table[arm]["range"] = [min(p.values()),max(p.values())]
            table[arm]["by_third"] = [metric("bearing",answers,{k:v for k,v in p.items() if indices[k]*3//len(stamps)==j})
                                      for j in range(3)]
            bounds = {}
            for key,limit in (("max_prediction",max(p.values())),("max_training_life",training_max)):
                optimistic = {k:min(limit,max(0.,y)) for k,y in answers.items()}
                floor = metric("bearing",answers,optimistic)
                bounds[key] = {"limit":limit,"ideal_rmse":floor,"mse_ratio":(floor/table[arm]["score"])**2}
            table[arm]["range_diagnostic"] = bounds
            for key,condition in (("within_training",True),("beyond_training",False)):
                subgroup = {k:v for k,v in p.items() if (answers[k]<=training_max)==condition}
                table[arm][key] = {"rows":len(subgroup),"score":metric("bearing",answers,subgroup)}
        tables[mode] = table
        print(mode,{k:round(table[k]["score"],2) for k in saved},flush=True)
    atomic_json(out/"scores.json",{"issued_sha256":sha256(path.read_bytes()).hexdigest(),"suites":tables,
                "first_remaining_minutes":{k:max(a.values()) for k,a in targets.items()},"records":len(keys),
                "largest_gap_minutes":float(elapsed.max()),"timestamp_files":len(ts)})


if __name__=="__main__":
    main()
