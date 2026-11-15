"""评估三个训练与公开验证均未使用的整台轴承，只取预定的原始双轴测量。"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from struct import unpack_from
from tempfile import TemporaryDirectory
from time import sleep
from zipfile import ZipFile,ZIP_DEFLATED
import zlib

import numpy as np
import requests

from compare_fixed_tree_references import OUT,frozen,paths,tree_values
from compare_group_objectives import ROOT,_trainer,atomic_json,official,read,summary
from harness.models.model_artifacts import ModelArtifacts
from unseen_sources import channel


REV="274226336036cd8deb2c40b4ba323af2d11a7290"
SIZE=5447931446
URL=f"https://huggingface.co/datasets/DavidNguyen/XJTU-SY_Bearing_Datasets/resolve/{REV}/XJTU-SY_Bearing_Datasets.zip"
LOCAL=ROOT/"data/public/xjtu-sy"
UNITS={f"Bearing{i}_5" for i in (1,2,3)}


def fetch(start,end):
    for attempt in range(4):
        try:
            response=requests.get(URL,params={"part":start},headers={"Range":f"bytes={start}-{end}"},timeout=35)
            if (response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {start}-{end}/{SIZE}"
                    or len(response.content)!=end-start+1):
                raise ValueError("镜像范围、大小或请求身份不符")
            return response.content
        except (requests.RequestException,ValueError):
            if attempt==3:
                raise
            sleep(attempt+1)


def collect(*, second_offset=False):
    start=SIZE-3_000_000
    class Tail(BytesIO):
        def seek(self,offset,whence=0):
            return super().seek(offset-start if whence==0 else offset,whence)+start
        def tell(self):
            return super().tell()+start
    raw=fetch(start,SIZE-1)
    with ZipFile(Tail(raw)) as z:
        catalog={u:[] for u in UNITS}
        for m in z.infolist():
            p=Path(m.filename)
            if p.suffix.lower()==".csv" and p.parent.name in UNITS:
                catalog[p.parent.name].append(m)
    manifest=LOCAL/("fifth-offset6.json" if second_offset else "fifth-devices.json")
    cache=read(manifest) if manifest.is_file() else None
    items=[]
    ends={}
    for unit,members in sorted(catalog.items()):
        by_seq={int(Path(m.filename).stem):m for m in members}
        end=len(by_seq)
        if not end or set(by_seq)!=set(range(1,end+1)):
            raise ValueError("设备原始测量不连续")
        ends[unit]=end
        sequences=range(6,end,10) if second_offset else sorted(set(range(1,end+1,10))|{end})
        items.extend(by_seq[i] for i in sequences)
    if second_offset:
        original=read(LOCAL/"fifth-devices.json")
        if original["ends"]!=ends or {m.filename for m in items}&{r["member"] for r in original["rows"]}:
            raise ValueError("新测量与历史测量重叠或终点变更")
    if cache:
        if (cache["url"]!=URL or cache["directory_sha256"]!=sha256(raw).hexdigest()
                or [r["member"] for r in cache["rows"]]!=[m.filename for m in items]):
            raise ValueError("已保存特征与冻结的采样原件不符")
        return cache
    def read_member(m):
        local=LOCAL/"raw"/m.filename
        if local.is_file():
            content=local.read_bytes()
            origin="local-raw"
        else:
            first=m.header_offset
            stream=fetch(first,first+30+len(m.filename.encode())+512+m.compress_size-1)
            name,extra=unpack_from("<HH",stream,26)
            if (stream[:4]!=b"PK\x03\x04" or stream[30:30+name].decode()!=m.filename
                    or m.compress_type!=ZIP_DEFLATED or m.flag_bits & 1 or extra>512):
                raise ValueError("成员头或压缩类型不匹配")
            pos=30+name+extra
            content=zlib.decompress(stream[pos:pos+m.compress_size],-15)
            origin="mirror-range"
        if len(content)!=m.file_size or zlib.crc32(content)!=m.CRC:
            raise ValueError("原始波形长度或 CRC 不匹配")
        x=np.loadtxt(BytesIO(content),delimiter=",",skiprows=1)
        if x.shape!=(32768,2) or not np.isfinite(x).all():
            raise ValueError("双轴波形形状或内容无效")
        return {"member":m.filename,"crc32":m.CRC,"source":origin,"sha256":sha256(content).hexdigest(),
                "features":list(channel(x[:,0])+channel(x[:,1]))}
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(read_member,items))
    result={"url":URL,"revision":REV,"repository_lfs_sha256":"3cc815649a315ac7da202980c489f33db44ca2db0317bbe3bcb9dcf415375e10",
            "directory_sha256":sha256(raw).hexdigest(),"ends":ends,"rows":rows}
    atomic_json(manifest,result)
    return result


def main():
    pkg,_,batches=official("bearing")
    if {s.unit_id for s in (*pkg.train,*pkg.validation)}&UNITS:
        raise ValueError("新设备与源包设备相交")
    inputs=collect()
    keys=[r["member"] for r in inputs["rows"]]
    x=np.asarray([r["features"] for r in inputs["rows"]])
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    y_fit=np.asarray([r.label for r in pkg.train if r.sample_id in ids])
    trees,_=frozen("bearing")
    parent,refs=paths("bearing","bearing")
    base=OUT/"teacher-readout/bearing"
    out=base/"new-xjtu"
    model=base/"models/teacher-ridge.npz"
    if sha256(model.read_bytes()).hexdigest()!=read(base/"training.json")["runs"]["teacher-ridge"]["sha256"]:
        raise ValueError("新设备评估权重未冻结")
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B") as trainer:
        path=Path(tmp)/"model.npz"
        path.write_bytes(pkg.checkpoint)
        original=trainer.Model.load(path).predict(x)
        predictions={"original":original,"teacher-ridge":trainer.Model.load(model).predict(x),
                     "tree":tree_values(trees["catboost"],"bearing",x),"median":np.full(len(keys),np.median(y_fit)),
                     "mean":np.full(len(keys),y_fit.mean())}
        predictions["half"]=.5*(original+predictions["teacher-ridge"])
        for arm,folder in {"current":parent,"paper":refs["paper"]}.items():
            r=read(folder/"run.json")["rounds"][-1]
            if read(folder/r["submitted"])["payload"]["model_id"]!=r["selected"]:
                raise ValueError("比较权重不属于源端实际发布版")
            _,blob=ModelArtifacts(folder/"models").read(r["model_artifact"])
            path.write_bytes(blob)
            predictions[arm]=trainer.Model.load(path).predict(x)
    sealed=out/"issued.json"
    atomic_json(sealed,{"feature_source_sha256":sha256((LOCAL/"fifth-devices.json").read_bytes()).hexdigest(),
                        "predictions":{k:dict(zip(keys,v.tolist(),strict=True)) for k,v in predictions.items()}})
    p=read(sealed)["predictions"]
    groups={k:Path(k).parent.name for k in keys}
    truth={k:inputs["ends"][groups[k]]-int(Path(k).stem) for k in keys}
    table=summary("bearing",truth,{"parent":p["current"],**p},groups)
    for arm,r in table.items():
        r["device_equal"]=float(np.sqrt(np.mean(np.square(list(r["group_scores"].values())))))
    atomic_json(out/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":table})
    print({k:[round(r[n],2) for n in ("score","device_equal","worst_group")] for k,r in table.items()})


if __name__=="__main__":
    main()
