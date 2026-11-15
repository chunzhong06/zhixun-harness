"""取开发时未使用的采集窗口，验证冻结政策实际交出的各轮权重。"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from zipfile import ZipFile

import numpy as np
import requests

from compare_error_slice import BASE,ROOT,ModelArtifacts,_trainer,read,reference_folders,released_checkpoints
from compare_group_objectives import summary
from compare_public_selection import match_bearing_window
from harness.contracts import Sample, digest
from harness.runtime.storage import atomic_json
from unseen_sources import FEMTO,REPO,REV,TEST_UNITS,channel

DEST = BASE/"readout-live"/"fresh-windows"


def fetch_windows(offset=16):
    if offset not in (8,16,24,40,48,54):
        raise ValueError("只提供已登记的波形窗口")
    index=FEMTO/f"offset{offset}-manifest.json"
    old=read(FEMTO/"full-test-manifest.json")
    previous={r["source"] for r in old}
    ends={r["unit"]:r["end"] for r in old}
    if set(ends)!=TEST_UNITS:
        raise ValueError("服务设备集与先前清单不同")
    if offset in (8,24,40,48,54):
        previous.update(read(FEMTO/"offset16-manifest.json")["files"])
    if offset in (24,40,48,54):
        previous.update(read(FEMTO/"offset8-manifest.json")["files"])
    if offset in (40,48,54):
        previous.update(read(FEMTO/"offset24-manifest.json")["files"])
    if offset in (48,54):
        previous.update(read(FEMTO/"offset40-manifest.json")["files"])
    if offset==54:
        previous.update(read(FEMTO/"offset48-manifest.json")["files"])
    paths=[f"Full_Test_Set/{u}/acc_{i:05d}.csv" for u in sorted(TEST_UNITS) for i in range(offset,ends[u],60)]
    if previous.intersection(paths):
        raise ValueError("新测量与开发清单相交")
    if not index.exists():
        tree=requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1",timeout=35)
        tree.raise_for_status()
        listing=tree.json()
        if listing["truncated"] or listing["sha"]!=REV:
            raise ValueError("来源清单被截断或版本不符")
        wanted=set(paths)
        entries={x["path"]:x["sha"] for x in listing["tree"] if x["path"] in wanted}
        if entries.keys()!=wanted:
            raise ValueError("来源中的新窗口不全")
        atomic_json(index,{"revision":REV,"files":entries})
    manifest=read(index)
    if manifest["revision"]!=REV or set(paths)!=manifest["files"].keys():
        raise ValueError("缓存与预注册不符")
    def obtain(path):
        f=FEMTO/path
        def matches(blob):
            return sha1(b"blob "+str(len(blob)).encode()+b"\0"+blob).hexdigest()==manifest["files"][path]
        if f.is_file() and matches(f.read_bytes()):
            return f
        for attempt in range(4):
            try:
                response=requests.get(f"https://raw.githubusercontent.com/{REPO}/{REV}/{path}",timeout=25)
                response.raise_for_status()
                if not matches(response.content):
                    raise ValueError("上游波形文件内容不符")
                f.write_bytes(response.content)
                return f
            except requests.RequestException:
                if attempt==3:
                    raise
                sleep(1+attempt)
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(obtain,paths)),ends


def clock(path):
    with path.open("rb") as stream:
        h,m,s,us=map(float,stream.readline().replace(b";",b",").split(b",")[:4])
    return h*3600+m*60+s+us/1e6


def window_inputs(offset):
    files,ends=fetch_windows(offset)
    rounds=defaultdict(list)
    metadata={}
    for f in files:
        i=int(f.stem.split("_")[1])
        unit=f.parent.name
        key=f"PHM12/{unit}/{i}"
        sep=";" if b";" in f.read_bytes().splitlines()[0] else ","
        xy=np.loadtxt(f,delimiter=sep,usecols=(4,5))
        if xy.shape!=(2560,2) or not np.isfinite(xy).all():
            raise ValueError("未见双轴波形不能按原口径计算")
        values=channel(xy[:,0])+channel(xy[:,1])
        s=Sample(key,1,values,None,"serve",unit,0,unit,(i-1)//6+1,(i-1)//6+1)
        rounds[min((s.time_end-1)//5,9)].append((key,match_bearing_window(s).values))
        metadata[key]=(unit,i,f)
    return rounds,metadata,ends


def window_answers(metadata,ends):
    truth={}
    for key,(unit,i,f) in metadata.items():
        end=ends[unit]
        sec=(clock(f.with_name(f"acc_{end:05d}.csv"))-clock(f))%86400
        if abs(sec-10*(end-i))>.05:
            raise ValueError("新窗口时钟或终点不符")
        truth[key]=sec/60
    return truth


def compare():
    rounds,metadata,ends=window_inputs(16)
    fixed,paper=reference_folders("pronostia","mature")
    paths={"parent":fixed,"project":fixed.parent/"project","mature":paper,"simple":fixed.parent/"simple",
           "bias":BASE/"readout-live"/"pronostia", "first":BASE/"live-first"/"pronostia"}
    streams,weights={},{}
    archive,=(ROOT/"data/official/bearing").glob("*.zip")
    from harness.data.official_rul import ARCHIVE_SHA
    if sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA["bearing"]:
        raise ValueError("官方预测代码版本变化")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),"B") as trainer:
        for arm,folder in paths.items():
            releases=released_checkpoints(folder,arm,True)
            store=ModelArtifacts(folder/"models")
            stream, used={},{}
            for r,records in sorted(rounds.items()):
                ref=releases[arm if r==9 else f"{arm}@{r+1:02}"]
                head,blob=store.read(ref)
                used[r]=head["sha256"]
                p=Path(tmp)/"published.npz"
                p.write_bytes(blob)
                ids,x=zip(*records)
                pred=trainer.Model.load(p).predict(np.asarray(x))
                if not np.isfinite(pred).all() or stream.keys() & set(ids):
                    raise ValueError("新窗口输出无效")
                stream.update(zip(ids,map(float,pred),strict=True))
            streams[arm],weights[arm]=stream,used
    issued=DEST/"issued.json"
    atomic_json(issued,{"revision":REV,"inputs":digest(rounds),"weights":weights,"predictions":streams})
    truth=window_answers(metadata,ends)
    stats=summary("bearing",truth,read(issued)["predictions"],{k:v[0] for k,v in metadata.items()})
    atomic_json(DEST/"scores.json",{"issued_sha256":sha256(issued.read_bytes()).hexdigest(),
                "observations":len(truth),"devices":len(ends),"arms":stats})
    print(len(truth),{k:round(v["score"],3) for k,v in stats.items()},flush=True)


if __name__=="__main__":
    compare()
