"""独立寿命数据的身份校验、特征转换和设备划分。此模块不读取候选或给策略评分。"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path
from threading import local
import json
import time
from zipfile import ZipFile

import numpy as np
import requests
from harness.contracts import Sample
from harness.data.cmapss_input import condition_key
from harness.data.official_bearing import MEASURES
from harness.runtime.storage import atomic_json

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT / "results/experiments/unseen-agents"
FEMTO = ROOT / "data/public/pronostia-sampled"
REV = "129d43e0aa71e1da992bb4c0f69d54fd990c58cf"
REPO = "Lucky-Loek/ieee-phm-2012-data-challenge-dataset"
PHM08_URL = "https://data.nasa.gov/docs/legacy/Challenge_Data.zip"
PHM08_TRAIN_SHA = "14792b7ec05e3b1e23e2943f4709bb326242b3c9d11dda0b8b686c7a3b7ed27f"
ROLES = {"Bearing1_1":"explore","Bearing2_2":"explore","Bearing1_2":"adopt",
         "Bearing3_1":"adopt","Bearing2_1":"serve","Bearing3_2":"serve"}


TEST_UNITS = {f"Bearing{speed}_{n}" for speed in (1, 2) for n in range(3, 8)} | {"Bearing3_3"}


def fetch_bearing(*, full_test=False):
    FEMTO.mkdir(parents=True,exist_ok=True)
    split = "Full_Test_Set" if full_test else "Learning_set"
    expected = TEST_UNITS if full_test else set(ROLES)
    index = FEMTO/("full-test-manifest.json" if full_test else "manifest.json")
    if index.exists():
        entries = json.loads(index.read_bytes())
    else:
        resp=requests.get(f"https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1",timeout=45)
        resp.raise_for_status()
        tree=resp.json()
        if tree["truncated"]:
            raise ValueError("来源目录不完整")
        units=defaultdict(dict)
        for entry in tree["tree"]:
            parts=entry["path"].split("/")
            if len(parts)==3 and parts[0]==split and parts[1] in expected and parts[2].startswith("acc_"):
                units[parts[1]][int(parts[2][4:-4])]=entry
        if set(units)!=expected:
            raise ValueError("指定设备未全部收录")
        entries=[]
        for uid, records in sorted(units.items()):
            end=max(records)
            if sorted(records)!=list(range(1,end+1)):
                raise ValueError(f"{uid} 源轨迹不连续")
            for i in sorted(set(range(1,end+1,30)) | {end}):
                rec=records[i]
                entries.append({"unit":uid,"index":i,"end":end,"sha":rec["sha"],"source":rec["path"]})
        atomic_json(index,entries)
    clients=local()
    def download(rec):
        dest=FEMTO/rec["source"]
        def correct(data):
            return sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==rec["sha"]
        if dest.exists() and correct(dest.read_bytes()):
            return
        if not hasattr(clients,"client"):
            clients.client=requests.Session()
        for attempt in range(4):
            try:
                resp=clients.client.get(f"https://raw.githubusercontent.com/{REPO}/{REV}/"+rec["source"],timeout=35)
                resp.raise_for_status()
                if not correct(resp.content):
                    raise ValueError("下载内容与固定 Git blob 不符")
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(resp.content)
                return
            except requests.RequestException:
                if attempt==3: raise
                time.sleep(1+attempt)
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(download,entries))
    return entries


def channel(x, *, sample_rate=25600):
    rms=np.sqrt(np.mean(x*x)); peak=np.max(np.abs(x)); mean_abs=np.mean(np.abs(x))
    centered=x-x.mean(); std=centered.std()
    f=np.fft.rfftfreq(len(x),1/sample_rate)
    ps=(np.abs(np.fft.rfft(centered))/len(x))**2
    spectral_dev=ps.std()
    vals={"rms":rms,"peak":peak,"peak_to_peak":np.ptp(x),"mean_abs":mean_abs,
          "kurtosis":np.mean((centered/std)**4),"skewness":np.mean((centered/std)**3),
          "crest_factor":peak/rms,"shape_factor":rms/mean_abs,"impulse_factor":peak/mean_abs,
          "margin_factor":peak/(np.sqrt(np.abs(x)).mean()**2),"spectral_centroid":np.average(f,weights=ps),
          "spectral_rms":np.sqrt(ps.mean()),"spectral_kurtosis":np.mean(((ps-ps.mean())/spectral_dev)**4),
          "dominant_freq":f[ps.argmax()],"band1_energy_0_2k":ps[f<2000].sum(),
          "band2_energy_2k_6k":ps[(f>=2000)&(f<6000)].sum(),"band3_energy_6k_12k":ps[(f>=6000)&(f<=12800)].sum()}
    return tuple(float(vals[k]) for k in MEASURES)


def check_features():
    archive=ROOT/"data/official/bearing/cpu-b-rul-20261110-v1.zip"
    checked=0
    with ZipFile(archive) as z:
        for r in map(json.loads,z.read("public/train.jsonl").splitlines()):
            unit=r["meta"]["bearing"]; i=r["meta"]["snapshot"]
            if unit not in ("Bearing1_1","Bearing1_2","Bearing1_3") or i not in (1,30,80):
                continue
            path=next((ROOT/"data/public/xjtu-sy/raw").glob(f"*/*/{unit}/{i}.csv"))
            x=np.loadtxt(path,delimiter=",",skiprows=1)
            for col,name in enumerate(("horizontal","vertical")):
                actual=channel(x[:,col])
                expected=[r["inputs"]["features"][name][k] for k in MEASURES]
                if not np.allclose(actual,expected,rtol=2e-6,atol=1e-9):
                    raise ValueError(f"官方特征转换不一致 {unit} {i} {name}: {[(k,a,e) for k,a,e in zip(MEASURES,actual,expected) if not np.isclose(a,e,rtol=2e-6,atol=1e-9)]}")
            checked+=1
    if checked!=9: raise ValueError("官方训练资料的转换核对不完整")
    return checked


def bearing_data(*, full_test=False):
    count=check_features()
    entries=fetch_bearing(full_test=full_test)
    units=defaultdict(list)
    for rec in entries:
        i,end=rec["index"],rec["end"]
        if (i-1)%30: continue
        path=FEMTO/rec["source"]
        sep=";" if b";" in path.read_bytes().splitlines()[0] else ","
        xy=np.loadtxt(path,delimiter=sep,usecols=(4,5))
        if xy.shape!=(2560,2): raise ValueError("FEMTO 波形尺寸不符")
        values=channel(xy[:,0])+channel(xy[:,1])
        if not np.isfinite(values).all(): raise ValueError("FEMTO 特征非有限值")
        uid="PHM12/"+rec["unit"]
        role = "serve" if full_test else ROLES[rec["unit"]]
        units[uid].append(Sample(f"{uid}/{i}",1,values,(end-i)/6,role,uid,0,uid,
                                 (i-1)//6+1,(i-1)//6+1))
    return units,{u:rows[0].role for u,rows in units.items()},{"revision":REV,"reference_snapshots":count,
        "sha256_manifest":sha256((FEMTO/("full-test-manifest.json" if full_test else "manifest.json")).read_bytes()).hexdigest(),"snapshot_seconds":0.1,
        "sampling_minutes":5,"source":("FEMTO PHM2012 Full_Test_Set, eleven complete run-to-failure trajectories" if full_test
                                       else "FEMTO PHM2012 Learning_set, six complete run-to-failure trajectories")}


def track_rows(data):
    array=np.loadtxt(data,dtype=np.float64)
    if array.ndim!=2 or array.shape[1]!=26 or not np.isfinite(array).all():
        raise ValueError("发动机输入列不符")
    for uid in np.unique(array[:,0]):
        unit=array[array[:,0]==uid]
        if not np.array_equal(unit[:,1],np.arange(1,len(unit)+1)):
            raise ValueError("设备循环缺失或重复")
        yield int(uid),np.ascontiguousarray(unit[:,2:],dtype="<f8")


def fingerprints(track):
    return {sha256(track[:1].tobytes()).digest(), *(sha256(track[i:i+4].tobytes()).digest() for i in range(len(track)-3))}


def engine_data():
    source=ROOT/"data/public/phm08/train.txt"
    if not source.exists():
        response=requests.get(PHM08_URL,timeout=90)
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as archive:
            data=archive.read("train.txt")
        if sha256(data).hexdigest()!=PHM08_TRAIN_SHA:raise ValueError("新下载的 NASA 训练轨迹摘要不符")
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(data)
    if sha256(source.read_bytes()).hexdigest()!=PHM08_TRAIN_SHA:
        raise ValueError("PHM08 数据摘要不符")
    previous=set()
    with ZipFile(ROOT/"data/public/cmapss/CMAPSSData.zip") as z:
        for entry in z.namelist():
            if entry.startswith(("train_FD","test_FD")):
                with z.open(entry) as stream:
                    for _,t in track_rows(stream): previous.update(fingerprints(t))
    tracks={}; excluded=[]; used=set()
    for uid,t in track_rows(source):
        marks=fingerprints(t)
        if not marks.isdisjoint(previous) or not marks.isdisjoint(used):
            excluded.append(uid);continue
        tracks[uid]=t;used.update(marks)
    if len(tracks)<9: raise ValueError(f"去重后仅 {len(tracks)} 台")
    units={};roles={}
    for j,uid in enumerate(sorted(tracks,key=lambda i:sha256(f"unseen-v1/engine/{i}".encode()).digest())):
        name=f"PHM08/{uid}";t=tracks[uid];role=("explore","adopt","serve")[j%3]
        roles[name]=role
        units[name]=[Sample(f"{name}/{i}",1,tuple(v[3:])+tuple(v[:3]),len(t)-i,role,
                            condition_key(v[:3]),0,name,i,i) for i,v in enumerate(t,1)]
    return units,roles,{"archive":"NASA PHM08 Challenge_Data.zip/train.txt","sha256":sha256(source.read_bytes()).hexdigest(),
                        "source_devices":len(tracks)+len(excluded),"excluded_devices":excluded,"reference_signatures":len(previous)}
