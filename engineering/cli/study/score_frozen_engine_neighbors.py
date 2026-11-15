"""将 PHM08 选择的近邻候选用于真正未参与选型的测点。另读终点与全部同源参照。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_engine_neighbor_updates import DEST
from compare_calibration_expiry import archived
from compare_error_slice import BASE,ROOT,RAW,SUBSETS,_trainer,read_cmapss,test_answers
from compare_group_objectives import atomic_json,digest,official,read,summary
from harness.models.device_neighbors import nearest


def main():
    pkg,checksum,batches=official("engine")
    ids={r.sample_id for r in batches[0].samples if r.role=="fit"}
    fit=[r for r in pkg.train if r.sample_id in ids]
    x,y=np.asarray([r.values for r in fit]),np.asarray([r.label for r in fit])
    units=np.asarray([r.unit_id for r in fit])
    m,scale=x[:,:21].mean(0),np.maximum(x[:,:21].std(0),1e-12)
    bank=np.clip((x[:,:21]-m)/scale,-10,10)
    tracks={s:read_cmapss(RAW,subset=s,partition="test") for s in SUBSETS}
    keys,data,info=[],[],{}
    discovery=read(DEST/"phm08"/"issued.json")
    prior=read(BASE/"live-first"/"cross-stream-offset6"/"issued.json")["predictions"]
    ends=read(BASE/"live-first"/"transfer"/"engine"/"issued.json")["predictions"]
    for subset,ts in tracks.items():
        for t in ts:
            for r in range(11):
                cycle=6+20*r if r<10 else len(t.values)
                if r<10 and (subset=="FD001" or cycle>=len(t.values)):
                    continue
                key=f"{subset}/{t.unit}/{cycle}" if r<10 else t.unit_id
                row=t.values[cycle-1]
                keys.append(key)
                data.append((*row[3:],*row[:3]))
                info[key]=(t.unit_id,subset,r,len(t.values)-cycle)
    if set(keys)!=prior["fixed"].keys()|ends["fixed"].keys() or len(keys)!=len(set(keys)):
        raise ValueError("冻结的新测点与已有基准交接不同")
    data=np.asarray(data)
    _,est=nearest(bank,y,units,np.clip((data[:,:21]-m)/scale,-10,10),np.zeros(len(x)),np.zeros(len(data)))
    archive,=(ROOT/"data/official/engine").glob("*.zip")
    with TemporaryDirectory() as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"C") as trainer:
        p=Path(tmp)/"start.npz"
        p.write_bytes(pkg.checkpoint)
        original=trainer.Model.load(p)
        fourth,identity,_=archived(BASE/"live-first"/"phm08",trainer,Path(tmp),"engine")
        if identity!=discovery["artifacts"]["4"]["parent"]:
            raise ValueError("第4轮实际交接版本不同")
        output={"fixed":{**prior["fixed"],**ends["fixed"]},"current":{**prior["error-first"],**ends["error-first"]},
                "simple":{**prior["simple"],**ends["simple"]},"mature":{**prior["mature"],**ends["mature"]}}
        for step,model in ((0,original),(4,fourth)):
            arm=f"blend-after-{step}"
            hist=discovery["history"][arm]
            use={v["batch"]-1:v["using"] for v in hist}
            base=model.predict(data)
            stream={}
            for i,key in enumerate(keys):
                r=min(9,info[key][2])
                if r<step:
                    stream[key]=output["current"][key]
                else:
                    stream[key]=float(.5*(base[i]+est[i]) if use[r] else base[i])
            output[arm]=stream
    folder=DEST/"phm08"/"transfer"
    sealed=folder/"issued.json"
    atomic_json(sealed,{"source":checksum,"stored_policies":sha256((DEST/"phm08"/"issued.json").read_bytes()).hexdigest(),
                         "inputs":digest([keys,data.tolist()]),"predictions":output})
    saved=read(sealed)["predictions"]
    end_truth={i:y for s,t in tracks.items() for i,y in test_answers(RAW,s,t).items()}
    truth={i:end_truth[uid]+gap for i,(uid,subset,r,gap) in info.items()}
    table={}
    for suite,subset_keys in (("fresh",prior["fixed"]),("terminal",ends["fixed"])):
        armvals={k:{i:vals[i] for i in subset_keys} for k,vals in saved.items()}
        tab=summary("engine",truth,{"parent":armvals["current"],**armvals},{i:v[0] for i,v in info.items()})
        for arm,row in tab.items():
            pred=armvals["current" if arm=="parent" else arm]
            row["subsets"]={s:float(np.sqrt(np.mean([(v-truth[i])**2 for i,v in pred.items() if info[i][1]==s]))) for s in sorted({info[k][1] for k in subset_keys})}
        table[suite]=tab
        print(suite,len(subset_keys),{k:round(v["score"],3) for k,v in tab.items()})
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"suites":table})


if __name__=="__main__":
    main()
