"""给独立验证准备论文适配。显式隔离源端确认机，所有评价寿命留到交接后。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import h5py
import numpy as np

from score_flight_load_neighbors import DEST as STUDY, SITES, dataset_path, signals, read_flights, batch_sequence, call_paper, score
from harness.contracts import digest
from harness.runtime.handoff import LocalAcceptance
from harness.runtime.storage import LocalStore, atomic_json
from score_ncmapss_field import file_hash

DEST = STUDY/"paper"


def main():
    with LocalStore(DEST/"lock").exclusive(), TemporaryDirectory(dir=DEST) as scratch:
        if (DEST/"scores.json").exists():
            raise ValueError("已有论文实测")
        seq,ys,early,valid=[],[],[],[]
        for site in SITES:
            with h5py.File(dataset_path(site)) as f:
                rows=signals(f,"dev")
                y=read_flights(f,"dev",answers=True)
                held=set(sorted({u for u,_ in rows})[::5])
                for unit in sorted({u for u,_ in rows}):
                    for i,k in enumerate(sorted(k for k in rows if k[0]==unit)):
                        early.append(i<8)
                        valid.append(unit in held)
                        ys.append(y[k])
                        seq.append(batch_sequence(rows[k][1]))
        work=Path(scratch)
        np.savez(work/"source.npz",x=np.stack(seq),y=ys,early=early,valid=valid)
        file=DEST/"paper.pt"
        call_paper(["train",str(work/"source.npz"),str(file)])
        receiver=LocalAcceptance()
        receiver.output=DEST/"accepted"
        seq, site_keys = [],{}
        for site in SITES:
            with h5py.File(dataset_path(site)) as f:
                rows=signals(f,"test")
                keys=sorted(rows)
                site_keys[site]=keys
                seq.extend(batch_sequence(rows[k][1]) for k in keys)
        np.save(work/"inputs.npy",np.stack(seq))
        call_paper(["predict",str(work/"inputs.npy"),str(work/"pred.npy"),str(file)])
        values=np.load(work/"pred.npy")
        pos,requests=0,{}
        for site,keys in site_keys.items():
            p=values[pos:pos+len(keys)]
            pos+=len(keys)
            payload={"ids":[list(k) for k in keys],"predictions":{"Darrah双向序列本地适配":p.tolist()},
                     "model":file_hash(file)}
            key=digest([site,payload])
            ack=receiver.submit(key,payload)
            if not ack["accepted"] or ack["request_id"]!=key:
                raise ValueError("论文预测未发出")
            requests[site]=key
        results={}
        for site,key in requests.items():
            with h5py.File(dataset_path(site)) as f:
                results[site]=score(site,receiver,key,f)
        atomic_json(DEST/"scores.json",{"model":file_hash(file),"sites":results})
        print(json.dumps({"训练飞行":len(ys),"内部确认飞行":sum(valid)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
