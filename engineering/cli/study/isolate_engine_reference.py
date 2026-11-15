"""对同一组逐轮预测只更换历史寿命参照，分离均值收缩与传感器相似性的作用。"""
from hashlib import sha256
from time import perf_counter

import numpy as np

from compare_current_neighbors import DEST
from compare_error_slice import RAW, SUBSETS, read_cmapss, test_answers
from compare_group_objectives import atomic_json, digest, official, read, remaining_cycles, summary
from harness.models.feedback_neighbors import CurrentNeighbors


class Controls:
    def __init__(self, training, checksum):
        ordered = sorted(training,key=lambda r:(r.unit_id,r.time_end))
        self.lookup = CurrentNeighbors(None,checksum,ordered)
        units, counts = np.unique(self.lookup.units,return_counts=True)
        if set(counts)!={8} or len({(r.unit_id,r.time_end) for r in ordered})!=len(ordered):
            raise ValueError("训练参照应该每台有八个不重复的测点")
        self.n = len(units)
        y = self.lookup.y.reshape(self.n,8)
        self.labels = {"neighbors":y}
        self.labels["only-rank"] = np.broadcast_to(y.mean(axis=0),y.shape)
        self.labels["only-device"] = np.broadcast_to(y.mean(axis=1,keepdims=True),y.shape)
        self.mean = float(y.mean())
        for seed in (17,47,89):
            permutation = np.random.default_rng(seed).permutation(self.n)
            self.labels[f"permuted-{seed}"] = y[permutation]
        self.norms = (self.lookup.bank**2).sum(1)

    def predict(self, inputs, base):
        x = (np.asarray(inputs)[:,:21]-self.lookup.mean)/self.lookup.scale
        query = np.clip(x,-10,10)
        result = {k:np.empty(len(x)) for k in self.labels}
        for left in range(0,len(x),128):
            end = min(left+128,len(x))
            distance = (self.norms-2*query[left:end]@self.lookup.bank.T).reshape(-1,self.n,8)
            closest = np.argmin(distance,axis=2)
            sorted_devices = np.argsort(np.take_along_axis(distance,closest[:,:,None],axis=2)[:,:,0],axis=1,kind="stable")[:,:5]
            within = np.take_along_axis(closest,sorted_devices,axis=1)
            for arm,y in self.labels.items():
                result[arm][left:end] = y[sorted_devices,within].mean(axis=1)
        arms = {k:.5*(base+values) for k,values in result.items()}
        arms["global-mean"] = .5*(base+self.mean)
        arms["current"] = base
        return arms


def score(folder,controls,keys,inputs,current,verified,groups,answers):
    start = perf_counter()
    base = np.array([current[i] for i in keys])
    arms = controls.predict(inputs,base)
    delta = np.max(np.abs(arms["neighbors"]-np.array([verified[i] for i in keys])))
    if delta>1e-7 or len(keys)!=len(set(keys)) or set(keys)!=current.keys():
        raise ValueError(f"真实逐轮近邻未复现 {delta}")
    sealed = folder/"issued.json"
    atomic_json(sealed,{"inputs":digest([keys,inputs]),"training_bank":controls.lookup.identity,
                        "reference_mean":controls.mean,"ids":keys,"predictions":{k:v.tolist() for k,v in arms.items()}})
    seconds = perf_counter()-start
    data = read(sealed)
    streams = {k:dict(zip(keys,v,strict=True)) for k,v in data["predictions"].items()}
    table = summary("engine",answers(),{"parent":current,**streams},groups)
    atomic_json(folder/"scores.json",{"issued_sha256":sha256(sealed.read_bytes()).hexdigest(),"arms":table,
                                      "shared_seconds":seconds})
    print(folder.name,{k:round(v["score"],3) for k,v in table.items()},flush=True)


def main():
    pkg, checksum, batches = official("engine")
    fit_ids = {r.sample_id for r in batches[0].samples if r.role=="fit"}
    controls = Controls([r for r in pkg.train if r.sample_id in fit_ids],checksum)
    root = DEST/"live"
    target = root/"attribution"
    for cohort in ("engine","phm08"):
        folder = root/cohort
        state = read(folder/"regular"/"state.json")
        base = {i:v for batch in state["completed"].values() for i,v in batch["predictions"].items()}
        verified = read(folder/"fixed-half.json")["predictions"]
        keys = list(base)
        rows = [state["samples"][i] for i in keys]
        answers = (lambda:{r.sample_id:r.label for r in pkg.validation}) if cohort=="engine" else remaining_cycles
        score(target/cohort,controls,keys,[r["values"] for r in rows],base,verified,
              {r["sample_id"]:r["unit_id"] for r in rows},answers)
    issued = read(root/"transfer"/"issued.json")
    if issued["early_weights"]!=[.5]*10:
        raise ValueError("外部固定权重与实际来源选择不一致")
    refs = issued["predictions"]
    tracks = {s:read_cmapss(RAW,subset=s,partition="test") for s in SUBSETS}
    for terminal in (False,True):
        inputs, keys, labels = [], [], {}
        for s,units in tracks.items():
            for unit in units:
                cycles = [len(unit.values)] if terminal else [16+20*r for r in range(10) if s!="FD001" and 16+20*r<len(unit.values)]
                for cycle in cycles:
                    key = unit.unit_id if terminal else f"{s}/{unit.unit}/{cycle}"
                    row = unit.values[cycle-1]
                    inputs.append([*row[3:],*row[:3]])
                    keys.append(key)
                    labels[key] = (unit.unit_id,len(unit.values)-cycle)
        def answers():
            ends = {k:v for s,units in tracks.items() for k,v in test_answers(RAW,s,units).items()}
            return {k:ends[unit]+gap for k,(unit,gap) in labels.items()}
        score(target/("terminal" if terminal else "fresh"),controls,keys,inputs,
              {i:refs["current"][i] for i in keys},refs["source-weight-from-one"],
              {k:v[0] for k,v in labels.items()},answers)


if __name__=="__main__":
    main()
