"""给冻结模型做现场偏差校准，用独立已失效设备确认，再在运行中的其他设备交付。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from compare_fixed_tree_references import OUT, frozen, tree_values
from compare_group_objectives import ROOT, _trainer, atomic_json, official, read, summary
from harness.models.source_combinations import SourceCombinations
from score_completed_choice import femto, forecast, xjtu
from score_unseen_xjtu_devices import LOCAL
from unseen_sources import PHM08_TRAIN_SHA, RESULT, track_rows


PROTOCOL = ROOT/"engineering/configs/experiments/completed-residual-transfer.json"
DEST = OUT/"completed-residual"


def engines():
    raw = ROOT/"data/public/phm08/train.txt"
    if sha256(raw.read_bytes()).hexdigest() != PHM08_TRAIN_SHA:
        raise ValueError("发动机数据与冻结原档不同")
    roles = read(RESULT/"issued/engine/study.json")["roles"]
    ref, history, service, service_ends = {}, [], [], {}
    for uid, track in track_rows(raw):
        unit = f"PHM08/{uid}"
        if unit not in roles:
            continue
        n = len(track)
        x = np.column_stack((track[:,3:],track[:,:3]))
        if roles[unit] == "serve":
            service_ends[unit] = n
            service.extend((f"{unit}/{i+1}",unit,i+1,v) for i,v in enumerate(x))
        else:
            ref[unit] = n
            for j in np.linspace(0,n-1,5).round().astype(int):
                history.append((unit,f"{unit}/{j+1}",x[j],n-j-1))
    if len(ref)!=146 or len(service_ends)!=72:
        raise ValueError("失效资料与服务设备数量变化")
    def truth():
        return {k:service_ends[u]-cycle for k,u,cycle,_ in service}
    return ref,history,service,truth,[raw]


def issue(name, data, forecast, identity):
    ends, samples, service, truth, paths = data
    service.sort(key=lambda r:(r[2],r[0]))
    owners = {k:u for k,u,_,_ in service}
    if set(owners.values()) & ends.keys() or len(owners)!=len(service):
        raise ValueError("回标与服务重叠或服务重复")
    ordered = sorted(ends,key=lambda u:(ends[u],u))
    bank = {u:[r for r in samples if r[0]==u] for u in ordered}
    pred = forecast(np.asarray([r[3] for r in service]))
    arms = {a:{} for a in (*pred,"always-fit","confirmed")}
    pos = 0
    means = []
    check = []
    shift = 0.
    approved = False
    events = []
    issued_choices = {}
    for i,(key,_,now,_) in enumerate(service):
        released = []
        while pos<len(ordered) and ends[ordered[pos]]<now:
            u = ordered[pos]
            group = bank[u]
            if len(group)!=5:
                raise ValueError("完成设备没有对应五个定位记录")
            p = forecast(np.asarray([r[2] for r in group]))["base"]
            y = np.asarray([r[3] for r in group])
            role = "fit" if pos%2==0 else "confirm"
            if role=="fit":
                means.append(float(np.mean(y-p)))
            else:
                check.append((p,y))
            released.append({"unit":u,"role":role,"end":ends[u],"ids":[r[1] for r in group]})
            pos+=1
        if released:
            shift = float(np.mean(means))
            errors = None
            if check:
                errors = [float(np.mean([np.mean((p-y)**2) for p,y in check])),
                          float(np.mean([np.mean((np.maximum(0,p+shift)-y)**2) for p,y in check]))]
                approved = errors[1]<errors[0]
            events.append({"time":now,"released":released,"offset":shift,"confirm_mse":errors,"accept":approved})
        for a,v in pred.items():
            arms[a][key] = float(v[i])
        raw = arms["base"][key]
        corrected = max(0,raw+shift) if pos else raw
        arms["always-fit"][key] = corrected
        arms["confirmed"][key] = corrected if approved else raw
        issued_choices[key] = [now,approved]
    path = DEST/name/"issued.json"
    atomic_json(path,{"models":identity,"source":{p.name:sha256(p.read_bytes()).hexdigest() for p in paths},
                      "protocol_sha256":sha256(PROTOCOL.read_bytes()).hexdigest(),"events":events,"clock":issued_choices,
                      "owners":owners,"predictions":arms})
    p = read(path)["predictions"]
    table = summary("engine" if name=="phm08" else "bearing",truth(),{"parent":p["base"],**p},owners)
    del table["parent"]
    gate = (table["confirmed"]["score"]<=.99*min(table[a]["score"] for a in ("base","always-fit"))
            and table["confirmed"]["worst_group"]<=1.01*table["base"]["worst_group"])
    accepted = sum(v[1] for v in issued_choices.values())
    result = {"arms":table,"meets_gate":gate,"rows":len(service),"devices":len(set(owners.values())),
              "feedback_labels":5*pos,"first_feedback":events[0]["time"] if events else None,
              "corrected_rows":accepted,"issued_sha256":sha256(path.read_bytes()).hexdigest()}
    atomic_json(path.with_name("scores.json"),result)
    print(name,result["rows"],accepted,{a:round(v["score"],3) for a,v in table.items()},flush=True)


def main():
    for task in ("bearing","engine"):
        pkg, checksum, batches = official(task)
        fit = [r for r in batches[0].samples if r.role=="fit"]
        tree, receipts = frozen(task)
        archive, = (ROOT/"data/official"/task).glob("*.zip")
        with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),"B" if task=="bearing" else "C") as trainer:
            root = Path(tmp)/"original.npz"
            root.write_bytes(z.read("checkpoint.npz"))
            original = trainer.Model.load(root)
            identity={"package":checksum,"tree":receipts["catboost"]["sha256"]}
            if task=="bearing":
                median=float(np.median([r.label for r in fit]))
                def predict(x):
                    p=forecast(x,tree["catboost"],median,original)
                    return {"base":p["phase"],"median":p["median"],"tree":p["tree"],"original":p["original"]}
                issue("femto",femto(48),predict,identity)
                issue("xjtu",xjtu(pkg,LOCAL/"fifth-offset6.json"),predict,identity)
            else:
                repair=SourceCombinations([r.values for r in fit])
                identity["input_restore"]=repair.identity
                def predict(x):
                    mapped=repair.transform(x)
                    return {"base":original.predict(mapped),"original":original.predict(x),
                            "tree":tree_values(tree["catboost"],"engine",mapped)}
                issue("phm08",engines(),predict,identity)


if __name__=="__main__":
    main()
