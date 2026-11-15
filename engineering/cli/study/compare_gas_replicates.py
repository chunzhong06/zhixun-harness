"""按真实硬件分组与交错分组建立等维候选，仅用官方 fit 行和当前一条测量。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from zipfile import ZipFile

import numpy as np

from check_gas_concentration import OUT as CONCENTRATION
from compare_calibration_expiry import archived, source_window
from compare_group_objectives import BASE, ROOT, LocalStore, _trainer, atomic_json, delivered, digest, metric, official, read, reference_folders
from harness.data.gas_input import REGISTERED_SHA256, read_gas_batches
from harness.runtime.handoff import LocalAcceptance

OUT = BASE/"gas-replicate-readout"
# 采集论文 CH0–CH15 依次两两同型，第二套八通道重复这四种型号。
MODELS = np.array([[0,1,8,9], [2,3,10,11], [4,5,12,13], [6,7,14,15]])


def view(z, recipe):
    if recipe=="unpooled":
        return z
    groups = MODELS if recipe=="physical" else MODELS.T
    return np.median(z.reshape(-1,16,8)[:,groups,:],axis=2).reshape(len(z),32)


def lookup(train, y, x, probabilities):
    forecasts = []
    norms = np.einsum("ij,ij->i",train,train)
    for start in range(0,len(x),128):
        chunk = x[start:start+128]
        # 每个查询向量的范数对同一行所有距离相同，可以略去。
        ranks = np.argsort(norms-2*chunk@train.T,kind="stable")[:,:5]
        votes = np.eye(6)[y[ranks]].mean(axis=1)
        forecasts.extend((probabilities[start:start+len(chunk)]+votes).argmax(axis=1).tolist())
    return np.array(forecasts)+1


def residual_readout(model, train, labels, query, probabilities, recipe):
    """只用官方训练行拟合六类预测残差。受保护的类别修正包含在父版概率里。"""
    hidden, _ = model.forward(model.normalized(train))
    current = model.predict(train)
    query_hidden, _ = model.forward(model.normalized(query))
    def design(h):
        ones = np.ones((len(h),1))
        return ones if recipe=="bias" else np.column_stack((h,ones))
    bank, inputs = design(hidden),design(query_hidden)
    target = np.eye(6)[labels]-current
    weights = np.linalg.solve(bank.T@bank+len(bank)*0.1*np.eye(bank.shape[1]),bank.T@target)
    return (probabilities+inputs@weights).argmax(axis=1)+1,weights


def compare(*, readout=False, svm=False, ranks=False, kinetic=False):
    out = OUT
    if kinetic:
        out = BASE/"gas-response-ratios"
    elif ranks:
        out = BASE/"gas-sensor-ranks"
    elif svm:
        out = BASE/"gas-source-svm"
    elif readout:
        out = BASE/"gas-readout-ridge"
    pkg, checksum, batches = official("gas")
    old = BASE/"selected/gas"
    fixed,paper = reference_folders("gas","mature")
    samples = read(fixed/"state.json")["samples"]
    fit_ids = {s.sample_id for s in batches[0].samples if s.role=="fit"}
    train = [s for s in pkg.train if s.sample_id in fit_ids]
    if len(train)!=445 or len(fit_ids)!=len(train):
        raise ValueError("候选训练清单与原包不匹配")
    x_train = np.array([r.values for r in train])
    y_train = np.array([r.label-1 for r in train])
    validation = [s for s in samples.values() if s["role"]!="fit"]
    later = [r for b in read_gas_batches(ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256)
             if b.batch_number>=3 for r in b.records]
    ids = [s["sample_id"] for s in validation]+[r.sample_id for r in later]
    if len(set(ids))!=len(ids):
        raise ValueError("候选核验输入重叠")
    x = np.array([s["values"] for s in validation]+[r.values for r in later])
    archive, = (ROOT/"data/official/gas").glob("*.zip")
    with LocalStore(out).exclusive(), TemporaryDirectory(prefix="gas-readout-") as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),"A") as trainer:
        model,identity,records = archived(old,trainer,Path(tmp),"gas")
        cpu = process_time()
        p = model.predict(x)
        common_cpu = process_time()-cpu
        root = dict(zip(ids,(p.argmax(axis=1)+1).tolist(),strict=True))
        z_train,z_x = model.normalized(x_train),model.normalized(x)
        base = read(CONCENTRATION/"issued.json")
        if sha256((CONCENTRATION/"issued.json").read_bytes()).hexdigest()!=read(CONCENTRATION/"scores.json")["issued_sha256"]:
            raise ValueError("浓度核验时的封存参照摘要不符")
        if identity!=base["weights"]["project"] or any(root[r.sample_id]!=base["predictions"]["project"][r.sample_id] for r in later):
            raise ValueError("未复现原末版权重与后八批逐条输出")
        prefix, first_four = {}, []
        for record in records[:4]:
            payload = read(old/record["submitted"])["payload"]
            if payload["model_id"]!=record["selected"]:
                raise ValueError("前缀交接与父模型不符")
            prefix.update(payload["predictions"])
            first_four.append(payload["model_id"])
        outputs,events,cost = {},{},{}
        endpoint = LocalAcceptance()
        endpoint.output = out/"accepted"
        recipes = ("bias","ridge") if readout else ("unpooled","physical","sham")
        if svm or ranks or kinetic:
            from gas_source_svm import classify,KINETIC_VIEWS
            recipes = ("raw-full","raw-gated","shape-full","shape-gated")
            if ranks:
                recipes += ("rank-full","rank-gated")
            if kinetic:
                recipes = tuple(f"{v}-{mode}" for v in KINETIC_VIEWS for mode in ("full","gated"))
        for recipe in recipes:
            cpu = process_time()
            if svm or ranks or kinetic:
                values, evidence = classify(model,x_train,y_train,x,p,recipe)
                cost[recipe] = {"fit_and_predict_cpu_seconds":process_time()-cpu, **evidence}
            elif readout:
                values,weights=residual_readout(model,x_train,y_train,x,p,recipe)
                cost[recipe]={"fit_and_predict_cpu_seconds":process_time()-cpu,"weight_bytes":weights.nbytes,
                              "fit_rows":len(train)}
            else:
                bank = view(z_train,recipe)
                values=lookup(bank,y_train,view(z_x,recipe),p)
                cost[recipe] = {"additional_lookup_cpu_seconds":process_time()-cpu,"bank_bytes":bank.nbytes+y_train.nbytes,
                                "dimensions":bank.shape[1],"fit_rows":len(train)}
            pred = dict(zip(ids,values.tolist(),strict=True))
            selected = False
            stream = dict(prefix)
            decisions = []
            for j in range(4,10):
                explore,confirm,serve = source_window(samples,j)
                if any(r["label"] is not None for r in serve):
                    raise ValueError("服务资料出现答案")
                event = {"round":j+1,"candidate_offered":not selected}
                if not selected:
                    for role,rows in (("explore",explore),("confirm",confirm)):
                        truth = {r["sample_id"]:r["label"] for r in rows}
                        scores = [metric("gas",truth,{i:m[i] for i in truth}) for m in (root,pred)]
                        event[role] = scores
                        if scores[1]<=scores[0]+1e-12:
                            break
                        if role=="confirm":
                            selected = True
                active = pred if selected else root
                batch = {r["sample_id"]:active[r["sample_id"]] for r in serve}
                payload = {"round":j+1,"recipe":recipe,"selected":selected,"predictions":batch}
                key = digest(payload)
                if endpoint.submit(key,payload)["accepted"] is not True or stream.keys()&batch.keys():
                    raise ValueError("逐批输出没有独立交接")
                stream.update(batch)
                decisions.append({**event,"using_candidate":selected,"receipt":key})
            served = {r.sample_id:(pred if selected else root)[r.sample_id] for r in later}
            outputs[recipe] = {"candidate":pred,"public_delivered":stream,"external_delivered":served}
            events[recipe] = decisions
        archive_path = out/"issued.json"
        receipt = {"official_sha256":checksum,"parent_sha256":identity,"fit_ids":digest([r.sample_id for r in train]),
                   "input_sha256":sha256(x.astype("<f8").tobytes()).hexdigest(),"prefix":first_four,
                   "outputs":outputs,"events":events}
        if archive_path.exists() and read(archive_path)!=receipt:
            raise ValueError("同配方未复现上次交接")
        atomic_json(archive_path,receipt)
        public_truth = {s.sample_id:s.label for s in pkg.validation}
        later_truth = {r.sample_id:r.label for r in later}
        public = {a:delivered(folder) for a,folder in (("project",old),("fixed",fixed),("paper",paper))}
        hold = public["project"].keys()
        by_batch = [[r.sample_id for r in b.records] for b in read_gas_batches(
            ROOT/"data/public/gas-drift/gas-drift.zip",expected_sha256=REGISTERED_SHA256) if b.batch_number>=3]
        stats = {}
        for name, o in outputs.items():
            trial = o["candidate"]
            if len(hold)!=414 or o["public_delivered"].keys()!=hold:
                raise ValueError("公开交接未覆盖相同服务清单")
            stats[name] = {"public_delivered":metric("gas",public_truth,o["public_delivered"]),
                           "external_delivered":metric("gas",later_truth,o["external_delivered"]),
                           "public_fixed_candidate":metric("gas",public_truth,{i:trial[i] for i in hold}),
                           "external_fixed_candidate":metric("gas",later_truth,{i:trial[i] for i in later_truth}),
                           "external_by_batch":[metric("gas",later_truth,{i:trial[i] for i in group}) for group in by_batch],
                           "acquired":any(e["using_candidate"] for e in events[name]),**cost[name]}
            if kinetic:
                for key,truth,pred in (("public",public_truth,o["public_delivered"]),("external",later_truth,o["external_delivered"]),
                                       ("candidate",later_truth,{i:trial[i] for i in later_truth})):
                    stats[name][key+"_recall"] = {str(label):float(np.mean([v==label for i,v in pred.items() if truth[i]==label]))
                                                  for label in range(1,7)}
        result = {"issued_sha256":sha256(archive_path.read_bytes()).hexdigest(),"common_parent_cpu_seconds":common_cpu,
                  "public_reference":{a:metric("gas",public_truth,v) for a,v in public.items()},
                  "external_reference":metric("gas",later_truth,{i:root[i] for i in later_truth}),"arms":stats}
        atomic_json(out/"scores.json",result)
        print({k:[round(v[f],5) for f in ("public_delivered","external_delivered","public_fixed_candidate","external_fixed_candidate")]
               for k,v in stats.items()})


if __name__=="__main__":
    from argparse import ArgumentParser
    parser=ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--readout",action="store_true",help="比较常数和隐藏表示的六类残差校正")
    group.add_argument("--svm",action="store_true",help="比较源批 SVM 的分类和分歧纠错")
    group.add_argument("--ranks",action="store_true",help="在相同 SVM 上比较单条传感器相对次序和原有表示")
    group.add_argument("--kinetic",action="store_true",help="比较同一传感器的瞬态与稳态幅值关系")
    args=parser.parse_args()
    compare(readout=args.readout,svm=args.svm,ranks=args.ranks,kinetic=args.kinetic)
