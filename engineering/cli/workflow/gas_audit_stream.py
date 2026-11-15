"""从原始服务批次运行可计概率的抽查、同预算训练、F1估计与发放，答案仅在交接后读取。"""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engineering/src"),str(ROOT)]
from engineering.cli.workflow.gas_residual_stream import (checked, gas_query_stream, gas_returned_errors, gas_response_votes,
    gas_rank_readout, gas_class_reference, field_classification, feedback_query_risk, LocalAcceptance, LocalStore, atomic_json, digest)
from harness.methods import gas_audit_feedback as audit


class BatchLearner:
    def __init__(self, source, classes):
        self.source = source
        self.sx = np.asarray([r["values"] for r in source])
        self.sy = np.asarray([r["label"] for r in source])
        if set(self.sy) != set(range(1,classes+1)):
            raise ValueError("源训练未覆盖声明的类别")
        self.classes = classes
        self.learner = gas_class_reference.ClassReference(self.sx,self.sy)
        self.rank = gas_rank_readout.RankReadout(self.sx,self.sy)
        self.svm = SVC(C=4,gamma="scale",class_weight="balanced",random_state=17).fit(self.learner.source,self.sy)
        self.history, self.inputs = [], []
        self.corrector = gas_returned_errors.ReturnedErrors(classes)

    def predictions(self, x):
        self.inputs.append(x)
        pool = np.vstack(self.inputs)
        scale, base, px = gas_query_stream.train(self.source,self.history,self.classes)
        z = scale.transform(x)
        prob = base.predict_proba(z)
        hy = np.asarray([r["y"] for r in self.history],dtype=int)
        fixed = {k:self.learner.predict(pool,px,hy,x,k).argmax(1)+1 for k in ("原刻度","独立刻度","只用现场")}
        fixed.update(gas_response_votes.kernel_models(self.sx,self.sy,x,px,hy,pool,self.svm))
        bank = self.rank.predict(x,pool,px,hy)
        hard = .4*np.eye(self.classes)[fixed["固定合训核"]-1]
        fixed["现场秩等权与合训核"] = 1+(bank[:,bank.shape[1]//2:].mean(1)+hard).argmax(1)
        fixed["两域秩固定各半与合训核"] = 1+(bank.mean(1)+hard).argmax(1)
        votes = np.column_stack([fixed[k] for k in gas_returned_errors.TOOLS])
        corrected, scores, _ = self.corrector.predict(x,votes)
        return {**fixed,**corrected}, z, prob, votes, scores

    def receive(self,x,votes,y,scores):
        self.corrector.receive(x,votes,y,scores)
        self.history.extend({"x":v.tolist(),"y":int(target)} for v,target in zip(x,y,strict=True))


def run(manifest, out, policy, seed=17, budget=8, through=None):
    root = manifest.parent
    spec = json.loads(manifest.read_bytes())
    classes = spec["classes"]
    source = checked(root/spec["source"],spec["source_sha256"])
    batches = spec["batches"]
    stop = len(batches) if through is None else through
    if policy not in audit.POLICIES or not 0<=stop<=len(batches) or budget<0:
        raise ValueError("抽查作业参数错误")
    deps = (audit,gas_returned_errors,gas_response_votes,gas_rank_readout,gas_class_reference,
            field_classification,feedback_query_risk,gas_query_stream)
    identity = digest([spec,policy,budget,seed,Path(__file__).read_text(encoding="utf8"),
                       *[Path(m.__file__).read_text(encoding="utf8") for m in deps]])
    learner = BatchLearner(source,classes)
    estimate = audit.OfflineScoreEstimate(classes)
    receiver = LocalAcceptance()
    receiver.output = out/"accepted"
    store = LocalStore(out/"checkpoint")
    with store.exclusive():
        state = store.load() or {"identity":identity,"events":[]}
        if state["identity"]!=identity or len(state["events"])>stop:
            raise ValueError("续算状态与当前协议不符")
        known = {r["id"] for r in source}
        if len(known)!=len(source):
            raise ValueError("源数据含重复ID")
        for i,b in enumerate(batches[:stop]):
            rows = checked(root/b["inputs"],b["inputs_sha256"])
            ids = [r["id"] for r in rows]
            if len(set(ids))!=len(ids) or known.intersection(ids) or any(set(r)!={"id","values"} for r in rows):
                raise ValueError("批次身份或字段不符")
            known.update(ids)
            x = np.asarray([r["values"] for r in rows])
            fixed,z,prob,votes,scores = learner.predictions(x)
            choices,f1 = estimate.choices()
            pred = {k:np.asarray(v,dtype=int).tolist() for k,v in fixed.items()}
            pred.update({k:pred[name] for k,name in choices.items()})
            payload = {"ids":ids,"预测":pred,"已见回标":estimate.rows,"采用":choices,"估计F1":f1.tolist()}
            key = digest([identity,payload])
            replay = i<len(state["events"])
            if replay:
                existing = json.loads((receiver.output/(key+".json")).read_bytes())
                if state["events"][i]["request"]!=key or existing["payload"]!=payload:
                    raise ValueError("断点已发预测无法复现")
            elif not receiver.submit(key,payload)["accepted"]:
                raise ValueError("本批预测未接受")
            ix,inverse = (audit.sample(policy,ids,x,z,prob,budget,seed+i) if i+1<len(batches) else ([],np.empty(0)))
            if len(ix)!=(min(len(x),budget) if i+1<len(batches) else 0) or len(set(ix))!=len(ix):
                raise ValueError("抽查超出预算或重复")
            y = np.empty(0,dtype=int)
            if ix:
                returned = checked(root/b["feedback"],b["feedback_sha256"])
                y = np.asarray([returned[ids[j]] for j in ix],dtype=int)
                estimate.update({k:np.asarray(pred[k])[ix] for k in audit.ARMS},y,inverse)
                learner.receive(x[ix],votes[ix],y,scores[ix])
            event = {"request":key,"selected":[ids[j] for j in ix],"labels":y.tolist(),"逆包含概率或单位权":inverse.tolist()}
            if replay:
                if event!=state["events"][i]:
                    raise ValueError("已使用的抽查反馈无法复现")
            else:
                state["events"].append(event)
                store.save(state)
        result = {"identity":identity,"events":state["events"],"returned":estimate.rows}
        atomic_json(out/"run.json",result)
        return result


if __name__=="__main__":
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("--stream",required=True,type=Path)
    cli.add_argument("--output",required=True,type=Path)
    cli.add_argument("--policy",choices=audit.POLICIES,default="混合抽查")
    cli.add_argument("--seed",default=17,type=int)
    cli.add_argument("--budget",default=8,type=int)
    cli.add_argument("--through",type=int)
    args = cli.parse_args()
    run(args.stream,args.output,args.policy,args.seed,args.budget,args.through)
