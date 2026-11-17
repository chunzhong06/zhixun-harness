"""将输入校正及生效范围单独保存；底层工业权重归原模型维护流程所有。"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from harness.core.contracts import digest
from harness.evaluation.metrics import macro_f1


def signature(x):
    if len(x)==0 or x.ndim!=2 or not np.isfinite(x).all():
        raise ValueError("输入范围需要有限的已到达测量")
    return np.quantile(x, [.25,.5,.75], axis=0)


def distance(a, b):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))


def split_indices(rows):
    units = defaultdict(list)
    for j,r in enumerate(rows):
        units[r["unit_id"]].append(j)
    if len(units)>1:
        keys = sorted(units,key=digest)
        halves = [[j for key in keys[side::2] for j in units[key]] for side in range(2)]
    else:
        order = sorted(range(len(rows)),key=lambda j:digest(rows[j]["sample_id"]))
        halves = [order[::2],order[1::2]]
    return halves


def split_distance(x, rows):
    halves = split_indices(rows)
    if any(not part for part in halves):
        return 0.
    return distance(signature(x[halves[0]]),signature(x[halves[1]]))


def cov_factor(x, exponent):
    if len(x)<2:
        raise ValueError("至少两条观测才可估计协方差")
    eig,vec = np.linalg.eigh(np.cov(x,rowvar=False)+np.eye(x.shape[1]))
    return (vec*eig**exponent)@vec.T


def closest(x, bank):
    """按行算最近参考点；分块限制临时矩阵大小，其他服务行不会影响该行决定。"""
    bank=np.asarray(bank)
    squared=np.einsum("ij,ij->i",bank,bank)
    parts=[]
    for start in range(0,len(x),256):
        chunk=x[start:start+256]
        norms=np.einsum("ij,ij->i",chunk,chunk)[:,None]
        nearest=(norms+squared[None,:]-2*chunk@bank.T).min(axis=1)
        parts.append(np.sqrt(np.maximum(nearest,0)/bank.shape[1]))
    return np.concatenate(parts)


def score(task, labels, predictions):
    if task=="gas":
        return macro_f1(zip(labels,predictions,strict=True))
    return float(np.sqrt(np.mean((np.asarray(labels)-predictions)**2)))


def group_changes(task, rows, before, after):
    groups = defaultdict(list)
    for j,r in enumerate(rows):
        groups[str(r["label"] if task=="gas" else r["unit_id"])].append(j)
    labels = np.array([r["label"] for r in rows])
    if task=="gas":
        return {g:[float(np.mean(p[idx]==labels[idx])) for p in (before,after)] for g,idx in groups.items()}
    return {g:[score(task,labels[idx],p[idx]) for p in (before,after)] for g,idx in groups.items()}


class ScopedCalibration:
    """一批一次提案。无标签时只用上批读数决定旧校正是否到期，之后才接收当前服务输入。"""

    def __init__(self, model, model_sha, task, source_x, policy, bank=None):
        if policy not in ("keep","refresh","expiry","off","row","expiry-row","bank-keep","bank-refresh","bank-expiry",
                          "bank-route","bank-route-refresh"):
            raise ValueError("未知校正使用策略")
        self.model, self.task, self.policy = model,task,policy
        self.bank = bank
        if policy.startswith("bank-") != (bank is not None):
            raise ValueError("训练预设策略须绑定同一份训练目录")
        source = model.normalized(source_x)
        self.center = source.mean(axis=0)
        self.sqrt = cov_factor(source,.5)
        self.identity = digest([sha256(Path(__file__).read_bytes()).hexdigest(),model_sha,task,policy,
                                self.center.tolist(),self.sqrt.tolist(),None if bank is None else bank.identity])

    def predictions(self, layer, x):
        if layer is not None and layer.get("group")!=0:
            z = self.model.normalized(x)
            if layer.get("group")==-1:
                shifted=self.bank.transform_rows(z)
            else:
                shifted=z@np.array(layer["matrix"])+layer["offset"]
            adjusted = shifted*self.model.scale+self.model.mean
            if "bank" in layer:
                inside=closest(z,layer["bank"])<=layer["row_radius"]
                x=np.where(inside[:,None],adjusted,x)
            else:
                x=adjusted
        values = self.model.predict(x)
        if not np.isfinite(values).all() or len(values)!=len(x):
            raise ValueError("工业模型输出无效")
        return values.argmax(axis=1)+1 if self.task=="gas" else values

    def advance(self, state, batch, explore, confirm, serve):
        if state and state["identity"]!=self.identity:
            raise ValueError("校正状态属于另一权重、参照池或策略")
        if not serve or any(r["label"] is not None for r in serve):
            raise ValueError("服务数据需要无答案的观测")
        if {r["sample_id"] for r in explore} & {r["sample_id"] for r in confirm}:
            raise ValueError("探索与确认资料有重合")
        live = state.get("layer")
        x = np.array([r["values"] for r in explore]) if explore else None
        normalized = self.model.normalized(x) if explore else None
        known = signature(normalized) if explore else state.get("previous_inputs")
        event = {"batch":batch,"range_check":None,"expired":False,"proposed":False,"adopted":False,
                 "reference_before":live["origin"] if live else None}
        if live and self.policy in ("refresh","bank-refresh","bank-route-refresh"):
            live = None
        elif live and self.policy=="bank-expiry" and known is not None:
            closest_group = self.bank.match(known)
            event["range_check"] = {"nearest_training_group":closest_group,"adopted_group":live["group"]}
            if closest_group!=live["group"]:
                live=None
                event["expired"]=True
        elif live and self.policy in ("expiry","expiry-row") and known is not None:
            gap = distance(known,live["reference"])
            event["range_check"] = {"distance":gap,"radius":live["radius"]}
            if gap>live["radius"]:
                live = None
                event["expired"] = True
        if len(explore)>=2 and confirm and self.policy!="off":
            if self.policy.startswith("bank-route"):
                trial={"origin":batch,"group":-1}
                event["nominated_group"]=-1
            elif self.bank is not None:
                idx=self.bank.match(known)
                trial={"origin":batch,"group":idx,**self.bank.options[idx]}
                event["nominated_group"]=idx
            else:
                matrix = cov_factor(normalized,-.5)@self.sqrt
                trial = {"origin":batch,"matrix":matrix.tolist(),"offset":(self.center-normalized.mean(axis=0)@matrix).tolist()}
            if self.policy in ("row","expiry-row"):
                a,b=split_indices(explore)
                bank=normalized[a]
                limits=closest(normalized[b],bank)
                trial.update(bank=bank.tolist(),row_radius=float(np.quantile(limits,.95,method="higher")))
            duplicate = self.bank is not None and trial["group"]==(live["group"] if live else 0)
            event["proposed"] = not duplicate
            roles=() if duplicate else (("explore",explore),("confirm",confirm))
            for role,rows in roles:
                values = x if role=="explore" else np.array([r["values"] for r in rows])
                labels = np.array([r["label"] for r in rows])
                pair = [self.predictions(t,values) for t in (live,trial)]
                scores = [score(self.task,labels,p) for p in pair]
                event[role] = scores
                if (scores[1]-scores[0])*(1 if self.task=="gas" else -1)<=1e-12:
                    break
                event[role+"_groups"] = group_changes(self.task,rows,*pair)
                if role=="confirm":
                    if self.bank is None:
                        check = self.model.normalized(values)
                        radius = max(distance(known,signature(check)),split_distance(normalized,explore),split_distance(check,confirm))
                        trial.update(reference=known.tolist(),radius=radius)
                    live = None if trial.get("group")==0 else {**trial,"confirmed_groups":event["confirm_groups"]}
                    event["adopted"] = True
        # 服务行在决定冻结后进入预测。其分布仅能影响以后批次。
        x_serve = np.array([r["values"] for r in serve])
        y = self.predictions(live,x_serve)
        event["using"] = live["origin"] if live else None
        if live and "bank" in live:
            event["rows_back_to_base"]=int(np.sum(closest(self.model.normalized(x_serve),live["bank"])>live["row_radius"]))
        out = {r["sample_id"]:v for r,v in zip(serve,y.tolist(),strict=True)}
        if len(out)!=len(serve):
            raise ValueError("服务行编号重复")
        return {"identity":self.identity,"layer":live,
                "previous_inputs":signature(self.model.normalized(x_serve)).tolist()},event,out


class TrainingCalibrationBank:
    """从训练划分一次构造可部署的变换；窗口和单条测试输入均不能重估这些参数。"""

    def __init__(self, model, fit):
        from sklearn.cluster import KMeans
        x=model.normalized(fit)
        k=max(2,min(8,int(np.sqrt(len(x)/8))))
        partition=KMeans(n_clusters=k,random_state=17,n_init=1,algorithm="lloyd").fit_predict(x)
        pooled=cov_factor(x,.5)
        center=x.mean(axis=0)
        self.signatures=[signature(x)]
        self.centers=[]
        self.options=[{"matrix":np.eye(x.shape[1]).tolist(),"offset":np.zeros(x.shape[1]).tolist()}]
        for label in range(k):
            members=x[partition==label]
            if len(members)<4:
                continue
            matrix=cov_factor(members,-.5)@pooled
            self.centers.append(members.mean(axis=0))
            self.signatures.append(signature(members))
            self.options.append({"matrix":matrix.tolist(),"offset":(center-members.mean(axis=0)@matrix).tolist()})
        self.identity=digest([[s.tolist() for s in self.signatures],self.options,[c.tolist() for c in self.centers]])

    def match(self, observed):
        return min(range(len(self.signatures)),key=lambda g:distance(self.signatures[g],observed))

    def transform_rows(self,x):
        """每行仅由训练簇中心分配一项已冻结变换，可单行调用。"""
        if not self.centers:
            return x
        membership=np.argmin(np.column_stack([np.sum((x-c)**2,axis=1) for c in self.centers]),axis=1)
        output=np.empty_like(x)
        for k in np.unique(membership):
            mask=membership==k
            option=self.options[k+1]
            output[mask]=x[mask]@np.asarray(option["matrix"])+option["offset"]
        return output
