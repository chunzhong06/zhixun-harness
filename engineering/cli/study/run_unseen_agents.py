"""逐轮运行四种陌生数据维护策略，服务答案只在全部预测封存后用于评分。"""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import sleep

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from engineering.cli.study.run_luna_rul_comparison import Replay, LocalStore, digest, frozen, np, read, rmse
from harness.contracts import BatchContext
from harness.methods.planning.rul_continuation import device_biases
from unseen_sources import RESULT, bearing_data, engine_data, MEASURES

DRIVERS=("project","mature","simple")
ARMS=("static","fixed",*DRIVERS)
ADDITIONAL=("project-frozen","mature-live")
PRIMARY=("static","fixed","project-frozen","mature-live","simple")
PROMPTS={
    "project":"遵循既有项目的维护流程。按设备检查近期偏差与上轮反馈，在当前父模型上优先比较项目的均衡续训、设备均衡及周期清单；证据不足可保持。给出本轮可执行提案。",
    "mature":"依据 Self-Healing ML 的诊断与选样办法，观察分布和错误，提出能检验原因的训练样本筛选条件。可以比较全量均衡训练，或保持。不接触服务设备的答案。",
    "simple":"你的目标是降低后续设备的寿命预测 RMSE。你可以直接用 Python 分析提供的数据，自行决定实验。可用工具及预算见本次请求。"}


class ExternalReplay(Replay):
    driven_arms=(*DRIVERS,*ADDITIONAL)
    record_invalid_actions=True

    def __init__(self,task,stack):
        # 父类初始化的协议描述旧训练池。新实验的轨迹、角色、预算另存在 study.json。
        super().__init__(task, RESULT/"runtime", stack)
        units,assignment,source=(engine_data if task=="engine" else bearing_data)()
        self.truth={}
        shards=[[] for _ in range(10)]
        width=20 if task=="engine" else 5
        for uid,rows in units.items():
            role=assignment[uid]
            if role=="serve":
                for row in rows:
                    phase=min((row.time_end-1)//width,9)
                    self.truth[row.sample_id]=row
                    shards[phase].append(replace(row,label=None,available_at=phase))
                continue
            for phase in range(10):
                section=rows[phase*len(rows)//10:(phase+1)*len(rows)//10]
                count=min(8,len(section))
                for j in range(count):
                    r=section[j*(len(section)-1)//max(1,count-1)]
                    shards[phase].append(replace(r,available_at=phase))
        self.batches=[BatchContext(task,f"external-{n+1}",n,tuple(rows)) for n,rows in enumerate(shards)]
        for n in range(1,11):
            if not all(self.roles(n)[r] for r in ("explore","adopt","serve")):raise ValueError("出现空角色分片")
        self.study=RESULT/"issued"/task
        self.study.mkdir(parents=True,exist_ok=True)
        self.model_path(self.original).parent.mkdir(exist_ok=True)
        if not self.model_path(self.original).exists():self.model_path(self.original).write_bytes(self.package.checkpoint)
        self.source=source
        frozen(self.study/"study.json",{"source":source,"roles":assignment,"initial_checkpoint":self.original,
                                       "arms":list(ARMS),"rounds":10,"service_rows":len(self.truth),
                                       "service_ids_sha256":digest(sorted(self.truth)),"role_prompts":PROMPTS})

    def request_path(self,kind,arm,number):
        return RESULT/"drivers"/arm/kind/self.task/f"{number:02d}.json"

    def file_lines(self,split,rows,path):
        if split=="train":
            return super().file_lines(split,rows,path)
        if any(r.role!="explore" or r.label is None for r in rows):
            raise ValueError("训练器只允许探索资料参与回测")
        with path.open("w",encoding="utf-8") as stream:
            for r in rows:
                if self.task=="engine":
                    inputs={"sensors":list(r.values[:21]),"settings":list(r.values[21:])}
                else:
                    inputs={"features":{c:dict(zip(MEASURES,r.values[i*17:(i+1)*17],strict=True))
                                        for i,c in enumerate(("horizontal","vertical"))}}
                stream.write(json.dumps({"id":r.sample_id,"inputs":inputs,"answer":r.label})+"\n")

    def request(self,arm,number,parent,prior):
        rows=self.roles(number)["explore"]
        predictions=self.predict(parent,rows)
        err=predictions-np.asarray([r.label for r in rows])
        splits=self.diagnostics(rows,err)
        by_device=device_biases(rows,dict(zip([r.sample_id for r in rows],predictions.tolist(),strict=True)))
        raw=RESULT/"drivers"/arm/"observations"/self.task/f"{number:02d}.json"
        observations={"columns":self.feature_names,"explore":[{"unit":r.unit_id,"x":[r.time_end,*r.values],
                      "truth":r.label,"prediction":p} for r,p in zip(rows,predictions.tolist(),strict=True)]}
        frozen(raw,observations)
        train=raw.parent/"train.json"
        if not train.exists():
            frozen(train,{"columns":self.feature_names,"fit":[{"unit":r.unit_id,"x":[r.time_end,*r.values],
                    "truth":r.label} for r in self.fit]})
        policy={"mature-live":"mature","project-frozen":"project"}.get(arm,arm)
        request={"task":self.task,"round":number,"role":PROMPTS[policy],"parent_sha256":parent,
                 "objective":"minimize future service pooled RMSE","units":self.package.task.target_unit,
                 "explore_rows":len(rows),"rmse":rmse(err),"mean_signed_error":float(err.mean()),
                 "bias":"prediction minus truth","device_bias":by_device[:3],"feature_splits":splits,
                 "previous":prior,"legal_data":[str(train),str(raw)],"kinds":["balanced",*self.extra,"include","exclude"],
                 "budget":{"proposals":1,"max_candidates":2,"max_rows_each":len(self.fit),"max_epochs_each":1},
                 "tools":"balanced uses all fit rows; device-balanced and cycle-window if listed are the stored project lists. include/exclude takes field from feature_splits, op le or gt, exact threshold from that field. Keep with actions []. All arms have identical access to these tools and may use Python to read the two legal_data files.",
                 "selection":"Best exploration RMSE must improve the incumbent and improve independent confirmation RMSE before serving. Serve labels hidden.",
                 "reply":{"observation_key":"copy this key","actions":"list up to two objects, e.g. {kind:balanced} or {kind:include,field:time,op:le,threshold:number}","reason":"short hypothesis"}}
        if arm=="project-frozen":
            request["kinds"]=["balanced",*self.extra]
            request["tools"]="项目冻结工具：balanced 全池；发动机另有 device-balanced（每台七条）及 cycle-window（81–161 周期替换）。[] 为保持。可以分析 legal_data。本支保持上次项目实现的动作范围。"
        if arm=="mature-live":
            request["role"]+=" 每轮由模型根据当前反馈亲自提案。"
        request["observation_key"]=digest(request)
        frozen(self.request_path("requests",arm,number),request)
        return request

    def score_external(self,arms=ARMS,folder="scores"):
        if not all(self.state_path(a,10).exists() for a in arms):
            raise ValueError("各组尚未交齐十轮预测")
        summary={}
        for arm in arms:
            parent=self.original;seen=set();errors=[];devices=defaultdict(list);wins=[];trials=[];adopted=[]
            for n in range(1,11):
                s=read(self.state_path(arm,n));p=s["predictions"]
                if set(p)!={r.sample_id for r in self.roles(n)["serve"]} or seen.intersection(p) or s["parent"]!=parent:
                    raise ValueError("服务编号或接续权重出错")
                if arm in self.driven_arms and (read(self.request_path("responses",arm,n))!=s["decision"] or
                     read(self.request_path("requests",arm,n))["observation_key"]!=s["request_key"]):
                    raise ValueError("原始提案或观察已改变")
                if s["adopted"]:adopted.append(n)
                trials.extend(s["trials"]);parent=s["selected"];seen.update(p);current=[]
                for k,val in p.items():
                    r=self.truth[k];e=val-r.label;current.append(e);devices[r.unit_id].append(e)
                errors.extend(current);wins.append(rmse(current))
            if seen!=self.truth.keys():raise ValueError("预测遗漏")
            unit={u:rmse(v) for u,v in devices.items()}
            summary[arm]={"rmse":rmse(errors),"rounds":wins,"by_device":unit,"worst_round":max(wins),
                "device_equal_mean":float(np.mean(list(unit.values()))),"worst_device":max(unit.values()),
                "rows":len(seen),"devices":len(unit),"fits":len(trials),"train_rows":sum(x["rows"] for x in trials),
                "cpu_train_seconds":sum(x["training_cpu_seconds"] for x in trials),"adopted":adopted}
        frozen(RESULT/folder/(self.task+".json"),summary)
        return summary


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("engine","bearing"))
    parser.add_argument("mode",choices=("watch","score","prepare","additional","primary"))
    args=parser.parse_args()
    with ExitStack() as stack:
        # 附加支线独占不同组的快照；共享的内容寻址模型采用原子发布。
        lock_path=RESULT/("additional-locks" if args.mode=="additional" else "issued")/args.task
        stack.enter_context(LocalStore(lock_path).exclusive())
        run=ExternalReplay(args.task,stack)
        if args.mode=="score":
            run.score_external();return
        if args.mode=="primary":
            run.score_external(PRIMARY,"primary-scores");return
        arms=ADDITIONAL if args.mode=="additional" else ARMS
        while True:
            moved=False
            for a in arms:
                moved=run.step(a,publish_only=args.mode=="prepare") or moved
            if args.mode=="prepare":return
            if all(run.state_path(a,10).exists() for a in arms):
                if args.mode!="additional":print({k:v["rmse"] for k,v in run.score_external().items()})
                return
            if not moved:sleep(1.5)


if __name__=="__main__":
    main()
