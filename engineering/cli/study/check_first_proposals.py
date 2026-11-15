"""回放两组已发出的改法，统一父版、训练器与服务设备，定位最早的决定差别。"""
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile

from compare_native_proposer import ROOT, OUTPUT
from run_unseen_agents import ExternalReplay
from run_luna_rul_comparison import rmse
from harness.contracts import digest
from harness.runtime.storage import atomic_json
from verify_unseen_scores import remaining_minutes, remaining_cycles


def load(path):
    return json.loads(path.read_bytes())


def compare(task, numbers):
    source = ROOT/"results/experiments/unseen-agents"
    truth = (remaining_cycles if task=="engine" else remaining_minutes)()
    with ExitStack() as stack:
        replay = ExternalReplay(task,stack)
        original_models = replay.study/"models"
        replay.study = OUTPUT/"paired-proposals"/task
        (replay.study/"models").mkdir(parents=True,exist_ok=True)
        result = []
        for n in numbers:
            # 父版本固定取比较前项目当时的在用版。动作取当时原始提交次序，
            # 不用事后的服务成绩提名。记录首项和两项预算的两种口径。
            parent = load(source/"issued"/task/"project-frozen"/f"{n:02d}.json")["parent"]
            if not replay.model_path(parent).exists():
                copyfile(original_models/(parent+".npz"),replay.model_path(parent))
            if sha256(replay.model_path(parent).read_bytes()).hexdigest()!=parent:raise ValueError("父版权重不一致")
            roles = replay.roles(n)
            explore, confirm, serve = (roles[k] for k in ("explore","adopt","serve"))
            def service_score(key):
                return rmse([pred-truth[row.sample_id] for row,pred in zip(serve,replay.predict(key,serve),strict=True)])
            base_exp=rmse(replay.errors(parent,explore));base_cfm=rmse(replay.errors(parent,confirm))
            record={"round":n,"parent":parent,"parent_service":service_score(parent),"service_rows":len(serve),"arms":{}}
            for arm in ("project-frozen","simple"):
                request=load(source/"drivers"/arm/"requests"/task/f"{n:02d}.json")
                reply=load(source/"drivers"/arm/"responses"/task/f"{n:02d}.json")
                candidates=[]
                for j,act in enumerate(reply["actions"],1):
                    rows=replay.training_rows(act,request)
                    trial=replay.train(arm,n,j,act,rows,parent,explore)
                    key=trial["candidate"]
                    candidates.append({"action":act,"rows":len(rows),"key":key,"explore":rmse(replay.errors(key,explore)),
                                       "confirm":rmse(replay.errors(key,confirm))})
                def issue(cap):
                    best=min(candidates[:cap],key=lambda c:c["explore"],default=None)
                    key=best["key"] if best and best["explore"]<base_exp-1e-12 and best["confirm"]<base_cfm-1e-12 else parent
                    return {"served":key,"rmse":service_score(key)}
                record["arms"][arm]={"candidates":candidates,"first_only":issue(1),"up_to_two":issue(2)}
            # 两组首项实际选的资料不同才检验“哪几行”与“行数”的差别。沿用此前冻结的三个种子。
            original=record["arms"]["project-frozen"]["candidates"][0]
            primary=record["arms"]["simple"]["candidates"][0]
            if primary["action"]!=original["action"]:
                controls={}
                for seed in (17,29,41):
                    sample=sorted(replay.fit,key=lambda r:digest([seed,n,1,r.sample_id]))[:primary["rows"]]
                    job=replay.train(f"same-count-{seed}",n,1,{"seed":seed},sample,parent,explore)
                    key=job["candidate"]
                    exp=rmse(replay.errors(key,explore))
                    cfm=rmse(replay.errors(key,confirm)) if exp<base_exp-1e-12 else None
                    used=key if cfm is not None and cfm<base_cfm-1e-12 else parent
                    controls[str(seed)]={"rows":len(sample),"explore":exp,"confirm":cfm,"adopted":used!=parent,
                                         "served_rmse":service_score(used),"candidate_rmse":service_score(key)}
                record["first_proposal_same_count"]=controls
            result.append(record)
        atomic_json(replay.study/"scores.json",result)
        for r in result:
            print(task,r["round"],r["parent_service"],[(k,v["first_only"]["rmse"],v["up_to_two"]["rmse"]) for k,v in r["arms"].items()])


if __name__=="__main__":
    compare("engine",(1,2))
    compare("bearing",(1,))
