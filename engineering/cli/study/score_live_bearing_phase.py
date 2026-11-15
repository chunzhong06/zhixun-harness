"""检验阶段树从实际维护流恢复并交付的结果，按源时钟独立重算。"""
from hashlib import sha256

from compare_group_objectives import ROOT,atomic_json,digest,read,remaining_minutes,summary
from compare_public_selection import external_batches
from engineering.cli.workflow.engine_reference import run_prepared
from engineering.cli.workflow.public_maintenance import assemble_public
from score_bearing_life_scale import DEST
from score_local_defaults import cost
from harness.data.official_bearing import CHANNELS, MEASURES


def single(name):
    import json
    import subprocess
    import sys
    folder=DEST/("integrated/bearing" if name=="bearing" else "live/pronostia")
    report=read(folder/"run.json")
    state=read(folder/"regular/state.json")
    expected={}
    for item in report["accepted"]:
        payload=read(folder/"accepted"/(item["request"]+".json"))["payload"]
        if digest([report["identity"],payload])!=item["request"] or expected.keys() & payload["predictions"].keys():
            raise ValueError("预测回执重复或变更")
        expected.update(payload["predictions"])
    requests=[]
    for key in expected:
        r=state["samples"][key]
        if r["role"]!="serve" or r["label"] is not None:
            raise ValueError("请求不属于未回标服务测量")
        x=r["values"]
        f={c:dict(zip(MEASURES,x[i*len(MEASURES):(i+1)*len(MEASURES)],strict=True)) for i,c in enumerate(CHANNELS)}
        requests.append(json.dumps({"task":"regression","inputs":{"features":f}}))
    proc=subprocess.run([sys.executable,"-B",str(ROOT/"engineering/cli/workflow/run_workflow.py"),"bearing-phase-predict",
                         "--run",str(folder)],input="\n".join(requests)+"\n",encoding="utf-8",capture_output=True)
    if proc.returncode:
        raise ValueError(proc.stderr)
    diff=max(abs(json.loads(line)["outputs"]["value"]-v) for line,v in zip(proc.stdout.splitlines(),expected.values(),strict=True))
    if diff>1e-8 or report["completed_rounds"]!=10:
        raise ValueError("真实单条推理和十轮交付不一致")
    atomic_json(folder/"single-rows.json",{"rows":len(expected),"max_abs_difference":diff,"model":report["tree_artifact"],
                "run_sha256":sha256((folder/"run.json").read_bytes()).hexdigest()})
    print(name,len(expected),diff)


def main():
    output=DEST/"live/pronostia"
    root=ROOT/"data/official"
    for steps in (3,10):
        model,batches=assemble_public("bearing",root,output/"regular","selected",{"mode":"rule"})
        batches=external_batches("bearing",batches,match_window=True)
        model.tools.values={r.sample_id:r.values for b in batches for r in b.samples if r.role!="fit"}
        if steps==3 and (output/"run.json").exists() and read(output/"run.json")["completed_rounds"]>3:
            continue
        report=run_prepared(root,output,model,batches,steps,policy="phase-fixed",task="bearing")
    old=DEST/"bearing/pronostia/issued.json"
    if sha256(old.read_bytes()).hexdigest()!=read(old.with_name("scores.json"))["issued_sha256"]:
        raise ValueError("冻结参照来源不同")
    controls=read(old)["predictions"]
    data,groups={},{}
    for item,batch in zip(report["accepted"],batches,strict=True):
        payload=read(output/"accepted"/(item["request"]+".json"))["payload"]
        pred=payload["predictions"]
        current={r.sample_id:r.unit_id for r in batch.samples if r.available_at==batch.ordinal and r.role=="serve"}
        if (digest([report["identity"],payload])!=item["request"] or digest(pred)!=item["outputs_sha256"]
                or pred.keys()!=current.keys() or data.keys()&pred.keys()):
            raise ValueError("实际服务回执、分片或轮次错误")
        data.update(pred)
        groups.update(current)
    delta=max(abs(v-controls["phase-fixed"][k]) for k,v in data.items())
    if delta>1e-8 or data.keys()!=controls["phase-fixed"].keys():
        raise ValueError("公开训练重建的阶段树与独立发放不同")
    fit={r.sample_id for r in batches[0].samples if r.role=="fit"}
    p={k:controls[k] for k in ("current","paper","direct","median")}
    table=summary("bearing",remaining_minutes("full-test-manifest.json"),{"parent":p["current"],**p,"live":data},groups)
    atomic_json(output/"scores.json",{"arms":table,"max_abs_diff":delta,"run_sha256":sha256((output/"run.json").read_bytes()).hexdigest(),
                "ordinary":cost(output/"regular",model.tools.source["package_sha256"],fit),
                "reference_seconds":sum(x["reference_seconds"] for x in report["accepted"])})
    print(len(data),table["live"]["score"],delta)


if __name__=="__main__":
    from argparse import ArgumentParser
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--single",choices=("bearing","pronostia"))
    args=parser.parse_args()
    if args.single:
        single(args.single)
    else:
        main()
