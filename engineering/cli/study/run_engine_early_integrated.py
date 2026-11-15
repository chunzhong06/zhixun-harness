"""验证开工工况参照进入三任务实际交付与单条预测的数值身份。"""
import json
from pathlib import Path
from argparse import ArgumentParser
import subprocess
import sys

from score_engine_early_reference import DEST, ROOT, official, read, atomic_json
from verify_bearing_hybrid_integration import OLD,predictions
from score_integrated_observer import main as score
from engineering.cli.workflow.engine_input_predict import infer


def main(folder,source_choice=False,full_condition=False,continued=False,inquiry=False):
    if (folder/"comparison.json").exists():
        raise ValueError("本次整套验证已有结果")
    option = "--engine-source-choice" if source_choice else "--engine-early-reference"
    if full_condition:
        option = "--engine-full-condition"
    if continued:
        option = "--engine-continued-condition"
    if inquiry:
        option = "--engine-training-inquiry"
    choice = [option,"adaptive"] if inquiry else [option]
    args = [sys.executable,"-B",str(ROOT/"engineering/cli/workflow/run_workflow.py"),"public-run","all",
            "--gas-feedback","32","--output",str(folder)]
    stages = []
    for run_args in ([*args,*choice,"--rounds","4"], [*args,*choice,"--rounds","10"], args):
        r = subprocess.run(run_args,cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
        if r.returncode:
            raise RuntimeError(r.stderr[-2200:])
        stages.append({t:predictions(folder,t) for t in ("gas","bearing","engine")})
    for task,complete in stages[1].items():
        if complete!=stages[2][task] or any(complete.get(k)!=v for k,v in stages[0][task].items()):
            raise ValueError("接续改变了已交付预测")
        if task!="engine" and complete!=predictions(OLD,task):
            raise ValueError("非发动机任务的交付发生变化")
    score(folder)
    features = {r.sample_id:r.values for r in official("engine")[0].validation}
    report = read(folder/"engine/run.json")
    checked = 0
    previous = predictions(DEST/"integrated","engine") if option=="--engine-early-reference" and folder!=DEST/"integrated" else {}
    changed = {k for k,v in previous.items() if v<0}
    checked_changed = set()
    for j,event in enumerate(report["accepted"],1):
        payload = read(folder/"engine/accepted"/(event["request"]+".json"))["payload"]
        ids = list(payload["predictions"])
        # 每一批至少覆盖首尾两条及中间点，向既有 JSON 单条入口真正提交。
        chosen = sorted({ids[0],ids[len(ids)//2],ids[-1]} | (changed & set(ids)))
        requests = ({"inputs":{"sensors":list(features[k][:21]),"settings":list(features[k][21:])}} for k in chosen)
        answers = list(infer(folder/"engine",j,ROOT/"data/official",requests))
        if any(abs(a["outputs"]["value"]-payload["predictions"][k])>1e-8 for a,k in zip(answers,chosen,strict=True)):
            raise ValueError("单条接口没有复现本轮已发放预测")
        checked+=len(chosen)
        checked_changed.update(changed.intersection(chosen))
        if previous:
            legacy = list(infer(DEST/"integrated/engine",j,ROOT/"data/official",
                               ({"inputs":{"sensors":list(features[k][:21]),"settings":list(features[k][21:])}} for k in chosen)))
            if any(abs(a["outputs"]["value"]-previous[k])>1e-8 for k,a in zip(chosen,legacy,strict=True)):
                raise ValueError("旧回执的单条预测发生变化")
    if previous and (checked_changed!=changed or any(abs(v-max(0.,previous[k]))>1e-8 for k,v in stages[1]["engine"].items())):
        raise ValueError("输出修正范围与原交付不一致")
    result = {"scores":{k:v["score"] for k,v in read(folder/"scores.json")["tasks"].items()},
              "single_rows":checked,"checked_negative_rows":len(checked_changed),"resume_matches":True,"gas_bearing_match":True}
    if source_choice:
        result.update(source_choice=report["source_choice"],source_scores=report["source_scores"])
    if full_condition or continued:
        result.update(training_rows=report["training_rows"],anchor_rows=report["anchor_rows"],native_sha256=report["native_sha256"])
    if inquiry:
        trace=report["training_inquiry"]
        result.update(proposal=trace["proposal"],accepted=trace["accepted"],published=trace["published"],
                      confirmation=trace["confirmation"],fit_count=trace["fit_count"])
    atomic_json(folder/"comparison.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("--output",type=Path,default=DEST/"integrated")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--source-choice",action="store_true")
    mode.add_argument("--full-condition",action="store_true")
    mode.add_argument("--continued",action="store_true")
    mode.add_argument("--inquiry",action="store_true")
    args=parser.parse_args()
    main(args.output.resolve(),args.source_choice,args.full_condition,args.continued,args.inquiry)
