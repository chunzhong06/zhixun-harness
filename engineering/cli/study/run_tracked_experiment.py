"""新科研作业先登记SwanLab，再沿原预算运行器执行；不扫描或上传旧实验。"""
from argparse import ArgumentParser,REMAINDER
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"engineering/src"))
from harness.integrations.research_tracking import ResearchSwanLab


def scalars(value,prefix=""):
    result={}
    if isinstance(value,dict):
        for k,v in value.items():result.update(scalars(v,f"{prefix}/{k}" if prefix else k))
    elif isinstance(value,(int,float,bool)):result[prefix]=value
    return result


def display_protocol(path, payload, checksum, description=None):
    """优先展示与原文件身份一致的译文；旧协议可随记录附中文说明和原件位置。"""
    translation = ROOT / "engineering/configs/tracking/protocols" / (path.stem + ".zh.json")
    if translation.is_file():
        translated = json.loads(translation.read_bytes())
        if translated.get("原协议SHA256") == checksum:
            return translated["实验协议"]
    if "中文协议" in payload:
        return payload["中文协议"]
    if any(any("\u4e00" <= char <= "\u9fff" for char in key) for key in payload if key not in {"中文名称", "中文说明"}):
        return payload.get("实验协议", payload)
    return {"说明": description or payload.get("中文说明") or "详细步骤及评分条件见本次冻结的协议文件。",
            "原件位置": path.relative_to(ROOT).as_posix()}


def main():
    ap=ArgumentParser();ap.add_argument("--protocol",required=True);ap.add_argument("--input-checkpoint",required=True,type=int)
    ap.add_argument("--kind",choices=("exploratory","confirmatory","pilot","smoke"),required=True)
    ap.add_argument("--receipt-root",default="results/experiments/local-defaults/whole-maintenance/jobs")
    ap.add_argument("--name",help="SwanLab中文实验名称")
    ap.add_argument("--description",help="SwanLab中文实验说明")
    ap.add_argument("command",nargs=REMAINDER);args=ap.parse_args()
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    if not command:raise ValueError("缺少实验命令")
    settings=json.loads((ROOT/"engineering/configs/tracking/swanlab.json").read_bytes())
    state=json.loads((ROOT/"research-state.json").read_bytes());number=len(state["runs"])+1;run_id=f"run_{number:04d}"
    if state["active_run"] or number<settings["first_new_run_number"] or number>state["budget"]["max_runs"]:
        raise ValueError("只允许预算内的下一项新作业，不可用本入口回传旧记录")
    protocol=ROOT/args.protocol;payload=json.loads(protocol.read_bytes())
    checksum=sha256(protocol.read_bytes()).hexdigest()
    checkpoint=next((c for c in state["checkpoints"] if c["id"]==args.input_checkpoint),None)
    if checkpoint is None or args.protocol not in checkpoint["artifacts"]:
        raise ValueError("先建立包含协议的有效输入检查点，再创建云端作业")
    # 避免准备阶段就失败却先产生一次云端实验；正式运行器仍会再次核对，防止中途变化。
    for name,expected in checkpoint["artifacts"].items():
        path=ROOT/name
        if not path.is_file() or sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError("冻结输入缺失或已经变化："+name)
    folder=ROOT/args.receipt_root/run_id
    tracker=ResearchSwanLab(settings,ROOT/"results/engineering/research-workflow/swanlab"/run_id,
        name=args.name or payload.get("中文名称",f"研发实验｜{run_id}"),kind=args.kind,
        description=args.description or payload.get("中文说明","按冻结协议开展真实实验，记录实际交付评分、对照与资源消耗；证据范围以协议和本地回执为准。"),
        config={"研究运行号":run_id,"历史摘要":False,"实验协议":display_protocol(protocol,payload,checksum,args.description),
        "原协议文件":args.protocol,"协议SHA256":checksum,
        "输入检查点":args.input_checkpoint,"执行命令":command,"累计预算":state["budget"]})
    print(json.dumps({"swanlab_url":tracker.url,"research_run":run_id}),flush=True)
    tracker.log({"lifecycle/started":1},0)
    flow=Path.home()/".agents/skills/autoresearch/scripts/research_flow.py"
    result=subprocess.run([sys.executable,"-B",str(flow),"run","--project",str(ROOT),"--protocol",args.protocol,
        "--input-checkpoint",str(args.input_checkpoint),"--receipt-root",args.receipt_root,"--kind",args.kind,
        "--",*command],cwd=ROOT,capture_output=True,text=True,encoding="utf-8")
    record=json.loads((folder/"run.json").read_bytes()) if (folder/"run.json").exists() else {}
    step=1
    log=folder/"stdout.log"
    if log.exists():
        for line in log.read_text(encoding="utf-8",errors="replace").splitlines():
            try:value=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(value,dict) and 'native_trace' in value:
                if not tracker.record_evidence(value['native_trace'],step=step):
                    continue
            metrics=scalars({k:v for k,v in value.items() if k!='native_trace'}) if isinstance(value,dict) else scalars(value)
            if metrics:tracker.log(metrics,step);step+=1
    tracker.log({"runtime/seconds":record.get("elapsed_seconds",0),"lifecycle/succeeded":int(result.returncode==0)},step)
    report=tracker.finish(result.returncode==0)
    # 全量指标保留在cloud.json和原始实验日志；终端只回报运行与云端核验状态。
    status_report={key:value for key,value in report.items() if key!="latest"}
    print(json.dumps({"run":run_id,"status":record.get("status","not_started"),"swanlab":status_report},ensure_ascii=False),flush=True)
    if result.returncode:print(result.stderr[-1500:],file=sys.stderr)
    return result.returncode or (0 if report["cloud_verified"] else 3)


if __name__=="__main__":raise SystemExit(main())
