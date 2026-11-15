"""从公开原包运行同预算对照。按任务提交结果、核对离线记录，中断后复用同一运行目录。"""
import argparse
from dataclasses import asdict
from hashlib import sha256
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys

from engineering.cli.study.audit_gas_policy import audit
from engineering.cli.study.evaluate_rul_baseline import PACKAGES
from engineering.cli.study.run_rul_series import SERIES, run as run_lifetime
from harness.contracts import digest
from harness.evaluation.gas_pilot import run_arm
from harness.integrations.pytrio_planner import PyTrioConfig
from harness.models.official_gas_cpu import PACKAGE_SHA
from harness.runtime.storage import LocalStore, atomic_json

ROOT = Path(__file__).resolve().parents[3]
GAS_PROTOCOL = ROOT / "engineering/configs/experiments/official-gas-public-comparison.json"
TASKS = ("gas", "bearing", "engine")


def definition(task, data_root, mode, service):
    """绑定原包、协议、驱动及实际代码。跨机器搬移同一份源码可恢复，修改后使用新目录。"""
    gas = task == "gas"
    filename, checksum = ("cpu-a-guide-20261110-v1.zip", PACKAGE_SHA) if gas else PACKAGES[task][:2]
    archive = data_root / task / filename
    if sha256(archive.read_bytes()).hexdigest() != checksum:
        raise ValueError(f"{task} 原包与已核对的官方版本不符")
    protocol = GAS_PROTOCOL if gas else SERIES
    sources = list((ROOT / "engineering/src/harness").rglob("*.py"))
    sources.extend([Path(__file__), ROOT / "engineering/cli/study/audit_gas_policy.py",
                    ROOT / "engineering/cli/study/run_rul_series.py", ROOT / "engineering/cli/study/run_rul_maintenance.py",
                    ROOT / "engineering/cli/study/evaluate_rul_baseline.py"])
    code = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sources}
    spec = json.loads(protocol.read_bytes())
    result = {"task": task, "package": filename, "sha256": checksum, "protocol_sha256": sha256(protocol.read_bytes()).hexdigest(),
              "rounds": spec["rounds"], "arms": spec["arms"], "driver": mode,
              "service_config": asdict(service) if service else None, "implementation": digest(code)}
    if not gas:
        once = ROOT / "engineering/configs/experiments/official-rul-device-maintenance.json"
        result["device_split"] = sha256(once.read_bytes()).hexdigest()
    return result


def verify_recording(workflow, output):
    """记录补交独立于训练。子进程封闭网络，读取真正的 SwanLab 离线文件并核对来源。"""
    subprocess.run([sys.executable, "-B", str(ROOT / "engineering/cli/verify/verify_offline_tracking.py"),
                    "--run", str(workflow), "--output", str(output), "--frozen-review"], check=True)
    checked = json.loads((output / "verification.json").read_bytes())
    if not checked["verified"] or checked["cloud_verified"] or not checked["model_experiment"]:
        raise ValueError("本地实测的 SwanLab 离线文件未通过核对")
    return {"mode": checked["mode"], "events": checked["events"],
            "linked_driver_calls": checked["linked_driver_calls"], "files": checked["files"]}


def run_task(spec, data_root, folder, driver):
    with LocalStore(folder / "comparison-lock").exclusive():
        identity = folder / "run-definition.json"
        if identity.is_file() and json.loads(identity.read_bytes()) != spec:
            raise ValueError(f"{spec['task']} 的源码、输入或驱动与已有运行不同")
        atomic_json(identity, spec)
        task = spec["task"]
        if task == "gas":
            archive = data_root / task / spec["package"]
            for arm in spec["arms"]:
                run_arm(archive, GAS_PROTOCOL, folder, arm, driver=driver if arm == "sequence-protected" else None)
            comparison = audit(folder, archive)
            strategies = {r["arm"]: {"score": r["holdout_macro_f1"], "adopted": r["adopted"],
                          "worst_before_update": r["before_batch_worst"],
                          "worker_cpu_seconds": r["total_job_cpu_seconds_including_final_evaluation"]}
                          for r in comparison["arms"]}
            strategies["static"] = {"score": comparison["static"]["holdout_macro_f1"], "adopted": []}
            metric, evaluated = "macro_f1", comparison["held_rows"]
        else:
            path = run_lifetime(task, folder, real=True, driver=driver, data_root=data_root)
            comparison = json.loads(path.read_bytes())
            strategies = {arm: {"score": r["overall"]["rmse"], "worst_round_rmse": r["worst_round_rmse"],
                          "adopted": [s["batch"] for s in r["rounds"] if s["selected"] != s["current"]],
                          "candidate_epochs": r["candidate_epochs"], "candidate_evaluations": r["candidate_evaluations"],
                          "logical_worker_cpu_seconds": r["logical_worker_cpu_seconds"]}
                          for arm, r in comparison["arms"].items()}
            metric, evaluated = "rmse", comparison["arms"]["static"]["overall"]["rows"]
        recordings = {}
        for arm in spec["arms"]:
            workflow = folder / arm if task == "gas" else folder / arm / "workflow"
            recordings[arm] = verify_recording(workflow, folder / "swanlab" / arm)
        receipt = {"task": task, "metric": metric, "evaluated": evaluated, "driver": spec["driver"],
                   "rounds": spec["rounds"], "strategies": strategies, "recordings": recordings,
                   "comparison_sha256": sha256((folder / "comparison.json").read_bytes()).hexdigest(),
                   "definition": digest(spec), "model_experiment": True}
        atomic_json(folder / "task-receipt.json", receipt)
        return receipt


def main(arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=(*TASKS, "all"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/official")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", choices=("rule", "mock", "service"), default="rule")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--enable-service-network", action="store_true")
    parser.add_argument("--enable-real-execution", action="store_true",
                        help="执行本地工业模型的公开对照。省略时仅核对原包、配置和依赖")
    args = parser.parse_args(arguments)
    net = args.driver == "service"
    if net != bool(args.model_config) or args.enable_service_network != (net and args.enable_real_execution):
        parser.error("服务驱动需提供模型配置；只有实际服务运行须另开网络")
    service = PyTrioConfig(**json.loads(args.model_config.read_bytes())) if net else None
    if service and service.base_url.rstrip("/") != PyTrioConfig.base_url:
        parser.error("请核对当前比赛允许的 PyTRiO 服务地址")
    selected = TASKS if args.task == "all" else (args.task,)
    plans = [definition(t, args.data_root, args.driver, service) for t in selected]
    missing = [n for n in ("numpy", "psutil", "swanlab") if find_spec(n) is None]
    if not args.enable_real_execution:
        atomic_json(args.output / f"preparation-{args.task}.json", {"tasks": plans, "missing_runtime_packages": missing,
                    "model_experiment": False, "service_called": False})
        print(f"已核对 {'、'.join(selected)} 的原包与对照协议。缺少的实测依赖：{'、'.join(missing) or '无'}。")
        return
    if missing:
        parser.error("实测环境缺少 " + "、".join(missing))
    driver = {"mode": args.driver}
    if net:
        key = os.environ.get("PYTRIO_API_KEY")
        if not key or not key.strip() or "\r" in key or "\n" in key:
            parser.error("真实服务运行需要在环境中配置有效的凭据")
        driver.update(config=service, allow_network=True, api_key=key)
    receipts = {}
    with LocalStore(args.output / "public-run-lock").exclusive():
        for plan in plans:
            name = plan["task"]
            receipts[name] = run_task(plan, args.data_root, args.output / name, driver)
        path = args.output / f"public-comparison-{args.task}.json"
        atomic_json(path, {"scope": list(selected), "completed": True, "tasks": receipts})
    print(path)
