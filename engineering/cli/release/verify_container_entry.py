"""从 Dockerfile 指定的源码和命令验收公开输入。此检查运行宿主解释器，实际镜像另行验收。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[3]


def verify(data, output, limit, bundle=None, extend_gas=False):
    """在仅含镜像 COPY 清单的空目录运行公开任务，核对真实权重、交接与离线记录。"""
    dockerfile = ROOT / "engineering/configs/container/Dockerfile"
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    entry = json.loads(next(s.removeprefix("ENTRYPOINT ") for s in lines if s.startswith("ENTRYPOINT ")))
    command = json.loads(next(s.removeprefix("CMD ") for s in lines if s.startswith("CMD ")))
    if entry[:2] != ["python", "-B"] or command[:2] != ["public-run", "all"]:
        raise ValueError("镜像默认命令应执行三任务真实维护")
    with TemporaryDirectory(prefix="public-entry-") as temp:
        workspace = Path(temp)
        app, target = workspace / "app", workspace / "output"
        app.mkdir()
        for line in lines:
            if line.startswith("COPY "):
                _, source, dest = line.split()
                if (ROOT / source).is_dir():
                    shutil.copytree(ROOT / source, app / dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                else:
                    shutil.copy2(ROOT / source, app / dest)
        portable = workspace/"public.zip"
        if bundle is not None:
            # 发布实际验收的文件，README 的项目内研究索引留在完整版工作区。
            (app/"README.md").write_text((ROOT/"README.md").read_text(encoding="utf-8").split("## 工作区资料")[0],
                                         encoding="utf-8")
            with ZipFile(portable, "w", ZIP_DEFLATED) as z:
                for path in sorted(app.rglob("*")):
                    if path.is_file():
                        z.write(path, path.relative_to(app))
            app = workspace/"unpacked"
            with ZipFile(portable) as z:
                z.extractall(app)
        argv = [str(data.resolve()) if arg == "/data" else str(target) if arg == "/output" else arg for arg in command]
        if limit is not None:
            argv += ["--rounds", str(limit)]
        env = {k:v for k,v in os.environ.items() if k.upper() in
               {"SYSTEMROOT","WINDIR","PATH","COMSPEC","PATHEXT","TEMP","TMP","LANG","LC_ALL"}}
        def execute(args):
            proc = subprocess.run([sys.executable, "-I", *entry[1:], *args], cwd=app, env=env,
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
            if proc.returncode:
                raise RuntimeError(proc.stderr[-4000:] + proc.stdout[-1200:])
        execute(argv)
        result = json.loads((target / "public-run-all.json").read_bytes())
        if result["evidence_kind"] != "real_model" or set(result["tasks"]) != {"gas", "bearing", "engine"}:
            raise ValueError("默认入口缺少实际运行的任务")
        if extend_gas and limit is not None and limit < 10:
            first = json.loads((target/"gas/run.json").read_bytes())["rounds"]
            execute([command[0],"gas","--data-root",str(data.resolve()),"--output",str(target),"--rounds","10"])
            resumed = json.loads((target/"gas/run.json").read_bytes())["rounds"]
            if resumed[:limit] != first or len(resumed) != 10:
                raise ValueError("气体跨轮恢复未保留已提交的决定或预测")
        evidence = {}
        for name, item in result["tasks"].items():
            folder = target/name
            run = json.loads((target / item["report"]).read_bytes())
            checks = json.loads((folder / run["swanlab"]).read_bytes())
            expected_rounds = 10 if extend_gas and name == "gas" else (limit or 10)
            if (run["completed_rounds"] != expected_rounds or not checks["verified"] or
                    checks["network_attempts"] or not item["predictions"]):
                raise ValueError(f"{name} 没有完整的服务预测或离线记录")
            state = json.loads((folder/"state.json").read_bytes())
            rounds = []
            for r in run["rounds"]:
                ref = r["model_artifact"]
                blob = folder/"models"/ref["key"]/"pipeline.bin"
                manifest = json.loads((blob.parent/"manifest.json").read_bytes())
                issued = json.loads((folder/r["submitted"]).read_bytes())["payload"]["predictions"]
                operation = state["operations"][r["batch"]+":serve:"+r["selected"]]
                job = operation["execution"]["request_id"]
                worker = json.loads((folder/"jobs"/job/"state.json").read_bytes())
                values = worker["value"]
                if "gas_repairs" in values:
                    values = values["predictions"]
                if (hashlib.sha256(blob.read_bytes()).hexdigest() != manifest["sha256"] or
                        worker["status"] != "completed" or issued != values):
                    raise ValueError(f"{name} 受控预测、发布权重和提交内容不一致")
                rounds.append({"batch":r["batch"], "current":r["current"], "selected":r["selected"],
                               "reason":r["reason"], "trial_count":len(r["trials"]), "weights":manifest["sha256"],
                               "prediction_count":len(issued), "first_prediction":r["first_prediction"],
                               "predictions_sha256":hashlib.sha256(json.dumps(issued,sort_keys=True,
                                   separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
                               "worker_request":job, "confirmation":r["confirmation"]})
            evidence[name] = rounds
        bundle_sha = None
        if bundle is not None:
            bundle.parent.mkdir(parents=True, exist_ok=True)
            bundle_sha = hashlib.sha256(portable.read_bytes()).hexdigest()
            # 在交付目录创建普通文件，继承交付目录的读取权限。临时私有目录中的
            # 文件直接替换到这里时，Windows 会保留其私有 ACL，WSL 收到的 ZIP 无法打开。
            with NamedTemporaryFile(dir=bundle.parent, prefix=".pending-release-", delete=False) as handle:
                staged = Path(handle.name)
            try:
                shutil.copyfile(portable, staged)
                os.replace(staged, bundle)
            finally:
                staged.unlink(missing_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"container_built":False,"host":sys.platform,"python":sys.version.split()[0],
              "source":"Dockerfile COPY list; isolated Python; no research results copied",
              "default_command":command,"round_limit":limit, "extend_gas":extend_gas,"tasks":evidence,
              "bundle_sha256":bundle_sha,"dockerfile_sha256":hashlib.sha256(dockerfile.read_bytes()).hexdigest()}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, choices=range(1,11), help="只验收到指定轮；默认覆盖镜像完整命令")
    parser.add_argument("--bundle", type=Path, help="用队友源码包解压验收，成功后发布这份包")
    parser.add_argument("--extend-gas", action="store_true", help="在同目录将气体从指定轮恢复到十轮")
    args = parser.parse_args()
    verify(args.data_root, args.output, args.rounds, args.bundle, args.extend_gas)
