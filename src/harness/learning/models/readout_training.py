"""从官方寿命检查点继续拟合输出层，保留原输入尺度、隐藏层和原包预测接口。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from harness.core.contracts import digest
from harness.adapters.datasets.official_rul import ARCHIVE_SHA
from harness.learning.models.official_rul_cpu import RulPilotTools, _subset, _trainer, read_parent
from harness.adapters.storage.local import atomic_json


def fit_readout(model, x, y, kind):
    """一步岭回归从父版权重出发，以训练行统计残差；保存原 Adam 动量以供其他配方接续。"""
    if kind not in ("ridge","bias") or model.meta["task"] != "regression" or len(y) != len(x) or not len(x):
        raise ValueError("输出层更新要求带目标的寿命训练行")
    hidden, logits = model.forward(model.normalized(x))
    ones = np.ones((len(x), 1))
    design = ones if kind == "bias" else np.column_stack((hidden, ones))
    residual = (y-model.meta["target_mean"])/model.meta["target_scale"] - logits[:,0]
    gram = design.T@design + (len(x)*0.1)*np.eye(design.shape[1])
    delta = np.linalg.solve(gram, design.T@residual)
    if kind == "ridge":
        model.params["w2"][:,0] += delta[:-1]
    model.params["b2"][0] += delta[-1]
    model.meta["steps"] += 1
    model.meta["epochs"] += 1
    model.validate()
    return float(np.mean((residual-design@delta)**2))


class BiasRulTools(RulPilotTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recipe_sha = sha256(Path(__file__).read_bytes()).hexdigest()
        self.identity = digest([self.identity, "bias", self.recipe_sha])

    def _fit_request(self, task, action, rows, parent):
        if action.fit_manifest not in (None,"full"):
            raise ValueError("偏差校正使用全部冻结训练行")
        return {**super()._fit_request(task, action, rows, parent),
                "readout_recipe":"bias", "readout_sha256":self.recipe_sha}


def execute_readout_job(folder, payload, artifacts):
    """只在工作进程中换用输出层拟合；模型和训练日志均由经过校验的随包代码保存。"""
    request, backend = payload["request"], payload["backend"]
    archive = Path(backend["archive"])
    profile = {"bearing":"B","engine":"C"}[backend["task"]]
    if (request["readout_recipe"] != "bias" or request["readout_sha256"] != sha256(Path(__file__).read_bytes()).hexdigest()
            or request["worker_sha256"] != sha256(Path(__file__).with_name("official_rul_cpu.py").read_bytes()).hexdigest()
            or request["kind"] != "continue" or request["epochs"] != 1 or payload["rows"]
            or request["package_sha256"] != ARCHIVE_SHA[backend["task"]]
            or request["package_sha256"] != sha256(archive.read_bytes()).hexdigest()):
        raise ValueError("偏差校正的原包或配方身份不符")
    with TemporaryDirectory(dir=folder) as tmp, ZipFile(archive) as z, _trainer(z,Path(tmp),profile) as trainer:
        work = Path(tmp)
        source = {"package_sha256":request["package_sha256"], "checkpoint_sha256":sha256(z.read("checkpoint.npz")).hexdigest()}
        header, blob, _ = read_parent(artifacts,request["parent"],source)
        if header["sha256"] != request["parent_sha256"] or source["checkpoint_sha256"] != request["checkpoint_sha256"]:
            raise ValueError("校正父版或初始检查点不符")
        if "fit_runtime" in request:
            import sys
            if backend["task"] != "bearing" or digest(request["fit_runtime"]) != digest([sys.version_info[:3],np.__version__]):
                raise ValueError("校正缓存的数值环境不符")
        parent, train, explore = work/"parent.npz",work/"train.jsonl",work/"explore.jsonl"
        parent.write_bytes(blob)
        diagnostic = backend.get("training_diagnostics",request)
        _subset(z,"train",request["fit_ids"],train)
        _subset(z,"validation",diagnostic["exploration_ids"],explore)
        base = trainer.Model
        class ReadoutModel(base):
            def fit(self,x,y,epochs):
                if epochs != 1:
                    raise ValueError("每次只拟合一遍")
                return [fit_readout(self,x,y,"bias")]
        trainer.Model = ReadoutModel
        try:
            report = trainer.train_run(str(work/"baselines"/"configs"/f"{profile}.json"),train,explore,
                                       work/"candidate", resume=parent, epochs=1, stage_id=diagnostic["stage"])
        finally:
            trainer.Model = base
        output = work/"candidate"/"checkpoint.npz"
        initial, child = base.load(parent), base.load(output)
        content = output.read_bytes()
        if (child.meta["steps"] != initial.meta["steps"]+1 or child.meta["epochs"] != initial.meta["epochs"]+1
                or report["parent_checkpoint_sha256"] != "sha256:"+header["sha256"]
                or report["checkpoint_sha256"] != "sha256:"+sha256(content).hexdigest()
                or any(not np.array_equal(initial.params[k],child.params[k]) for k in ("w1","b1","w2"))):
            raise ValueError("校正权重或实际训练步数不符")
        report.update(readout_recipe="bias",readout_sha256=request["readout_sha256"])
        atomic_json(folder/"training.json",report)
        (folder/"history.jsonl").write_bytes((work/"candidate"/"history.jsonl").read_bytes())
        return {"artifact":artifacts.create(request,lambda:content)}
