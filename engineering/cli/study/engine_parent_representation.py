"""配对检验原模型输入刻度和完整源训练资料，区分候选能力不足的两种原因。"""
from argparse import ArgumentParser
from hashlib import sha256
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.study.engine_parent_capacity import sources, labeled
from engineering.cli.workflow.london_code_driver import fit_source
from engineering.cli.workflow import public_maintenance as public
from harness.integrations.engine_parent_view import transform
from harness.integrations.published_parent import read
from harness.methods.training_inquiry import LabeledBatch
from harness.runtime.storage import LocalStore, atomic_json


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--prior", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/official")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with LocalStore(args.output).exclusive(), TemporaryDirectory(dir=args.output, prefix="temporary-") as temporary:
        if (args.output / "scores.json").exists(): raise ValueError("不覆盖既有配对实测")
        os.environ["TEMP"] = temporary; os.environ["TMP"] = temporary
        import tempfile
        tempfile.tempdir = temporary
        binding = read(args.prior / "binding.json"); ids = read(args.prior / "rows.json")
        state = read(ROOT / binding["run"] / "regular/state.json")["samples"]
        ordered = ids["explore"] + ids["confirm"] + ids["serve"]
        with np.load(args.prior / "input.npz", allow_pickle=False) as z:
            old_train_x, old_train_y, raw_x = z["train_x"], z["train_y"], z["predict_x"]
        features, parent = transform(binding, raw_x[:, :-1], args.data_root)
        if not np.allclose(parent, raw_x[:, -1], rtol=0, atol=1e-9): raise ValueError("新视图未复现旧父版")
        native_x = np.column_stack((features, raw_x[:, -1]))
        package = public.PACKAGES["engine"][2](args.data_root / "engine" / public.PACKAGES["engine"][0])
        full = sorted(package.train, key=lambda r: r.sample_id)
        source_units = {r.unit_id for r in full}
        if any(state[k]["unit_id"] in source_units for k in ordered): raise ValueError("源设备进入评价")
        raw_train = np.asarray([r.values for r in full]); full_y = np.asarray([r.label for r in full])
        native_train, train_parent = transform(binding, raw_train, args.data_root)
        table = {r.sample_id: i for i,r in enumerate(full)}; subset = np.asarray([table[k] for k in ids["train"]])
        if not np.array_equal(full_y[subset], old_train_y): raise ValueError("源标签身份不一致")
        if not np.allclose(train_parent[subset], old_train_x[:, -1], rtol=0, atol=1e-9): raise ValueError("源父版预测变化")
        views = {"raw": (np.column_stack((raw_train, train_parent)), raw_x),
                 "native": (np.column_stack((native_train, train_parent)), native_x)}
        n, m = len(ids["explore"]), len(ids["confirm"]); service = slice(n+m, len(raw_x))
        exploration = labeled([state[k] for k in ids["explore"]], raw_x[:n])
        confirmation = labeled([state[k] for k in ids["confirm"]], raw_x[n:n+m])
        predictions = {"parent": raw_x[:, -1]}; trials = {}; legal = {}
        for view, (train_x, predict_x) in views.items():
            for scope, fit_rows in (("anchors", subset), ("full", np.arange(len(full)))):
                data = args.output / (view + "-" + scope + ".npz")
                np.savez_compressed(data, train_x=train_x[fit_rows], train_y=full_y[fit_rows],
                                    train_ids=[full[i].sample_id for i in fit_rows], predict_x=predict_x, predict_ids=np.arange(len(raw_x)))
                for learner, source in sources().items():
                    name = f"{view}-{scope}-{learner}"
                    if view == "raw" and scope == "anchors":
                        folder = args.prior / "models" / learner
                        with np.load(folder / "base.npz", allow_pickle=False) as z: pred = z["prediction"]
                        trials[name] = {"reused_from": str(folder), "success":True}
                    else:
                        folder = args.output / "models" / name
                        index, pred = fit_source(source, data, folder)
                        if not np.array_equal(index, np.arange(len(raw_x))): raise ValueError("预测顺序变化")
                        trials[name] = {"source_sha256":sha256(source.encode()).hexdigest(), "success":True,
                                        "cost":read(folder / "cost.json")}
                    predictions[name] = pred
                    for weight in (.25, .5):
                        predictions[f"{name}-mix-{weight}"] = (1-weight)*predictions["parent"]+weight*pred
                    atomic_json(args.output / "trials.json", trials)
        for name, prediction in predictions.items():
            legal[name] = {"explore":exploration.score(prediction[:n]), "confirm":confirmation.score(prediction[n:n+m])}
        best = min(legal,key=lambda k:legal[k]["explore"]["error"])
        old, new = legal["parent"]["confirm"], legal[best]["confirm"]
        adopted = new["error"] < old["error"] and new["worst"] <= 1.01*old["worst"]
        atomic_json(args.output / "decision.json", {"legal_observations":legal,"nomination":best,"adopted":adopted,
                                                   "published":best if adopted else "parent"})
        np.savez_compressed(args.output / "issued.npz", ids=ids["serve"],names=list(predictions),
                            predictions=np.column_stack([p[service] for p in predictions.values()]))
        targets = {r.sample_id:r.label for r in (*package.train,*package.validation)}
        y=np.asarray([targets[k] for k in ids["serve"]]);scoring=LabeledBatch(raw_x[service],y,tuple(ids["serve_groups"]))
        scores={name:scoring.score(pred[service]) for name,pred in predictions.items()}
        result={"scope":"已见官方发动机的2×2输入刻度与训练量配对；无新现场模型调用",
                "scores":scores,"best_fixed":min(scores,key=lambda k:scores[k]["error"]),
                "nomination":best,"adopted":adopted,"published":best if adopted else "parent",
                "fit_rows":{"anchors":len(subset),"full":len(full)},"trials":trials}
        atomic_json(args.output / "scores.json",result)
        winner=result["best_fixed"]
        import json
        print(json.dumps({"best_fixed":winner,"best_fixed_score":scores[winner],"actual":scores[result["published"]],
                          "nomination":best,"adopted":adopted,"fit_rows":result["fit_rows"]},ensure_ascii=False))


if __name__=="__main__":main()
