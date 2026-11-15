"""在完整已交付发动机父版之上实训少量对手，先检查合法候选的修错空间。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "engineering/src")]
from engineering.cli.workflow.london_code_driver import fit_source
from engineering.cli.workflow import public_maintenance as public
from harness.integrations.published_parent import augment, bind, read
from harness.methods.training_inquiry import LabeledBatch
from harness.runtime.storage import LocalStore, atomic_json


def sources():
    """事前固定四类校正；原输入和父版预测均可见，拟合仅用源训练答案。"""
    specifications = {
        "parent-affine": ("make_pipeline(StandardScaler(), Ridge(alpha=10.))", "X[:, -1:]", True),
        "residual-ridge": ("make_pipeline(StandardScaler(), Ridge(alpha=100.))", "X", True),
        "residual-tree": ("HistGradientBoostingRegressor(max_iter=64, max_leaf_nodes=7, min_samples_leaf=32, l2_regularization=10., random_state=17)", "X", True),
        "direct-tree": ("HistGradientBoostingRegressor(max_iter=96, max_leaf_nodes=15, min_samples_leaf=32, l2_regularization=10., random_state=17)", "X", False),
    }
    result = {}
    for name, (model, view, residual) in specifications.items():
        target = "y - X[:, -1]" if residual else "y"
        prediction = f"X[:, -1] + self.model.predict({view})" if residual else f"self.model.predict({view})"
        result[name] = f'''from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

class ParentCandidate(BaseEstimator):
    def fit(self, X, y):
        self.model = {model}
        self.model.fit({view}, {target})
        return self

    def predict(self, X):
        return {prediction}

def build_model():
    return ParentCandidate()
'''
    return result


def labeled(rows, values):
    return LabeledBatch(values, np.asarray([r["label"] for r in rows]), tuple(r["unit_id"] for r in rows))


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=ROOT / "data/official")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    with LocalStore(args.output).exclusive(), TemporaryDirectory(dir=args.output, prefix="temporary-") as temporary:
        if (args.output / "scores.json").exists(): raise ValueError("不覆盖既有候选能力实验")
        os.environ["TEMP"] = temporary; os.environ["TMP"] = temporary
        import tempfile
        tempfile.tempdir = temporary
        report = read(args.run / "run.json"); samples = read(args.run / "regular/state.json")["samples"]
        rows = sorted(samples.values(), key=lambda r: r["sample_id"])
        fit = [r for r in rows if r["role"] == "fit"]
        explore = [r for r in rows if r["role"] == "explore" and r["available_at"] == 0]
        confirm = [r for r in rows if r["role"] == "adopt" and r["available_at"] == 0]
        serve = [r for r in rows if r["role"] == "serve"]
        groups = [{r["unit_id"] for r in batch} for batch in (fit, explore, confirm, serve)]
        if any(groups[i] & groups[j] for i in range(4) for j in range(i)):
            raise ValueError("训练、探索、确认和服务设备有交集")
        if any(r["label"] is not None for r in serve): raise ValueError("服务输入泄露答案")
        binding = bind(args.run, "engine", 1)
        x_train = augment(binding, [r["values"] for r in fit], args.data_root)
        all_rows = explore + confirm + serve
        raw = np.asarray([r["values"] for r in all_rows])
        x = augment(binding, raw, args.data_root)
        n, m = len(explore), len(confirm); service = slice(n + m, len(x))
        original = {}
        for event in report["accepted"]:
            original.update(read(args.run / "accepted" / (event["request"] + ".json"))["payload"]["predictions"])
        if not np.allclose(x[service, -1], [original[r["sample_id"]] for r in serve], rtol=0, atol=1e-10):
            raise ValueError("父版没有复现全部原实际交付")
        atomic_json(args.output / "binding.json", binding)
        data = args.output / "input.npz"
        np.savez_compressed(data, train_x=x_train, train_y=[r["label"] for r in fit],
                            train_ids=[r["sample_id"] for r in fit], predict_x=x, predict_ids=np.arange(len(x)))
        atomic_json(args.output / "rows.json", {"train": [r["sample_id"] for r in fit],
                    "explore": [r["sample_id"] for r in explore], "confirm": [r["sample_id"] for r in confirm],
                    "serve": [r["sample_id"] for r in serve], "serve_groups": [r["unit_id"] for r in serve]})
        exploration, confirmation = labeled(explore, x[:n]), labeled(confirm, x[n:n+m])
        predictions = {"parent": x[:, -1]}; trials = {}
        for name, source in sources().items():
            folder = args.output / "models" / name
            try:
                ids, prediction = fit_source(source, data, folder)
                if not np.array_equal(ids, np.arange(len(x))): raise ValueError("候选预测顺序改变")
                predictions[name] = prediction
                trials[name] = {"success": True, "source_sha256": sha256(source.encode()).hexdigest(),
                                "cost": read(folder / "cost.json")}
            except (ValueError, TimeoutError) as exc:
                trials[name] = {"success": False, "error": str(exc)}
            atomic_json(args.output / "trials.json", trials)
        # 所有混合比例事前固定；混合不会另拟合，同源权重身份保留。
        for name in list(predictions):
            if name == "parent": continue
            for weight in (.25, .5):
                predictions[f"{name}-mix-{weight}"] = (1-weight) * predictions["parent"] + weight * predictions[name]
        visible = {}
        for name, prediction in predictions.items():
            visible[name] = {"explore": exploration.score(prediction[:n]),
                             "confirm": confirmation.score(prediction[n:n+m])}
        best = min(visible, key=lambda name: visible[name]["explore"]["error"])
        parent_confirm, candidate_confirm = visible["parent"]["confirm"], visible[best]["confirm"]
        adopted = (candidate_confirm["error"] < parent_confirm["error"] and candidate_confirm["worst"] <= 1.01*parent_confirm["worst"])
        decision = {"nomination": best, "adopted": adopted, "published": best if adopted else "parent",
                    "legal_observations": visible}
        atomic_json(args.output / "decision.json", decision)
        np.savez_compressed(args.output / "issued.npz", ids=[r["sample_id"] for r in serve],
                            names=list(predictions), predictions=np.column_stack([p[service] for p in predictions.values()]))
        # 所有候选和合法发布决定已封存；终局答案只进入开发诊断。
        package = public.PACKAGES["engine"][2](args.data_root / "engine" / public.PACKAGES["engine"][0])
        targets = {r.sample_id: r.label for r in (*package.train, *package.validation)}
        y = np.asarray([targets[r["sample_id"]] for r in serve])
        scoring = LabeledBatch(raw[service], y, tuple(r["unit_id"] for r in serve))
        scores = {name: scoring.score(p[service]) for name, p in predictions.items()}
        oracle = np.min(np.column_stack([(p[service]-y)**2 for p in predictions.values()]), axis=1)
        result = {"scope": "已见官方发动机开发资料，固定候选能力筛选，无新现场模型调用",
                  "scores": scores, "decision": {k:v for k,v in decision.items() if k!='legal_observations'},
                  "best_fixed": min(scores, key=lambda name: scores[name]["error"]),
                  "row_oracle_diagnostic_rmse": float(np.sqrt(oracle.mean())),
                  "rows": {"fit":len(fit),"explore":n,"confirm":m,"serve":len(serve)},
                  "units": dict(zip(("fit","explore","confirm","serve"), map(len,groups))), "trials": trials}
        atomic_json(args.output / "scores.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
