"""开发对照同样获得损耗监督。每次整机队留出，只凭其余机队的实际误差决定怎样使用新工具。"""
import json
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from score_flight_fault_experts import BASE, DEST as PRIOR, ANCHOR, PAPER, SITES, file_hash, receipt, issue, LocalStore, atomic_json
from harness.models.flight_wear_transfer import WearTransfer
from harness.methods.flight_complement import fixed_mix

DEST = BASE / "flight-wear-transfer"
PRIMARY = "按三个航段损耗分歧选组合"
NEW = ("只凭估计损耗", "传感器和损耗联合训练", "损耗最近五台参照")
TOOLS = (ANCHOR, PAPER, *NEW)


def context(p, spread):
    return np.column_stack((spread, p, p[:, 2:] - p[:, :1], abs(p[:, 1] - p[:, 0])))


def run():
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("该阶段已交付")
        previous = json.loads((PRIOR / "scores.json").read_bytes())
        cache = PRIOR / "features.npz"
        if file_hash(cache) != previous["字段缓存SHA256"]:
            raise ValueError("传感器及源端标志缓存身份不一致")
        with np.load(cache) as z:
            x, h, y = [np.concatenate([z[s + key] for s in SITES]) for key in ("_source", "_health", "_life")]
            groups = np.concatenate([[f"{s}/{k}" for k in z[s + "_ids"][:, 0]] for s in SITES])
            model = WearTransfer().fit(x, h, y, groups)
            joblib.dump(model, DEST / "wear.joblib")
            records = {}
            from score_flight_life_pairs import dataset_path, read_flights
            import h5py
            for s, item in previous["sites"].items():
                old = receipt(PRIOR, s, item)
                ids = [tuple(k) for k in old["ids"]]
                with h5py.File(dataset_path(s)) as hf:
                    answers = read_flights(hf, "test", answers=True)
                if set(ids) != set(answers):
                    raise ValueError("飞行答案不匹配")
                out, spread = model.outputs(z[s + "_target"])
                pred = {**old["predictions"], **out}
                matrix = np.column_stack([pred[k] for k in TOOLS])
                records[s] = (ids, np.array([answers[k] for k in ids]), pred, matrix, context(matrix, spread))
        results = {}
        for s,(ids, truth, pred, matrix, info) in records.items():
            train = {k:v for k,v in records.items() if k != s}
            y, m, c = [np.concatenate([v[i] for v in train.values()]) for i in (1,3,4)]
            sites = np.concatenate([np.repeat(k, len(v[1])) for k,v in train.items()])
            w = np.array([len(y) / (len(train) * len(train[k][1])) for k in sites])
            fixed = fixed_mix(m, y, sites)
            # 从源航班拟合实际误差，再将当趟分量与强固定组合各占一半。
            gate = ExtraTreesRegressor(n_estimators=80, max_depth=4, min_samples_leaf=24, random_state=17, n_jobs=1)
            gate.fit(c, abs(m - y[:, None]), sample_weight=w)
            def weights(feat):
                inv = gate.predict(feat).clip(1) ** -2
                return inv / inv.sum(1)[:, None]
            dyn = np.sum(weights(info) * matrix, axis=1)
            pred["全部损耗工具与论文固定组合"] = matrix @ fixed
            pred["按源平均分量用损耗工具"] = matrix @ np.average(weights(c), weights=w, axis=0)
            pred["统一及损耗联合各半"] = (matrix[:, 0] + matrix[:, 3]) / 2
            pred["当趟损耗纯调权"] = dyn
            pred[PRIMARY] = (dyn + matrix @ fixed) / 2
            results[s] = issue(DEST, s, ids, pred, truth)
        means = {k: float(np.mean([v["scores"][k]["rmse"] for v in results.values()])) for k in pred}
        controls = [previous["decision"]["固定"], "三段和论文固定最优", *NEW, "全部损耗工具与论文固定组合",
                    "按源平均分量用损耗工具", "统一及损耗联合各半"]
        fixed = min(controls, key=means.get)
        paper = min((PAPER, "全部损耗工具与论文固定组合", "三段和论文固定最优", "故障专家与统一及论文固定组合"), key=means.get)
        gain = 1 - means[PRIMARY] / min(means[paper], means[fixed])
        decision = {"主候选":PRIMARY, "最强固定":fixed, "同工具论文": paper, "降幅":gain,"进独立":gain >= .01}
        atomic_json(DEST / "scores.json", {"sites": results, "means": means, "decision": decision, "权重SHA256": file_hash(DEST/"wear.joblib"),
                    "源机数": len(np.unique(groups)), "部件归一化源留机误差": model.error})
        print(json.dumps({"结果":{k:means[k] for k in (PRIMARY, fixed, paper, *NEW)}, "判定":decision},ensure_ascii=False))


if __name__ == "__main__":
    run()
