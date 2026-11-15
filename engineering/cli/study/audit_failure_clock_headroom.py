"""从已封存寿命预测算出反馈能影响哪些服务行及现有工具择时的乐观上限。"""
from hashlib import sha256
from math import sqrt

import numpy as np

from score_completed_residual_transfer import DEST, LOCAL, atomic_json, engines, femto, official, read, xjtu


OUT = DEST.parent / "failure-clock-headroom"


def audit(site, stream):
    failures, reference, measured, answers, _ = stream
    path = DEST / site / "issued.json"
    score = read(path.with_name("scores.json"))
    identity = sha256(path.read_bytes()).hexdigest()
    if identity != score["issued_sha256"]:
        raise ValueError("用于分析的交付记录有变化")
    issued = read(path)
    keys = [row[0] for row in measured]
    owners = issued["owners"]
    if set(keys) != set(owners) or len(keys) != len(owners) or set(owners.values()) & failures.keys():
        raise ValueError("反馈和服务记录重叠或不同")
    now = np.asarray([issued["clock"][k][0] for k in keys])
    if not np.array_equal(now, [row[2] for row in measured]):
        raise ValueError("回执时钟与原测量不一致")
    released = set()
    bank = {u: {r[1] for r in reference if r[0] == u} for u in failures}
    events = issued["events"]
    for event in events:
        ready = {u for u, end in failures.items() if end < event["time"]} - released
        if ready != {r["unit"] for r in event["released"]}:
            raise ValueError("回标开放时刻与实际回执不符")
        for row in event["released"]:
            if set(row["ids"]) != bank[row["unit"]] or len(row["ids"]) != 5:
                raise ValueError("回标行不符")
        released.update(ready)

    truth = answers()
    y = np.asarray([truth[k] for k in keys])
    predictions = {a: np.asarray([v[k] for k in keys]) for a, v in issued["predictions"].items()}
    loss = {a: (vals-y)**2 for a, vals in predictions.items()}
    for a, values in loss.items():
        if not np.isclose(sqrt(values.mean()), score["arms"][a]["score"], rtol=0, atol=1e-8):
            raise ValueError("历史交付未按原标签复现")
    # 每个参考失效只开放一次更新机会。乐观分析使用事后服务答案，禁止进入维护器。
    fixed = [arm for arm in loss if arm not in ("confirmed", "always-fit")]
    best = min(fixed, key=lambda a: loss[a].mean())
    event_times = np.asarray([e["time"] for e in events])
    epoch = np.searchsorted(event_times, now, side="right")
    before = epoch == 0
    oracle = np.empty_like(y, dtype=float)
    segments = []
    for i in np.unique(epoch):
        mask = epoch == i
        arm = "base" if i == 0 else min(fixed, key=lambda a: loss[a][mask].mean())
        oracle[mask] = loss[arm][mask]
        segments.append({"after_events": int(i), "rows": int(mask.sum()), "oracle_arm": arm})
    def root_mean(values):
        return float(np.sqrt(np.mean(values)))

    first_confirmation = next((e["time"] for e in events if e["confirm_mse"] is not None), np.inf)
    result = {"issued_sha256": identity, "units": len(set(owners.values())), "rows": len(keys),
              "feedback_units": len(released), "returned_labels": len(released)*5,
              "before_first_feedback": int(before.sum()), "before_first_confirm": int(np.sum(now < first_confirmation)),
              "first_feedback_clock": events[0]["time"] if events else None,
              "fixed": {a: root_mean(loss[a]) for a in fixed}, "best_fixed": best,
              "confirmed": root_mean(loss["confirmed"]), "always_fit": root_mean(loss["always-fit"]),
              "fixed_base": root_mean(loss["base"]), "epoch_oracle_after_base_start": root_mean(oracle),
              "perfect_after_first_feedback_floor": float(np.sqrt(loss["base"][before].sum()/len(y))),
              "intervals": segments}
    atomic_json(OUT / site / "scores.json", result)
    print(site, result["before_first_feedback"], result["before_first_confirm"], len(y), best,
          round(result["fixed"][best], 3), round(root_mean(oracle), 3), flush=True)
    return result


if __name__ == "__main__":
    package, _, _ = official("bearing")
    streams = {"femto": femto(48), "xjtu": xjtu(package, LOCAL/"fifth-offset6.json"), "phm08": engines()}
    atomic_json(OUT/"scores.json", {site: audit(site, stream) for site, stream in streams.items()})
