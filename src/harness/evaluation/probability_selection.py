"""冻结的两阶段分类选择规则；保留严格提升分支，概率只用于不退步的平分。

调用方负责概率来自真实模型、源类别及阶段权限。该模块不训练、不预测，
也不改变任务原有的 Macro-F1 类别口径。缺类或无真实概率时不启用破平分支。
"""
import numpy as np

TOLERANCE = 1e-12
LOGLOSS_CLIP = 1e-15


def balanced_logloss(truth, probabilities, probability_classes, source_classes):
    """各源类别先平均其真实标签 NLL，再等权平均；阶段缺类直接报错。"""
    y = np.asarray(truth)
    p = np.asarray(probabilities, dtype=float)
    columns, classes = list(probability_classes), list(source_classes)
    if not classes or len(set(classes)) != len(classes) or len(set(columns)) != len(columns) or set(columns) != set(classes):
        raise ValueError("概率列必须与源类别一一对应")
    if y.ndim != 1 or not len(y) or set(y.tolist()) != set(classes):
        raise ValueError("反馈阶段必须覆盖全部源类别且不含额外类别")
    if p.shape != (len(y), len(columns)) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError("真实概率形状或数值非法")
    if not np.allclose(p.sum(axis=1), 1., rtol=1e-6, atol=1e-7):
        raise ValueError("真实概率逐行之和必须为 1")
    # 只在计算 NLL 时剪裁，保持原始概率和原硬分类不变。
    return float(np.mean([-np.log(np.clip(p[y == label, columns.index(label)], LOGLOSS_CLIP, 1 - LOGLOSS_CLIP)).mean()
                          for label in classes]))


def select_two_stage(phases, source_classes):
    """返回旧/新决定及理由；phases 固定为 explore、confirm 两个阶段。

每阶段包含 parent_f1、candidate_f1、truth、两方 probabilities/classes，
以及两方 native_probability 布尔证据。严格提升原分支无需概率；只有
原规则拒绝时，才检查不退步、至少一期打平、真实概率与全部源类别覆盖。
    """
    if set(phases) != {"explore", "confirm"}:
        raise ValueError("选择必须恰有探索和确认两阶段")
    checks = {}
    for name in ("explore", "confirm"):
        stage = phases[name]
        parent, candidate = float(stage["parent_f1"]), float(stage["candidate_f1"])
        if not np.isfinite([parent, candidate]).all() or not 0 <= parent <= 1 or not 0 <= candidate <= 1:
            raise ValueError("Macro-F1 必须为有限的 0 至 1 数值")
        difference = candidate - parent
        checks[name] = {"parent_Macro-F1": parent, "candidate_Macro-F1": candidate, "Macro-F1_difference": difference,
                        "strictly_improved": difference > TOLERANCE,
                        "nondecreasing": difference >= -TOLERANCE, "tied": abs(difference) <= TOLERANCE}
    old = all(value["strictly_improved"] for value in checks.values())
    result = {"old_adopt": old, "new_adopt": old, "branch": "strict_improvement" if old else "keep_parent", "phases": checks}
    if old:
        result["reason"] = "两阶段主指标均严格提升，沿用旧采用"
        return result
    if not all(value["nondecreasing"] for value in checks.values()) or not any(value["tied"] for value in checks.values()):
        result["reason"] = "主指标未满足两阶段均不退且至少一期打平"
        return result
    for name in ("explore", "confirm"):
        stage = phases[name]
        if not stage.get("parent_native_probability", False) or not stage.get("candidate_native_probability", False):
            result["reason"] = name + " 缺少真实模型概率，保留父版"
            return result
        try:
            for method in ("parent", "candidate"):
                checks[name][method + "_balanced_logloss"] = balanced_logloss(
                    stage["truth"], stage[method + "_probabilities"], stage[method + "_classes"], source_classes)
        except (KeyError, TypeError, ValueError) as error:
            result["reason"] = name + " 无法进行概率破平：" + str(error)
            return result
        checks[name]["balanced_logloss_improved"] = (
            checks[name]["candidate_balanced_logloss"] < checks[name]["parent_balanced_logloss"] - TOLERANCE)
    if all(value["balanced_logloss_improved"] for value in checks.values()):
        result.update(new_adopt=True, branch="probability_tie_break", reason="两阶段主指标均不退且至少一期打平，两阶段类别均衡 logloss 均严格改善")
    else:
        result["reason"] = "类别均衡 logloss 未在两阶段同时严格改善"
    return result
