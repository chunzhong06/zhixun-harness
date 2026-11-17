"""寿命续训按官方训练分位数选样；只公开由当轮探索资料计算的误差摘要。"""
import numpy as np

from harness.core.contracts import digest
from harness.learning.models.official_rul_cpu import RulPilotTools


class SelectableRulTools(RulPilotTools):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rows = list(self.fit.values())
        names = ["time", *self.package.task.features]
        columns = np.asarray([(s.time_end, *s.values) for s in rows], dtype=float).T
        self.fields = []
        self.choices = {}
        for j, (name, values) in enumerate(zip(names, columns, strict=True)):
            thresholds = []
            for i, cutoff in enumerate(np.unique(np.quantile(values, [.25, .5, .75]))):
                left = values <= cutoff
                if left.all() or not left.any():
                    continue
                label = f"f{j}-q{i}"
                for side, mask in (("le", left), ("gt", ~left)):
                    self.choices[f"{label}-{side}"] = frozenset(row.sample_id for row, use in zip(rows, mask) if use)
                thresholds.append((label, float(cutoff), int(left.sum())))
            self.fields.append((name, thresholds))
        self.identity = digest([self.identity, "rul-selection-v1", self.fields])

    def fit_manifest_context(self, batch, baseline):
        rows = [s for s in batch.samples if s.role == "explore" and s.available_at == batch.ordinal and s.label is not None]
        if not rows or not baseline:
            return {"names": ["full"], "implicit_full": True, "fields": []}
        errors = np.asarray([baseline["predictions"][s.sample_id] - s.label for s in rows])
        observations = np.asarray([(s.time_end, *s.values) for s in rows], dtype=float).T
        fields = []
        for (name, cutoffs), values in zip(self.fields, observations, strict=True):
            splits = []
            contrast = 0.
            for label, cutoff, n_fit in cutoffs:
                left = values <= cutoff
                a = float(errors[left].mean()) if left.any() else None
                b = float(errors[~left].mean()) if not left.all() else None
                if min(int(left.sum()), int((~left).sum())) >= 3:
                    contrast = max(contrast, abs(a-b))
                splits.append({"threshold":cutoff, "le":label+"-le", "gt":label+"-gt", "fit_le":n_fit,
                               "explore_le":int(left.sum()), "bias_le":a, "bias_gt":b})
            if splits:
                fields.append((name, contrast, splits))
        fields.sort(key=lambda field: (field[0] != "time", -field[1], field[0]))
        shown = [{"field":name, "splits":splits} for name, _, splits in fields[:4]]
        names = [s[k] for f in shown for s in f["splits"] for k in ("le", "gt")]
        return {"names":["full", *names], "implicit_full":True, "fields":shown, "fit_n":len(self.fit),
                "explore_n":len(rows), "bias":"预测减真实寿命。le 表示小于等于阈值，gt 表示大于阈值。"}

    def select_fit_rows(self, name, eligible):
        ids = self.fit.keys() if name == "full" else self.choices[name]
        rows = tuple(row for row in eligible if row.sample_id in ids)
        if len(rows) != len(ids) or {r.sample_id for r in rows} != ids or any(self.fit[r.sample_id] != r for r in rows):
            raise ValueError("实际训练资料与选定清单不符")
        return rows

    def _fit_request(self, task, action, rows, parent):
        if action.fit_manifest in (None, "full"):
            return super()._fit_request(task, action, rows, parent)
        selected = self.select_fit_rows(action.fit_manifest, rows)
        if len(rows) != len(selected):
            raise ValueError("训练请求混入清单外样本")
        # 沿用父权重、官方原包和训练环境的全部校验，工作者只收到本次选定的训练编号。
        request = super()._fit_request(task, action, self.fit.values(), parent)
        return {**request, "fit_ids":sorted(r.sample_id for r in selected), "fit_manifest":action.fit_manifest}


class CoverageRulTools(SelectableRulTools):
    """诊断仍在原训练分界上比较。实际每份清单等长，取行规则及种子进入恢复身份。"""

    def __init__(self, *args, allocation, **kwargs):
        super().__init__(*args, **kwargs)
        if allocation not in ("targeted", "mixed", "uniform"):
            raise ValueError("未知的定向选样配额")
        self.allocation = allocation
        ordered = sorted(self.fit, key=lambda key: digest([17, key]))
        self.fit_budget = max(1, round(len(ordered) / 4))
        self.fallback = "budget-full"
        full_sample = frozenset(ordered[:self.fit_budget])
        for name, region in self.choices.items():
            if allocation == "uniform":
                self.choices[name] = full_sample
                continue
            quota = self.fit_budget if allocation == "targeted" else (self.fit_budget+1)//2
            focused = [key for key in ordered if key in region][:quota]
            kept = set(focused)
            background = [key for key in ordered if key not in kept][:self.fit_budget-len(kept)]
            self.choices[name] = frozenset(focused + background)
        self.choices[self.fallback] = full_sample
        self.identity = digest([self.identity, "coverage-v1", allocation, self.fit_budget, ordered])

    def fit_manifest_context(self, batch, baseline):
        catalog = super().fit_manifest_context(batch, baseline)
        # full 保持旧后端“原始全量”语义；预算下的全池抽样有单独的清单名。
        catalog["names"] = [self.fallback if n == "full" else n for n in catalog["names"]]
        catalog.update(fallback=self.fallback, implicit_full=False, allocation=self.allocation, fit_budget=self.fit_budget)
        return catalog
