"""用邻站实测浓度预测本站，仅用先于发放的稀疏参考训练站点差值。"""
from dataclasses import asdict, dataclass
from itertools import product
import json

import numpy as np
from sklearn.linear_model import Ridge

from harness.integrations.pytrio_planner import ChatRequest


@dataclass(frozen=True)
class Rule:
    background: str
    correction: str
    days: int
    strength: float
    where: str = "all"
    limit: float = 0.0
    features: tuple[str, ...] = ()
    learner: str = "Ridge"
    parameters: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "features", tuple(self.features))
        parameters=self.parameters.items() if isinstance(self.parameters,dict) else self.parameters
        object.__setattr__(self,"parameters",tuple(sorted(tuple(pair) for pair in parameters)))


class StationTools:
    def __init__(self, features, donors, weather, nonlinear=False, flexible=False):
        self.names = list(donors) + ["median", "mean"] + list(weather)
        self.x = np.asarray(features, dtype=float)
        self.backgrounds = list(donors) + ["median", "mean"]
        self.weather = list(weather)
        self.static = Rule("median", "bias", 180, 0)
        self.flexible=flexible
        self.feature_terms = []
        if nonlinear:
            # 名称直接绑定现有输入，组合只使用该时刻可见的测量，允许现场自行组装。
            original_names = self.names.copy()
            extra = []
            for i, left in enumerate(donors):
                for right in original_names[i:]:
                    if right in ("median", "mean", "weekend"):
                        continue
                    term = left + "*" + right
                    self.feature_terms.append(term)
                    extra.append(self.x[:, self.names.index(left)] * self.x[:, self.names.index(right)])
            self.names.extend(self.feature_terms)
            self.x = np.column_stack((self.x, *extra))

    @staticmethod
    def fixed(backgrounds):
        for name, kind, days, strength in product(backgrounds, ("bias", "linear", "weather", "network"),
                                                  (21, 60, 180), (.25, .5, .75, 1.)):
            yield Rule(name, kind, days, strength)

    def train(self, rule, fit, labels):
        idx = np.flatnonzero(fit)
        if not len(idx):
            raise ValueError("可用于拟合的到期参考为0条")
        idx = idx[idx >= idx[-1] - 24 * rule.days + 1]
        available = len(idx)
        selection = "保留全部历史行"
        if rule.where != "all":
            name, side = rule.where.split(":")
            col = self.x[idx, self.names.index(name)]
            idx = idx[col <= rule.limit] if side == "low" else idx[col > rule.limit]
            operator = "<=" if side == "low" else ">"
            selection = (f"条件 {name} {operator} {rule.limit:g}；"
                         f"该字段实际范围 [{float(col.min()):g}, {float(col.max()):g}]")
        if len(idx) < 12:
            raise ValueError(f"历史{rule.days}日筛选前{available}条，{selection}，筛选后{len(idx)}条；训练至少需要12条")
        background = self.names.index(rule.background)
        residual = labels[idx] - self.x[idx, background]
        columns = [] if rule.correction == "bias" else [background]
        if rule.correction in ("weather", "network"):
            columns.extend(self.names.index(name) for name in self.weather)
        if rule.correction == "network":
            columns = list(dict.fromkeys(columns + [self.names.index(n) for n in self.backgrounds[:-2]]))
        columns = list(dict.fromkeys(columns + [self.names.index(n) for n in rule.features]))
        center = self.x[idx][:, columns].mean(axis=0)
        scale = self.x[idx][:, columns].std(axis=0)
        scale[scale < 1e-8] = 1.
        intercept, coef = float(residual.mean()), np.zeros(len(columns))
        if columns:
            from harness.models.station_learners import make
            model = make(rule.learner,rule.parameters).fit((self.x[idx][:, columns] - center) / scale, residual)
            if isinstance(model,Ridge):
                intercept, coef = float(model.intercept_), model.coef_
            else:
                intercept,coef=0.,model
        return (background, columns, center, scale, intercept, coef, rule.strength)

    def issue(self, model, rows):
        background, columns, center, scale, intercept, coef, strength = model
        x = self.x[rows]
        standardized=(x[:, columns] - center) / scale
        residual=coef.predict(standardized) if hasattr(coef,"predict") else intercept+standardized@coef
        return np.maximum(0., x[:, background] + strength * residual)

    def parse(self, raw):
        r = Rule(**raw["recipe"])
        from harness.models.station_learners import make
        learner=make(r.learner,r.parameters)
        default=make("Ridge")
        is_default_ridge=type(learner) is type(default) and learner.get_params()==default.get_params()
        if not self.flexible and not is_default_ridge:
            raise ValueError("当前入口仅开放原学习器")
        # 显式写出默认值不改变学习器。常量修正也须按实际参数判等，
        # 否则现场提交历史参照的同一配方会被误拒并占掉一次实训机会。
        if r.correction=="bias" and not r.features and not is_default_ridge:
            raise ValueError("无输入的常量偏差仅支持默认岭回归，其他学习器需选择带输入的校正方式")
        filters = ["all", *(f"{n}:{side}" for n in self.names for side in ("low", "high"))]
        if (r.background not in self.backgrounds or r.correction not in ("bias", "linear", "weather", "network")
                or r.days not in (21, 60, 180) or r.strength not in (.25, .5, .75, 1.)
                or r.where not in filters or not np.isfinite(r.limit) or not raw["hypothesis"].strip()
                or (len(r.features) > 4 and r.features != tuple(self.feature_terms))
                or any(n not in self.feature_terms for n in r.features)):
            raise ValueError("提案不在允许范围")
        return r


def error(y, p):
    return float(np.sqrt(np.mean((y-p)**2)))


def tests(tools, model, windows, y):
    return np.array([error(y[m], tools.issue(model, m)) for m in windows])


def gate(old, new):
    # 探索两段等权，最后一段只在提名之后检查；单段胜负有可能反转。
    return bool(np.mean(new[:2]**2) < np.mean(old[:2]**2) and new[2] < old[2])


def facts_for(tools, active, windows, y, feedback=None):
    facts = {}
    for i, mask in enumerate(windows[:2]):
        x = tools.x[mask]
        pred = tools.issue(active, mask)
        if feedback is not None:
            from harness.models.sparse_reference import correct
            clock = np.flatnonzero(mask)
            pred = correct(pred[:,None],y[mask],clock,feedback)[:,0]
        diff = pred-y[mask]
        stats = {}
        for j, name in enumerate(tools.names):
            threshold = float(np.median(x[:,j]))
            lo = x[:,j] <= threshold
            if lo.any() and (~lo).any():
                stats[name] = [round(threshold,2),round(float(diff[lo].mean()),2),round(float(diff[~lo].mean()),2)]
        raw = {n:round(error(y[mask], x[:,tools.names.index(n)]),3) for n in tools.backgrounds}
        facts[f"window_{i+1}"] = {"n":int(mask.sum()),"parent_rmse":round(error(y[mask],pred),3),
                                  "raw_background_rmse":raw,"median_and_mean_error_low_high":stats}
    return facts


def history_trials(tools, fit, y, rules):
    """用三次早于正式探索的21日历史交付模拟比较配方；拟合与检验严格按时间先后。"""
    end = np.flatnonzero(fit)[-1] + 1
    clock = np.arange(len(fit))
    scores = np.empty((len(rules), 3))
    parent_losses = []
    fit_count = 0
    blocks = []
    for fold in range(3):
        first = end - (3-fold)*21*24
        train = fit & (clock < first)
        test = fit & (clock >= first) & (clock < first + 21*24)
        if train.sum() < 80 or test.sum() < 16:
            raise ValueError("历史回放缺少足够的早期拟合或后续参考")
        blocks.append({"last_fit_hour":int(clock[train][-1]),"first_test_hour":int(clock[test][0]),
                       "last_test_hour":int(clock[test][-1]),"test_rows":int(test.sum())})
        parent=tools.train(tools.static,train,y)
        parent_losses.append(error(y[test],tools.issue(parent,test)))
        r=tools.static
        cache = {(r.background,r.correction,r.days,r.features,r.learner,r.parameters):parent}
        fit_count+=1
        for i,r in enumerate(rules):
            # 衰减比例只改变应用校正的幅度，同一训练结果可供四种比例共同使用。
            key=(r.background,r.correction,r.days,r.features,r.learner,r.parameters)
            if key not in cache:
                cache[key]=tools.train(r,train,y)
                fit_count+=1
            candidate=(*cache[key][:-1],r.strength)
            scores[i,fold]=error(y[test],tools.issue(candidate,test))
    relative=scores / np.maximum(parent_losses,1.e-8)
    risk=relative.mean(axis=1)
    order=np.argsort(risk,kind="stable")
    def record(i):
        return {"recipe":asdict(rules[i]),"fold_rmse":scores[i].round(3).tolist(),
                "mean_relative_rmse":round(float(risk[i]),4)}
    top=[record(i) for i in order[:6]]
    groups={k:record(next(i for i in order if rules[i].background==k)) for k in tools.backgrounds}
    families={kind:record(next(i for i in order if bool(rules[i].features)==enabled))
              for kind,enabled in (("linear",False),("with_products",True))
              if any(bool(r.features)==enabled for r in rules)}
    learners={name:record(next(i for i in order if rules[i].learner==name)) for name in dict.fromkeys(r.learner for r in rules)}
    brief={"scope":"Three earlier 21-day target-site backtests; each model fit only on preceding data.",
           "baseline_fold_rmse":np.round(parent_losses,3).tolist(),"overall":top,
           "best_by_background":groups,"best_by_feature_family":families,"best_by_learner":learners}
    return brief,{"blocks":blocks,"fits":fit_count,"rmse":scores.tolist(),"ordered":order.tolist()}


def plan_site(inbox, site, arm, tools, parent, fit, windows, y, memo, history=None, paper_base=None, online=False,
              parent_feedback=None):
    """驱动没有工具调用权或文件读取需求；只交给它两个到期窗口和自己的实测结果。"""
    prompt = ("Predict target-site PM2.5 from simultaneous neighbor readings. Diagnose from two different past "
              "windows (median, mean prediction-minus-observation below/above median). Choose one concrete recipe, "
              "then revise it using its measured errors. Residual = target minus background; bias fits its mean, "
              "linear fits it against background, weather adds local weather, network also adds all raw donors. Strength scales the correction. "
              "Filter selects past training hours only. Return {hypothesis: short sentence, recipe:{background, "
              "correction, days, strength, where, limit}}. 'limit' is a numeric threshold, use 0 for 'all'.")
    if arm == "paper":
        paper_base=paper_base or Rule("median","weather",180,1.)
        prompt += (" Follow Self-Healing ML: hypothesize a training subset that fixes the drift, fit and backtest. "
                   "Keep the a-priori learner listed under allowed. "
                   "Select any feature threshold in either direction, or keep all rows.")
    else:
        prompt += " Choose the background, history, learner, attenuation and optional training condition yourself."
    options = {"background":tools.backgrounds, "days":[21,60,180], "correction":["bias","linear","weather","network"],
               "strength":[.25,.5,.75,1.], "where":["all", *(n+":"+s for n in tools.names for s in ("low","high"))]}
    if arm=="paper":
        options.update(background=[paper_base.background],days=[paper_base.days],
                       correction=[paper_base.correction],strength=[paper_base.strength])
    context = {"task_id":arm,"site":site,"allowed":options,"past":facts_for(tools,parent,windows,y,feedback=parent_feedback),
               "trials":[],"parent_rule":asdict(tools.static)}
    if parent_feedback is not None:
        context["parent_feedback"]={"gain":parent_feedback[0],"hours":parent_feedback[1]}
    if history is not None:
        context["past_trained_model_comparisons"]=history
    if tools.flexible:
        from harness.models.station_learners import DEFAULTS
        context["learner_defaults"]=DEFAULTS
        prompt += (" Also supply recipe.learner and recipe.parameters. Choose a learner from learner_defaults "
                   "and optionally override scalar constructor parameters. The learner fits target-minus-background "
                   "on standardized inputs. CPU budget: n_jobs=1, at most 128 trees and 256 iterations. "
                   "Use the measured model-family comparisons to choose which mechanism to test next.")
        if arm=="paper":
            context["fixed_learner"]={"learner":paper_base.learner,"parameters":dict(paper_base.parameters)}
            prompt += " Preserve the learner and parameters in fixed_learner while hypothesizing the training subset."
    if tools.feature_terms:
        prompt += (" You can add up to four feature products in recipe.features, chosen from allowed_features. "
                   "Each product is a currently visible measurement pair or measurement times clock covariate. "
                   "Choose features to address the observed residual patterns and examine the actual result. "
                   "An empty list keeps the original features. A historical model may contain the full list; "
                   "that exact list is also allowed as a fixed reference recipe.")
        context["allowed_features"] = tools.feature_terms if arm != "paper" else list(paper_base.features)
        if arm == "paper":
            prompt += " Keep recipe.features exactly equal to fixed_features, preserving the historical learner."
            context["fixed_features"] = list(paper_base.features)
    if online:
        prompt += (" The published model can use a periodic past target reference. Add top-level feedback "
                   "{gain,hours}; gain times the last measured error decays exponentially with elapsed hours, "
                   "and applies only AFTER that measurement. Choose from allowed_feedback. Recent window scores "
                   "use simulated online delivery with the same timing. Additional covariates are clock/calendar, "
                   "not measured meteorology. You may change the feedback setting in your second trial.")
        from harness.models.sparse_reference import SETTINGS
        context["allowed_feedback"]=[{"gain":g,"hours":h} for g,h in SETTINGS]
    candidates = {}
    for turn in (1,2):
        context["batch_id"] = f"{site}-trial-{turn}"
        body={"model":"gpt-6-luna","messages":[{"role":"system","content":prompt},
              {"role":"user","content":json.dumps(context,allow_nan=False,ensure_ascii=False)}]}
        reply=inbox.send(ChatRequest("local-inbox",body,200))
        raw=json.loads(reply.body["choices"][0]["message"]["content"])
        trial={"proposal":raw}
        try:
            rule=tools.parse(raw)
            if any(getattr(rule,k) not in values for k,values in options.items()):
                raise ValueError("超出本组工具")
            if arm == "paper" and rule.features != paper_base.features:
                raise ValueError("论文组保留历史提名的输入特征，仅改变训练子集与反馈")
            if arm=="paper":
                from harness.models.station_learners import make
                proposed=make(rule.learner,rule.parameters)
                fixed=make(paper_base.learner,paper_base.parameters)
                # 默认值是否显式写出不改变模型，比较真正传给学习器的完整参数。
                if type(proposed) is not type(fixed) or proposed.get_params()!=fixed.get_params():
                    raise ValueError("论文组保留历史最强学习器及参数")
            setting=None
            if online:
                from harness.models.sparse_reference import SETTINGS, setting_of
                setting=setting_of(raw["feedback"])
            key=(rule,setting) if online else rule
            if key in candidates:
                raise ValueError("重复试验")
            if rule not in memo:
                memo[rule]=tools.train(rule,fit,y)
            model=memo[rule]
            if online:
                from harness.models.sparse_reference import reference_losses
                # 在未涉及独立确认的时段只给试验返回两组误差。
                losses=reference_losses(tools,model,windows[:2],y,setting)
            else:
                losses=tests(tools,model,windows[:2],y)
            candidates[key]=(float(np.mean(losses**2)),model)
            trial["observed_rmse"]=losses.tolist()
            if tools.feature_terms:
                # 分组误差和总分使用同一份已发预测，反馈修正也属于候选行为。
                trial["remaining_errors"] = facts_for(tools,model,windows,y,feedback=setting)
        except (KeyError,TypeError,ValueError) as exc:
            trial["rejected"]=str(exc)
        context["trials"].append(trial)
    parent_errors=tests(tools,parent,windows,y)
    if parent_feedback is not None:
        from harness.models.sparse_reference import reference_losses
        parent_errors=reference_losses(tools,parent,windows,y,parent_feedback)
    decision={"trials":context["trials"],"adopted":False,"parent_errors":parent_errors.tolist()}
    if candidates:
        winner=min(candidates,key=lambda k:candidates[k][0])
        selected=candidates[winner][1]
        if online:
            losses=reference_losses(tools,selected,windows,y,winner[1])
            decision["feedback"]={"gain":winner[1][0],"hours":winner[1][1]}
        else:
            losses=tests(tools,selected,windows,y)
        decision.update(nominated=asdict(winner[0] if online else winner),errors=losses.tolist())
        if gate(np.array(decision["parent_errors"]),losses):
            parent=selected
            decision["adopted"]=True
    return parent,decision,list(candidates)
