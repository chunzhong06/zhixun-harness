"""同一芯片、同一装机费，比较新增波形配准在不同购瓶顺序下的实际分类。"""
import json
import numpy as np
from sklearn.metrics import f1_score, recall_score

from score_calibration_query import measurements, chemistry, GASES, CHEM, GasBase
from score_paired_design import DEST as PAIRS, LocalStore, LocalAcceptance, atomic_json, digest
from score_settling_cycle import DEST as PRIOR
from harness.models.calibration_query import query_checkpoints
from harness.models.bottle_alignment import ACTIVE, FROZEN, PRIMARY, PAPER, AlignmentQuery, AlignmentDeployment

DEST = PRIOR.parent/"bottle-alignment"


def previous_packet(folder, key):
    packet = json.loads((folder/"accepted"/(key+".json")).read_bytes())["payload"]
    if digest(packet) != key:
        raise ValueError("上轮已交付记录的身份变化")
    return packet


def evaluate_candidates(stage, query_cls, deployment_cls, active, frozen, primary, paper, extra_parent=None):
    ids, features, truth = measurements()
    source_chip, *chips = sorted(features)
    source = features[source_chip][:120]
    chem = chemistry(ids[:120])
    index = {int(uid): i for i, uid in enumerate(ids[:120])}
    pairs = json.loads((PAIRS/"scores.json").read_bytes())
    prior = json.loads((PRIOR/"scores.json").read_bytes())
    extended = json.loads((extra_parent/"scores.json").read_bytes()) if extra_parent else None
    receiver = LocalAcceptance()
    receiver.output = stage/"accepted"
    events = {}
    for gas in GASES:
        j = CHEM.index(gas)
        query = query_cls(source, chem, j, GasBase().fit(source, chem[:,j]))
        for chip in chips:
            x = features[chip]
            for size in (20,80):
                key = f"{chip}-{gas}-{size}"
                suffix = f"{gas}-{size}"
                parent = previous_packet(PRIOR, prior["requests"][key])
                old = previous_packet(PAIRS, parent["上轮"])
                if (pairs["requests"][key] != parent["上轮"] or old["配方"] != ids[120:].tolist()
                        or parent["配方"] != old["配方"] or query.cut != parent["阈值"]):
                    raise ValueError("装机配对、阈值或服务名单发生变化")
                public, previous_method = prior["controls"]["固定"][suffix]
                paper_query, paper_method = prior["controls"]["论文"][suffix]
                history = {public:old["装机"][public], paper_query:old["装机"][paper_query]}
                values = {public:{"上轮最强固定":parent["浓度"][public][previous_method]}}
                values.setdefault(paper_query,{})["上轮最强论文"] = parent["浓度"][paper_query][paper_method]
                previous_rules = [public, paper_query]
                predecessor = prior["requests"][key]
                if extended is not None:
                    predecessor = extended["requests"][key]
                    earlier = json.loads((extra_parent/"accepted"/(predecessor+".json")).read_bytes())["payload"]
                    if (digest([key, earlier]) != predecessor or earlier["配方"] != old["配方"]
                            or earlier["阈值"] != query.cut):
                        raise ValueError("前次逐瓶装机的实际发放已变化")
                    history.update(earlier["装机"])
                    for role, prefix in (("固定","前次最强固定"), ("论文","前次最强论文")):
                        which, name = extended["controls"][role][suffix]
                        previous_rules.append(which)
                        values.setdefault(which,{})[prefix] = earlier["浓度"][which][name]
                for rule in dict.fromkeys((active, frozen, *previous_rules)):
                    if rule in (active, frozen):
                        ix, measured = next(query_checkpoints(query, rule, (size,), lambda i:x[i].copy()))
                        history[rule] = ids[ix].tolist()
                    else:
                        ix = np.array([index[uid] for uid in history[rule]])
                        measured = x[ix]
                    if len(ix) != size or len(set(ix)) != size:
                        raise ValueError("查询数量与已付费标气不一致")
                    installed = deployment_cls().fit(source, chem, ix, measured, j)
                    predictions = installed.predict(x[120:])
                    if rule in old["浓度"]:
                        for name, estimates in predictions.items():
                            if name in old["浓度"][rule] and not np.allclose(estimates, old["浓度"][rule][name],
                                                                            rtol=1e-10, atol=1e-8):
                                raise ValueError("历史基线未复现")
                    values.setdefault(rule, {}).update({n:v.tolist() for n,v in predictions.items()})
                payload = {"装机":history, "浓度":values, "阈值":query.cut, "配方":ids[120:].tolist(),
                           "上轮":predecessor}
                req = digest([key, payload])
                if not receiver.submit(req, payload)["accepted"]:
                    raise ValueError("装机后的实际预测未交接")
                events[key] = req
    scores = {}
    for key, req in events.items():
        p = json.loads((receiver.output/(req+".json")).read_bytes())["payload"]
        if digest([key,p]) != req:
            raise ValueError("发放结果变化")
        gas = key.split("-")[1]
        labels = truth[gas][120:]>p["阈值"]
        scores[key] = {}
        for rule, models in p["浓度"].items():
            row = scores[key][rule] = {}
            for name, estimates in models.items():
                predicted = np.asarray(estimates)>p["阈值"]
                row[name] = {"F1":float(f1_score(labels,predicted,average="macro")),
                             "弱类召回":float(recall_score(labels,predicted,average=None).min())}
    controls = {"固定":{}, "论文":{}}
    for suffix, (incumbent, _) in prior["controls"]["固定"].items():
        keys = [k for k in scores if k.endswith(suffix)]
        for role in controls:
            # 固定端也可用留组试训发布规则。现场独有的增量须来自实际改变购瓶顺序。
            choices = [(r,n) for r,methods in scores[keys[0]].items() for n in methods if
                       (role=="固定" and (r in (incumbent, frozen, FROZEN) or r in FIXED_QUERIES
                           or (extended and r==extended["controls"]["固定"][suffix][0]))
                        or role=="论文" and (n==paper or "论文" in n or n=="全曲线回源"))]
            controls[role][suffix] = max(choices, key=lambda v:np.mean([scores[k][v[0]][v[1]]["F1"] for k in keys]))
    cases = {}
    for key, s in scores.items():
        suffix = key.split("-",1)[1]
        cases[key] = {"现场":s[active][primary]}
        for role, policy in controls.items():
            r,n = policy[suffix]
            cases[key][role] = s[r][n]
    means = {role:{m:float(np.mean([v[role][m] for v in cases.values()])) for m in ("F1","弱类召回")}
             for role in ("现场","固定","论文")}
    gap = means["现场"]["F1"]-max(means["固定"]["F1"],means["论文"]["F1"])
    decision = {r+"F1":v["F1"] for r,v in means.items()}
    decision.update({"差值":gap, "进入独立":gap>=.01 and means["现场"]["弱类召回"]>=means["固定"]["弱类召回"]-.02})
    atomic_json(stage/"scores.json", {"decision":decision,"means":means,"controls":controls,
                                     "cases":cases,"scores":scores,"requests":events})
    print(json.dumps({"装机配准实际发放":decision},ensure_ascii=False))


from harness.models.calibration_query import RULES as BASE_RULES, ACTIVE as OLD_ACTIVE
from harness.models.paired_design import DIRECT, UNIFORM, NO_SITE
FIXED_QUERIES = (set(BASE_RULES)-{OLD_ACTIVE}) | {DIRECT,UNIFORM,NO_SITE}


if __name__ == "__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("同一装机配准实验已完成")
        evaluate_candidates(DEST,AlignmentQuery,AlignmentDeployment,ACTIVE,FROZEN,PRIMARY,PAPER)
