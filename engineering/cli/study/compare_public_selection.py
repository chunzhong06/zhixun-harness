"""同一交付入口四方对照。先交齐各组盲预测，再从原始时钟和循环编号统一揭分。"""
from argparse import ArgumentParser
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, replace
from hashlib import sha256
from math import fsum, isfinite, sqrt
from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/"engineering/src")]
from engineering.cli.workflow.public_maintenance import assemble_public, run_prepared
from run_unseen_agents import ExternalReplay, frozen
from unseen_sources import bearing_data, engine_data, TEST_UNITS
from engineering.cli.study.run_rul_maintenance import PACKAGES
from verify_unseen_scores import remaining_cycles, remaining_minutes
from harness.contracts import digest
from harness.data.official_bearing import MEASURES
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import LocalStore, atomic_json

OUT = ROOT/"results/experiments/production-selection"
ARMS = ("fixed","project","mature","simple","original","project-luna")
WINDOW_ARMS = ("fixed", "project", "mature", "simple", "original")


def read(path):
    return json.loads(path.read_bytes())


def match_bearing_window(row):
    """把外部短波形两个频谱均方根换算到官方频点数，探索、确认与服务使用同一变换。"""
    values = list(row.values)
    if len(values) != 2 * len(MEASURES) or not row.sample_id.startswith("PHM12/"):
        raise ValueError("频点换算只适用于已核对的 PRONOSTIA 双通道输入")
    index = MEASURES.index("spectral_rms")
    scale = sqrt(1281 / 16385)
    values[index] *= scale
    values[len(MEASURES) + index] *= scale
    return replace(row, values=tuple(values))


def external_batches(task, official, *, match_window=False):
    """统一构造以官方训练池拟合、外部设备探索、确认和服务的十段输入。"""
    with ExitStack() as stack:
        previous = ExternalReplay(task, stack)
        extra = defaultdict(list)
        if task == "bearing":
            units, _, _ = bearing_data(full_test=True)
            if units.keys() != {"PHM12/" + name for name in TEST_UNITS}:
                raise ValueError("轴承设备与预注册不符")
            if any(row.unit_id in units for b in previous.batches for row in b.samples):
                raise ValueError("服务轴承出现在旧分组里")
            for members in units.values():
                for row in members:
                    phase = min((row.time_end - 1) // 5, 9)
                    extra[phase].append(replace(row, label=None, available_at=phase))
        batches = []
        for batch, old in zip(official, previous.batches, strict=True):
            external = [row for row in old.samples if task != "bearing" or row.role != "serve"]
            external.extend(extra[batch.ordinal])
            if any(row.label is not None for row in external if row.role == "serve"):
                raise ValueError("服务资料携带答案")
            rows = tuple(replace(row, values=tuple(map(float, row.values))) for row in external)
            if match_window:
                rows = tuple(match_bearing_window(row) for row in rows)
            fit = tuple(row for row in batch.samples if row.role == "fit")
            batches.append(replace(batch, samples=fit + rows))
        return batches


def run(task, arm, *, match_window=False):
    output = OUT/"short-window" if match_window else OUT
    folder = output/task/arm
    driver = {"mode":"rule"}
    if arm in ("simple","mature","project-luna"):
        driver = {"mode":"local-proposal","simple_proposal":True,"inbox":output/"queues"/arm,
                  "proposal_guidance":{"simple":None,"mature":"self_healing","project-luna":"device"}[arm]}
    policy = arm if arm in ("fixed","original") else "rule"
    with LocalStore(folder/"study-lock").exclusive():
        harness, official = assemble_public(task, ROOT/"data/official",folder,policy,driver,select_training_rows=True)
        batches = external_batches(task, official, match_window=match_window)
        harness.tools.values = {r.sample_id:r.values for b in batches for r in b.samples if r.role!="fit"}
        cfg = asdict(harness.config)
        # 原版从不触发维护，另外四组的预算、确认与训练配置须相同。
        cfg.pop("maintenance_trigger")
        controlled = {"task":task,"initial":harness.tools.source,"tools":harness.tools.identity,"config":cfg,
                      "batches":[digest([asdict(s) for s in b.samples]) for b in batches],
                      "service":[sorted(s.sample_id for s in b.samples if s.role=="serve") for b in batches]}
        frozen(folder/"controls.json",json.loads(json.dumps(controlled,ensure_ascii=False,allow_nan=False)))
        run_prepared(task,folder,policy,10,driver,harness,batches,count_rejected_proposals=True)


def score(task, *, match_window=False):
    output = OUT/"short-window" if match_window else OUT
    arms = WINDOW_ARMS if match_window else ARMS
    truth = remaining_cycles() if task=="engine" else remaining_minutes("full-test-manifest.json")
    controls = read(output/task/arms[0]/"controls.json")
    results, outputs, device_totals = {}, {}, {}
    for arm in arms:
        folder = output/task/arm
        if read(folder/"controls.json")!=controls:
            raise ValueError("各组看到的资料、预算或初始权重不一致")
        report, state = read(folder/"run.json"), read(folder/"state.json")
        rejected = set(report.get("rejected_proposals",[]))
        if len(report["rounds"])!=10 or any(e.get("kind")=="plan_invalid" and e.get("batch_id") not in rejected for e in state["events"]):
            raise ValueError("缺轮或有未入账的无效提案")
        if arm in ("mature","simple","project-luna"):
            audit = read(folder/"driver-exchanges.json")["exchanges"]
            failed = {e["batch_id"] for e in audit if e["status"]!="completed"}
            if len(audit)!=10 or len({e["batch_id"] for e in audit})!=10 or failed!=rejected:
                raise ValueError("代理原始回执不完整")
        residuals, seen, rounds = defaultdict(list), set(), []
        outputs[arm] = []
        for n, (r, ids) in enumerate(zip(report["rounds"],controls["service"],strict=True),1):
            predictions = read(folder/r["submitted"])["payload"]["predictions"]
            bundle = folder/"models"/r["model_artifact"]["key"]
            signature = sha256((bundle/"pipeline.bin").read_bytes()).hexdigest()
            if signature!=read(bundle/"manifest.json")["sha256"] or set(ids)!=predictions.keys() or seen.intersection(ids):
                raise ValueError("权重或服务交接标识不匹配")
            if len(r["trials"])>1:
                raise ValueError("超过候选预算")
            seen.update(ids)
            outputs[arm].append(predictions)
            errors = []
            for sid in ids:
                error = predictions[sid]-truth[sid]
                if not isfinite(error):
                    raise ValueError("误差非有限值")
                residuals[sid.rsplit("/",1)[0]].append(error)
                errors.append(error)
            rounds.append({"round":n,"selected_weights":signature,"rmse":rmse(errors),"trials":len(r["trials"]),
                           "adopted":r["current"]!=r["selected"],"actions":r["planned_actions"]})
        by_device = {unit:rmse(errors) for unit,errors in residuals.items()}
        device_totals[arm] = {u:(fsum(e*e for e in v),len(v)) for u,v in residuals.items()}
        all_errors = [e for errors in residuals.values() for e in errors]
        predicted = [p for window in outputs[arm] for p in window.values()]
        results[arm] = {"rmse":rmse(all_errors),"mean_bias":fsum(all_errors)/len(seen),"by_device":by_device,
                        "predicted_median":float(np.median(predicted)),"overestimates":sum(e>0 for e in all_errors),
                        "worst_device_rmse":max(by_device.values()),"devices":len(by_device),
                        "rows":len(seen),"trials":sum(r["trials"] for r in rounds),
                        "invalid_proposals":sorted(rejected),
                        "physical_fits":sum(1 for _ in (folder/"jobs").glob("*/training.json")),"rounds":rounds}
    reference = results["project"]["by_device"]
    for arm, info in results.items():
        info["devices_better_than_project"] = sum(v<reference[u] for u,v in info["by_device"].items())
    comparisons = {}
    units = sorted(reference)
    # 重抽整台设备，保留设备的所有时间点及各方法预测的配对关系。
    draws = np.random.default_rng(20261113).integers(0,len(units),size=(10000,len(units)))
    for arm in (name for name in arms if name != "project"):
        scores = []
        for name in ("project",arm):
            totals = np.asarray([device_totals[name][u] for u in units])
            sampled = totals[draws].sum(axis=1)
            scores.append(np.sqrt(sampled[:,0]/sampled[:,1]))
        intervals = np.quantile(scores[0]-scores[1],[.025,.975]).tolist()
        different = [sum(x[k]!=y[k] for k in x) for x,y in zip(outputs["project"],outputs[arm],strict=True)]
        comparisons[arm] = {"project_minus_arm_rmse":results["project"]["rmse"]-results[arm]["rmse"],
                            "paired_device_bootstrap_95":intervals,"different_predictions_by_round":different}
    targets = [truth[s] for s in seen]
    report = {"policies":results,"project_contrasts":comparisons,"always_zero_rmse":rmse(targets),
              "target_median":float(np.median(targets)),"target_max":max(targets),
              "cohort":"Full_Test_Set" if task=="bearing" else "PHM08 reused"}
    atomic_json(output/task/"scores.json",report)
    print(task,{a:round(info["rmse"],4) for a,info in results.items()})


def rmse(values):
    return sqrt(fsum(v*v for v in values)/len(values))


def candidate_check(task):
    """揭分后定位瓶颈。沿各条已发生的训练历史，逐测量检查实际产生的备选权重。"""
    scores = read(OUT/task/"scores.json")
    controls = read(OUT/task/"project/controls.json")
    units, _, _ = engine_data() if task=="engine" else bearing_data(full_test=True)
    values = {r.sample_id:r.values for rows in units.values() for r in rows if r.role=="serve"}
    truth = remaining_cycles() if task=="engine" else remaining_minutes("full-test-manifest.json")
    groups = {arm:(read(OUT/task/arm/"run.json"), read(OUT/task/arm/"state.json")["models"]) for arm in ARMS}
    pool_errors, project_errors, project_rounds, details = [], [], [], []
    with TemporaryDirectory(prefix="rul-candidate-check-") as work, ZipFile(ROOT/"data/official"/task/PACKAGES[task][0]) as archive:
        with _trainer(archive,Path(work),"B" if task=="bearing" else "C") as trainer:
            for j, ids in enumerate(controls["service"]):
                x = np.asarray([values[i] for i in ids],dtype=float)
                y = np.asarray([truth[i] for i in ids])
                cache, project = {}, []
                for arm,(report,models) in groups.items():
                    folder = OUT/task/arm
                    row = report["rounds"][j]
                    choices = {row["current"],row["selected"],*(t["model"] for t in row["trials"] if t["status"]=="completed")}
                    for mid in choices:
                        location = folder/"models"/models[mid]["artifact"]["key"]
                        path = location/"pipeline.bin"
                        key = read(location/"manifest.json")["sha256"]
                        if sha256(path.read_bytes()).hexdigest()!=key:
                            raise ValueError("候选权重身份不符")
                        if key not in cache:
                            p = trainer.Model.load(path).predict(x)
                            if p.shape!=y.shape or not np.isfinite(p).all():
                                raise ValueError("候选复算预测不完整")
                            cache[key]=p
                        if mid==row["selected"]:
                            issued = read(folder/row["submitted"])["payload"]["predictions"]
                            if not np.allclose(cache[key],[issued[i] for i in ids],rtol=1e-12,atol=1e-9):
                                raise ValueError("独立复算未复现实际交接预测")
                        if arm=="project":
                            project.append(cache[key])
                pe = np.square(np.asarray(project)-y)
                ae = np.square(np.asarray(list(cache.values()))-y)
                project_rounds.extend(pe[pe.sum(axis=1).argmin()].tolist())
                project_errors.extend(pe.min(axis=0).tolist())
                pool_errors.extend(ae.min(axis=0).tolist())
                details.append({"round":j+1,"rows":len(ids),"models_in_pool":len(cache)})
    result = {"served_project_rmse":scores["policies"]["project"]["rmse"],
              "project_best_one_version_each_round":float(np.sqrt(np.mean(project_rounds))),
              "project_best_prediction_each_row":float(np.sqrt(np.mean(project_errors))),
              "all_arms_best_prediction_each_row":float(np.sqrt(np.mean(pool_errors))),"rounds":details}
    atomic_json(OUT/task/"candidate-check.json",result)
    print(task, {k:v for k,v in result.items() if k!="rounds"})


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("engine","bearing"))
    parser.add_argument("arm",choices=(*ARMS,"score","candidates"))
    parser.add_argument("--match-window", action="store_true", help="统一 PRONOSTIA 短窗特征并在独立目录比较")
    args=parser.parse_args()
    if args.match_window and (args.task != "bearing" or args.arm not in (*WINDOW_ARMS, "score")):
        parser.error("短窗对照仅支持轴承的五个预定方案及统一计分")
    if args.arm=="score":
        score(args.task, match_window=args.match_window)
    elif args.arm=="candidates":
        candidate_check(args.task)
    else:
        run(args.task,args.arm, match_window=args.match_window)
