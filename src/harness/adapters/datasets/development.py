"""已公开数据的本地开发协议；输出统一清单与独立评价答案。

本模块只准备资料，不启动训练或预测。划分和反馈时间是显式的本地约定；
正式评测到来后，由平台适配器提供比赛事件，不能自动继承这里的时序。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path

from harness.core.contracts import BatchContext, digest
from harness.adapters.datasets.gas_input import gas_task, read_gas_batches, bind_gas_batch, REGISTERED_SHA256
from harness.adapters.datasets.cmapss_input import cmapss_task, read_cmapss, trajectory_samples, test_answers


def _roles(keys, *, seed=17):
    """固定身份排序划分；不按标签分层，不为少数类借用确认集。"""
    ordered = sorted(keys, key=lambda key: digest([seed, key]))
    if len(ordered) < 3 or len(set(ordered)) != len(ordered):
        raise ValueError("本地划分需要至少三个互异身份")
    held = max(1, len(ordered)//5)
    return dict(zip(ordered, ["fit"]*(len(ordered)-2*held)+["explore"]*held+["adopt"]*held))


def gas_development(path, *, seed=17, expected_sha256=REGISTERED_SHA256):
    """每批80%作为开发资料、20%先预测后释放反馈；后两批作为整批未来资料。

    开发内部再按60/20/20分配fit/explore/adopt。未知独立单位保持None，
    行身份的分离仅支持数据去重；不会声称同一来源行天然统计独立。
    """
    task, batches, answers = gas_task(), [], []
    pending = ()
    for source in read_gas_batches(path, expected_sha256=expected_sha256):
        ordinal = source.batch_number
        ordered = sorted((r.sample_id for r in source.records), key=lambda key: digest([seed, key]))
        n_serve = len(ordered) if ordinal >= 9 else max(1, len(ordered)//5)
        serve_ids = set(ordered[-n_serve:])
        development_ids = [k for k in ordered if k not in serve_ids]
        roles = _roles(development_ids, seed=seed) if development_ids else {}
        assignments = {r.sample_id: {"role": "serve" if r.sample_id in serve_ids else roles[r.sample_id],
                                    "available_at": ordinal} for r in source.records}
        batch = bind_gas_batch(source, batch_id=f"gas-source-{ordinal}", ordinal=ordinal, assignments=assignments)
        current = {r.sample_id: r for r in source.records}
        # serve答案只在下一次事件以revision2交回，不回流为新的采用证据。
        revealed = tuple(replace(s, revision=2, role="explore", label=label, available_at=ordinal)
                         for s, label in pending)
        batches.append(replace(batch, samples=(*revealed, *batch.samples)))
        pending = tuple((s, current[s.sample_id].label) for s in batch.samples if s.role == "serve") if ordinal < 9 else ()
        answers.extend({"sample_id": s.sample_id, "label": current[s.sample_id].label,
                        "prediction_ordinal": ordinal, "reveal_ordinal": ordinal+1 if ordinal < 9 else None}
                       for s in batch.samples if s.role == "serve")
    return task, tuple(batches), {"protocol": "gas-local-development-v1", "seed": seed,
        "roles": "source1..8: 80%开发内部60/20/20；20%先预测次批反馈；source9..10只评价",
        "independent_units": "unknown", "official": False, "answers": answers}


def cmapss_development(path, *, subset="FD001", window=20, stride=10, seed=17, rul_cap=None):
    """完整训练发动机按设备留出；测试末端输入和测试答案严格分开。"""
    task = cmapss_task(subset, window=window)
    trains = read_cmapss(path, subset=subset, partition="train")
    roles = _roles([t.unit_id for t in trains], seed=seed)
    rows = tuple(s for t in trains for s in trajectory_samples(t, window=window, stride=stride,
                    role=roles[t.unit_id], ordinal=1, rul_cap=rul_cap))
    tests = read_cmapss(path, subset=subset, partition="test")
    serving = tuple(s for t in tests for s in trajectory_samples(t, window=window, stride=stride,
                       role="serve", ordinal=2, final_only=True, short_prefix="available"))
    labels = test_answers(path, subset, tests)
    return task, (BatchContext(task.task_id, "completed-training-trajectories", 1, rows),
                  BatchContext(task.task_id, "unseen-test-prefixes", 2, serving)), {
        "protocol": "cmapss-local-development-v1", "official": False, "seed": seed,
        "condition_group_key": "rounded_settings_signed_zero_canonical_v2",
        "window": window, "stride": stride, "rul_cap": rul_cap,
        "short_test_prefix": "使用实际已观察前缀计算均值和变化；不填充未来、不删除设备",
        "roles": "完整训练轨迹按发动机60/20/20；测试只预测最后可见周期",
        "label_availability": "离线完整train寿命已公开；不模拟在线失效反馈",
        "answers": [{"sample_id": s.sample_id, "label": labels[s.unit_id]} for s in serving]}


def export_development(task, batches, protocol, folder):
    """运行清单置于runtime子目录，评价答案放在相邻evaluation目录。

    使用已存在且非空的目录会拒绝，避免改写旧证据；写入后由read_package
    检查身份和用途。调用者应将不同配置写入不同目录。
    """
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("输出目录非空，请选择新的开发输入目录")
    runtime, evaluation = folder / "runtime", folder / "evaluation"
    runtime.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir()
    columns = {k: k for k in ("sample_id", "revision", "label", "role", "group", "available_at", "unit_id", "time_start", "time_end")}
    columns["features"] = list(task.features)
    manifest = {"schema_version": 1, "kind": "harness_table_manifest", "origin": "user_supplied",
        "task": asdict(task), "columns": columns, "role_values": {r: r for r in ("fit", "explore", "adopt", "serve")},
        "label_values": {str(c): c for c in task.classes}, "batches": []}
    summary = []
    for batch in batches:
        name = f"batch-{batch.ordinal:03d}.jsonl"
        with (runtime / name).open("w", encoding="utf-8", newline="\n") as out:
            for s in batch.samples:
                raw = asdict(s)
                raw.pop("values")
                raw.update(zip(task.features, s.values))
                out.write(json.dumps(raw, ensure_ascii=False, allow_nan=False)+"\n")
        manifest["batches"].append({"path": name, "batch_id": batch.batch_id, "ordinal": batch.ordinal, "format": "jsonl"})
        summary.append({"ordinal": batch.ordinal, "roles": dict(Counter(s.role for s in batch.samples)),
                        "units": len({s.unit_id for s in batch.samples if s.unit_id}), "rows": len(batch.samples)})
    (runtime/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    from harness.adapters.datasets.package_input import read_package
    package = read_package(runtime/"manifest.json")
    from harness.evaluation.final_answers import service_scope
    scope = service_scope(task, package.batches)
    answers = {"schema_version": 1, "task_signature": task.signature,
        "input_digest": digest([asdict(b) for b in package.batches]), "protocol": protocol["protocol"],
        "answers": [{**row, "label_revision": row.get("label_revision", 1),
            **{k:scope[row["sample_id"]][k] for k in ("batch_id", "prediction_ordinal", "prediction_revision")}}
            for row in protocol["answers"]]}
    (evaluation/"answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder/"protocol.json").write_text(json.dumps({**{k:v for k,v in protocol.items() if k != "answers"},
        "task_signature": task.signature, "batches": summary, "training": False, "model_calls": False},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return package.inspection


def bearing_development(root, specification, *, seed=17):
    """按设备分离开发和留出输入；逐个CSV提取特征并核对完整寿命清单。

    specification显式声明condition、bearing和failure_index。全寿命是否完整
    由来源表和文件序号共同核对，不能把任意下载子集的最后一行当成失效。
    """
    from harness.adapters.datasets.bearing_input import bearing_task, read_vibration, bind_vibration
    root = Path(root).resolve()
    if len(specification) < 4:
        raise ValueError("设备留出协议需要至少四台明确设备")
    ids = [f"{item['condition']}:{item['bearing']}" for item in specification]
    if len(set(ids)) != len(ids):
        raise ValueError("设备清单重复")
    ordered = sorted(ids, key=lambda key: digest([seed, key]))
    serving_id = ordered[-1]
    roles = {**_roles(ordered[:-1], seed=seed), serving_id: "serve"}
    rows, serving, answers, sources = [], [], [], []
    for item, key in zip(specification, ids):
        folder = (root/item["path"]).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            raise ValueError("设备路径不在输入目录内")
        files = list(folder.glob("*.csv"))
        by_index = {int(p.stem): p for p in files}
        if (len(files) != len(by_index) or type(item["failure_index"]) is not int
                or sorted(by_index) != list(range(1, item["failure_index"]+1))):
            raise ValueError("设备文件序列与明确失效点不一致")
        for index, path in sorted(by_index.items()):
            record = read_vibration(path, bearing=item["bearing"], condition=item["condition"], file_index=index)
            sample = bind_vibration(record, ordinal=2 if key==serving_id else 1, role=roles[key],
                                    failure_index=item["failure_index"])
            (serving if key==serving_id else rows).append(sample)
            sources.append({"path": path.relative_to(root).as_posix(), "sha256": record.sha256})
            if key==serving_id:
                answers.append({"sample_id": sample.sample_id, "label": float(item["failure_index"]-index)})
    task = bearing_task()
    return task, (BatchContext(task.task_id, "completed-bearing-trajectories", 1, tuple(rows)),
                  BatchContext(task.task_id, "unseen-bearing-trajectory", 2, tuple(serving))), {
        "protocol": "bearing-local-development-v1", "official": False, "seed": seed,
        "target": "公开完整寿命的剩余分钟；未指定为比赛任务", "specification": specification,
        "source_files": sources, "answers": answers}
