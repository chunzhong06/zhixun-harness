"""N-CMAPSS整趟飞行的JSONL推理；传感器顺序按原件X_s_var，工况按W_var。"""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.models.flight_health import FlightHealth
from harness.models.fleet_flight import fleet_prediction
from harness.models.flight_segment_training import segments
from harness.models.flight_segment_precision import flight_evidence
from harness.models.flight_condition_response import read_conditions, measured_conditions
import numpy as np
import joblib
from hashlib import file_digest


def infer(model, requests, method):
    for record in requests:
        inputs = record["inputs"]
        value = model.predict(inputs["sensors"], inputs["settings"], method)
        yield {"outputs": {"value": value}}


def flight_rows(record):
    inputs = record["inputs"]
    sensor, settings = np.asarray(inputs["sensors"]), np.asarray(inputs["settings"])
    if (sensor.ndim != 2 or sensor.shape[1] != 14 or len(sensor) == 0 or
            settings.shape != (len(sensor), 4) or not np.isfinite(sensor).all() or not np.isfinite(settings).all()):
        raise ValueError("航班测量不完整")
    return np.column_stack((sensor, settings))


def infer_fleet(bundle, requests):
    for record in requests:
        rows = flight_rows(record)
        yield {"outputs": {"value": fleet_prediction(bundle, rows.mean(axis=0), rows[::16])}}


def infer_segments(bundle, requests):
    for record in requests:
        # 与训练相同，每16个原始时刻取一次再按采集顺序等分；预测不需要此前航班。
        views = segments(bundle["prep"], flight_rows(record)[::16])
        prediction, evidence = flight_evidence(bundle["learner"], views[None, :, :])
        weight = np.exp(-2 * evidence[0, :, 0])
        yield {"outputs": {"value": float(prediction[0] @ weight / weight.sum())}}


def infer_conditions(bundle, requests):
    for record in requests:
        rows = flight_rows(record)[::16]
        if bundle.get("input_view") == "measured":
            profiles = measured_conditions(rows, bundle["prep"], bundle["points"])
        else:
            profiles, _ = read_conditions(rows, bundle["prep"], bundle["points"])
        estimates, spread = bundle["readout"].readings(profiles[None, :, :])
        weight = (1 + spread[0]) ** -2
        yield {"outputs": {"value": float(estimates[0] @ weight / weight.sum())}}


def infer_component(bundle, requests):
    from harness.models.flight_component import ComponentReadings
    from harness.models.flight_masked_readings import combine

    for record in requests:
        rows = flight_rows(record)[::16]
        full = bundle["extractor"].read(rows)[0]
        profile = ComponentReadings.split(full[None,:,:])[-1]
        value = combine(*bundle["reader"].readings(profile))[0]
        yield {"outputs": {"value": max(0.,float(value + bundle["offset"]))}}


def infer_component_source(bundle, requests):
    # 同一源训练权重在没有既往设备回标的安装环境中可直接使用。
    yield from infer_component({**bundle,"offset":0.},requests)


def infer_block_mean(bundle, requests):
    from harness.models.flight_block import block_mean, component_view
    from harness.models.flight_masked_readings import combine

    for record in requests:
        features = component_view(bundle["extractor"], block_mean(flight_rows(record)))
        value = combine(*bundle["reader"].readings(features[None,:,:]))[0]
        yield {"outputs": {"value": float(value)}}


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    published = {"fleet": infer_fleet, "segment-precision": infer_segments, "condition-response": infer_conditions,
                 "component-calibration": infer_component, "component-source": infer_component_source,
                 "component-block-mean": infer_block_mean}
    parser.add_argument("--method", choices=("coverage", "whole", "supported", *published), default="coverage")
    args = parser.parse_args()
    requests = (json.loads(line) for line in sys.stdin if line.strip())
    if args.method in published:
        with args.artifact.open("rb") as artifact:
            if file_digest(artifact,"sha256").hexdigest() != args.sha256:
                raise ValueError("源端选择的机队模型与发布身份不同")
            artifact.seek(0)
            responses = published[args.method](joblib.load(artifact), requests)
    else:
        responses = infer(FlightHealth(args.artifact, args.sha256), requests, args.method)
    for response in responses:
        print(json.dumps(response, allow_nan=False), flush=True)
