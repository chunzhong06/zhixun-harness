"""用实际交付权重和源组合表，对单条发动机 JSON 输入预测剩余循环。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from harness.adapters.datasets import official_layout as public
from harness.core.contracts import digest
from harness.learning.models.model_artifacts import ModelArtifacts
from harness.learning.models.official_rul_cpu import _trainer
from harness.learning.models.source_combinations import SourceCombinations


def input_rows(requests):
    for request in requests:
        if (not isinstance(request, dict) or set(request)-{"inputs", "task"} or request.get("task", "regression") != "regression"
                or not isinstance(request.get("inputs"), dict) or set(request["inputs"]) != {"sensors", "settings"}):
            raise ValueError("需要与赛题相同的发动机 inputs 和可选 task")
        fields = request["inputs"]
        sensors, settings = fields["sensors"], fields["settings"]
        if (not isinstance(sensors, list) or not isinstance(settings, list) or len(sensors) != 21 or len(settings) != 3
                or any(type(v) not in (float, int) for v in sensors+settings)):
            raise ValueError("需要二十一项传感器读数和三项数值设置")
        yield np.asarray([sensors+settings])


def infer(folder, ordinal, data_root, requests):
    report = json.loads((folder / "run.json").read_bytes())
    if report["policy"] not in ("source-combinations","early-reference","source-tool-choice","full-condition-native","continued-condition-native","training-inquiry","training-inquiry-fixed") or not 1 <= ordinal <= len(report["accepted"]):
        raise ValueError("需要已完成该轮的发动机自动输入修复运行")
    entry = report["accepted"][ordinal-1]
    accepted = json.loads((folder / "accepted" / (entry["request"] + ".json")).read_bytes())
    payload = accepted["payload"]
    if accepted["request_id"] != entry["request"] or digest([report["identity"], payload]) != entry["request"]:
        raise ValueError("选择的服务回执身份不符")
    lookup = SourceCombinations.restore(json.loads((folder / "source-combinations.json").read_bytes()))
    if report["policy"] in ("training-inquiry","training-inquiry-fixed"):
        from harness.learning.models.engine_training_inquiry import load_bundle,predict
        manifest,blob=ModelArtifacts(folder/"references").read(report["inquiry_artifact"])
        spec=manifest["request"]
        archive=data_root/"engine"/public.PACKAGES["engine"][0]
        if (spec["policy"]!=report["policy"] or spec["repair"]!=lookup.identity
                or spec["package"]!=public.PACKAGES["engine"][1] or sha256(archive.read_bytes()).hexdigest()!=spec["package"]
                or digest([spec,manifest["sha256"]])!=payload["bank"] or payload["bank"]!=report["reference_identity"]):
            raise ValueError("训练实验的模型、输入刻度或原包与交接不同")
        with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"C") as trainer:
            bundle,trace=load_bundle(blob,trainer)
            if (trace!=report["training_inquiry"] or digest(trace)!=payload["decision"]["trace_sha256"]
                    or trace["published"]!=payload["decision"]["chosen"]):
                raise ValueError("实际发布版本与训练实验的决定不同")
            for x in input_rows(requests):
                yield {"outputs":{"value":float(predict(bundle,lookup.transform(x))[0])}}
        return
    if report["policy"] in ("full-condition-native","continued-condition-native"):
        from harness.learning.models.full_condition_engine import FullConditionEngine
        manifest,blob=ModelArtifacts(folder/"references").read(report["native_artifact"])
        spec=manifest["request"]
        archive=data_root/"engine"/public.PACKAGES["engine"][0]
        if (spec["policy"]!=report["policy"] or spec["repair"]!=lookup.identity or spec["package"]!=public.PACKAGES["engine"][1]
                or sha256(archive.read_bytes()).hexdigest()!=spec["package"]
                or digest([spec,manifest["sha256"]])!=payload["bank"] or payload["bank"]!=report["reference_identity"]):
            raise ValueError("正式预测输入变换或原模型身份与回执不符")
        with TemporaryDirectory() as tmp,ZipFile(archive) as pack,_trainer(pack,Path(tmp),"C") as trainer:
            condition,center,model,checksum=FullConditionEngine.load_bundle(blob,trainer)
            if report["native_sha256"]!=checksum or payload["decision"]["native_sha256"]!=checksum:
                raise ValueError("实际续训权重与本轮交接不符")
            for x in input_rows(requests):
                values=FullConditionEngine.inputs(condition,center,model,lookup.transform(x))
                yield {"outputs":{"value":max(0.,float(model.predict(values)[0]))}}
        return
    if report["policy"] in ("early-reference","source-tool-choice"):
        from harness.learning.models.early_engine_reference import EarlyEngineReference
        from harness.learning.models.training_tools import SklearnBackend
        manifest,blob = ModelArtifacts(folder/"references").read(report["tree_artifact"])
        spec = manifest["request"]
        if (spec["policy"]!=report["policy"] or spec["repair"]!=lookup.identity
                or digest([spec,manifest["sha256"]])!=payload["bank"] or payload["bank"]!=report["reference_identity"]):
            raise ValueError("发动机参照模型与交付身份不同")
        pipeline = SklearnBackend(execution_enabled=True).load(blob)
        predictor=EarlyEngineReference.predict
        if report["policy"]=="source-tool-choice":
            from harness.learning.models.engine_source_choice import SourceSelectedEngine
            if payload["decision"]["chosen"]!=pipeline["decision"]["choice"] or report["source_choice"]!=pipeline["decision"]["choice"]:
                raise ValueError("本轮选择与源端实际训练的工具不一致")
            predictor=SourceSelectedEngine.predict
        for x in input_rows(requests):
            value=float(predictor(pipeline,lookup,x)[0])
            if payload["decision"].get("nonnegative_cycles") is True:
                value=max(0.,value)
            yield {"outputs":{"value":value}}
        return
    if lookup.width != 24 or payload["bank"] != lookup.identity or report["setting_reference"] != lookup.identity:
        raise ValueError("单条预测使用的组合表与交付时不同")
    ref = json.loads((folder / "issued-models.json").read_bytes())[payload["selected"]]
    header, blob = ModelArtifacts(folder / "regular/models").read(ref)
    if header["sha256"] != payload["weights_sha256"]:
        raise ValueError("预测权重不属于本轮交接")
    archive = data_root / "engine" / public.PACKAGES["engine"][0]
    if sha256(archive.read_bytes()).hexdigest() != public.PACKAGES["engine"][1]:
        raise ValueError("官方模型加载源码来源变更")
    with TemporaryDirectory() as tmp, ZipFile(archive) as z, _trainer(z, Path(tmp), "C") as trainer:
        weights = Path(tmp) / "issued.npz"
        weights.write_bytes(blob)
        model = trainer.Model.load(weights)
        for x in input_rows(requests):
            yield {"outputs": {"value": float(model.predict(lookup.transform(x))[0])}}


def main(argv):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=public.ROOT / "data/official")
    args = parser.parse_args(argv)
    for result in infer(args.run, args.round, args.data_root, (json.loads(line) for line in sys.stdin if line.strip())):
        print(json.dumps(result, allow_nan=False), flush=True)
