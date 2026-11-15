"""从三任务总入口的交接回执取分，核对常规寿命维护及独立单条气体重跑。"""
from hashlib import sha256

from compare_group_objectives import BASE,atomic_json,digest,metric,official,read
from compare_fixed_tree_references import paths


def main(folder=None):
    folder=folder or BASE/"observer-integrated"
    receipt=read(folder/"public-run-all.json")
    if set(receipt["tasks"])!={"gas","bearing","engine"}:
        raise ValueError("总入口报告缺少任务")
    scores={}
    for name,entry in receipt["tasks"].items():
        actual=read(folder/entry["report"])
        data={}
        validate=official(name)[0].validation
        truth={r.sample_id:r.label for r in validate}
        if name=="gas":
            if entry["policy"]=="pooled-midway-cold":
                ref=BASE/"whole-maintenance/gas-cold-start/seed-17"
                events=read(ref/"scores.json")["requests"]
                for event,r in zip(actual["accepted"],events,strict=True):
                    accepted=read(folder/name/"accepted"/(event["request"]+".json"));p=accepted["payload"]
                    q=read(ref/"accepted"/(r+".json"))["payload"]
                    expected=dict(zip(q["ids"],q["predictions"]["fixed-midway"],strict=True))
                    if (event["request"]!=digest([actual["identity"],p]) or accepted["request_id"]!=event["request"] or
                            p["predictions"]!=expected or data.keys()&expected.keys()):
                        raise ValueError("保留首轮的完整交付与开发冻结输出不同")
                    data.update(p["predictions"])
                score=metric(name,truth,data)
                if len(data)!=entry["predictions"] or entry["rounds"]!=10:raise ValueError("气体完整交付范围不符")
                scores[name]={"metric":"macro_f1","score":score,"rows":len(data),"policy":entry["policy"],
                    "run_sha256":sha256((folder/entry["report"]).read_bytes()).hexdigest()}
                continue
            standalone=BASE/"gas-complementary-feedback/observer-errors/standalone"
            other=read(standalone/"run.json")
            for event,r in zip(actual["accepted"],other["accepted"],strict=True):
                p=read(folder/name/"accepted"/(event["request"]+".json"))["payload"]
                q=read(standalone/"accepted"/(r["request"]+".json"))["payload"]
                if (event["request"]!=digest([actual["identity"],p]) or p["snapshot"]!=q["snapshot"] or
                        p["predictions"]!=q["predictions"] or data.keys()&p["predictions"].keys()):
                    raise ValueError("整套气体输出与逐条交付核验不符")
                data.update(p["predictions"])
        elif entry["policy"]=="native-fixed":
            # 同一入口允许研究候选；从其真实权重重算，不能套用旧默认的逐条答案。
            import numpy as np
            from tempfile import TemporaryDirectory
            from zipfile import ZipFile
            from pathlib import Path
            from compare_group_objectives import ROOT,_trainer
            config=read(folder/name/"checkpoint-config.json")
            weights=Path(config["weights"])
            checksum=sha256(weights.read_bytes()).hexdigest()
            if checksum!=config["sha256"] or checksum!=actual["weights_sha256"] or checksum!=entry["weights_sha256"]:
                raise ValueError("总入口冻结权重身份不一致")
            samples=read(folder/name/"regular/state.json")["samples"]
            archive,=(ROOT/"data/official"/name).glob("*.zip")
            with TemporaryDirectory(dir=folder/name) as tmp,ZipFile(archive) as z,_trainer(z,Path(tmp),"B") as trainer:
                model=trainer.Model.load(weights)
                for event in actual["accepted"]:
                    accepted=read(folder/name/"accepted"/(event["request"]+".json"))
                    payload=accepted["payload"]
                    p=payload["predictions"]
                    # 本地接收端保存 request_id/payload；accepted=True 是提交时的返回值。
                    if (digest([actual["identity"],payload])!=event["request"] or
                            accepted["request_id"]!=event["request"] or data.keys()&p.keys() or
                            payload["weights_sha256"]!=checksum):
                        raise ValueError("冻结单权重交接回执无效或重复")
                    keys=sorted(p)
                    if any(samples[k]["label"] is not None or samples[k]["role"]!="serve" for k in keys):
                        raise ValueError("冻结权重的服务输入携带答案或角色不符")
                    x=np.asarray([samples[k]["values"] for k in keys])
                    np.testing.assert_allclose(model.predict(x),[p[k] for k in keys],rtol=0,atol=1e-7)
                    data.update(p)
        elif entry["policy"] in ("source-combinations","phase-fixed","early-reference","source-tool-choice","full-condition-native","continued-condition-native","training-inquiry","training-inquiry-fixed"):
            reference_root=BASE/"fixed-tree-references"
            if name=="bearing" and entry["policy"]=="phase-fixed":
                reference=reference_root/"life-scale/bearing/bearing/issued.json"
                arm="phase-fixed"
            elif name=="engine" and entry["policy"]=="source-combinations":
                reference=reference_root/"setting-reference/automatic/engine/issued.json"
                arm="current-auto"
            elif name=="engine" and entry["policy"]=="early-reference":
                source=BASE/"whole-maintenance/engine-early-reference"
                request=read(source/"scores.json")["engine"]["request"]
                original=read(source/"accepted"/(request+".json"))
                if original["request_id"]!=request or digest(["engine",original["payload"]])!=request:
                    raise ValueError("早期工况参照原预测交接不符")
                expected=original["payload"]["predictions"]["early"]
            elif name=="engine" and entry["policy"] in ("training-inquiry","training-inquiry-fixed"):
                mode="adaptive" if entry["policy"]=="training-inquiry" else "fixed-order"
                source=BASE/"whole-maintenance/training-inquiry/engine"
                study=read(source/"scores.json")
                request=study["request"]
                frozen=read(source/"accepted"/(request+".json"))
                record=frozen["payload"]
                if (frozen["request_id"]!=request or digest(["public",record])!=request
                        or study["trace"][mode]!=actual["training_inquiry"]):
                    raise ValueError("独立运行和正式入口的先导、后续实验或确认决定不一致")
                expected=dict(zip(record["ids"],record["predictions"][mode],strict=True))
            elif name=="engine" and entry["policy"]=="source-tool-choice":
                source=BASE/"whole-maintenance/engine-source-tool-choice"
                request=read(source/"scores.json")["cases"]["engine"]["request"]
                original=read(source/"accepted"/(request+".json"))
                frozen=original["payload"]
                if (original["request_id"]!=request or digest(["engine",frozen])!=request
                        or actual["source_choice"]!=frozen["choice"]
                        or set(actual["source_scores"])!=set(frozen["source_scores"])
                        or any(abs(actual["source_scores"][k]-v)>1e-9 for k,v in frozen["source_scores"].items())):
                    raise ValueError("源端取舍与独立复核不一致")
                expected=frozen["predictions"]["selected"]
            elif name=="engine" and entry["policy"]=="full-condition-native":
                source=BASE/"whole-maintenance/engine-full-condition"
                request=read(source/"scores.json")["engine"]["request"]
                frozen=read(source/"accepted"/(request+".json"))
                if (frozen["request_id"]!=request or digest(["engine",frozen["payload"]])!=request
                        or actual["native_sha256"]!=frozen["payload"]["states"]["early-native"]["sha256"]):
                    raise ValueError("三任务入口独立训练产生的赛方权重与冻结对照不同")
                expected=frozen["payload"]["predictions"]["full-early-native"]
            elif name=="engine" and entry["policy"]=="continued-condition-native":
                source=BASE/"whole-maintenance/engine-weight-average"
                request=read(source/"scores.json")["engine"]["request"]
                frozen=read(source/"accepted"/(request+".json"))
                if (frozen["request_id"]!=request or digest(["engine",frozen["payload"]])!=request
                        or actual["native_sha256"]!=frozen["payload"]["weight_hashes"]["epoch-60"]):
                    raise ValueError("正式训练末版权重与开发记录不同")
                expected=frozen["payload"]["predictions"]["epoch-60"]
            else:
                raise ValueError("参照策略与数据任务不符")
            if entry["policy"] in ("source-combinations","phase-fixed"):
                if sha256(reference.read_bytes()).hexdigest()!=read(reference.with_name("scores.json"))["issued_sha256"]:
                    raise ValueError("自动纠错参照身份不符")
                expected=read(reference)["predictions"][arm]
            for event in actual["accepted"]:
                accepted=read(folder/name/"accepted"/(event["request"]+".json"))
                payload=accepted["payload"]
                pred=payload["predictions"]
                if entry["policy"] in ("training-inquiry","training-inquiry-fixed") and (
                        payload["decision"]["trace_sha256"]!=digest(actual["training_inquiry"])
                        or payload["decision"]["chosen"]!=actual["training_inquiry"]["published"]
                        or payload["decision"]["nonnegative_cycles"] is not True):
                    raise ValueError("交接版本与独立确认选定版本不一致")
                if entry["policy"] in ("full-condition-native","continued-condition-native") and (
                        payload["decision"]["native_sha256"]!=actual["native_sha256"]
                        or payload["decision"]["nonnegative_cycles"] is not True or any(v<0 for v in pred.values())):
                    raise ValueError("发放的续训模型或输出范围与冻结版本不同")
                if entry["policy"]=="source-tool-choice" and (
                        payload["decision"]["chosen"]!=actual["source_choice"] or
                        payload["decision"]["nonnegative_cycles"] is not True or any(v<0 for v in pred.values())):
                    raise ValueError("发放工具或数值范围偏离源端决定")
                if entry["policy"]=="early-reference" and payload["decision"].get("nonnegative_cycles") is True:
                    if any(v<0 for v in pred.values()):
                        raise ValueError("非负寿命交接出现负值")
                    for k in pred:
                        expected[k]=max(0.,expected[k])
                if (digest([actual["identity"],payload])!=event["request"] or accepted["request_id"]!=event["request"]
                        or digest(pred)!=event["outputs_sha256"] or data.keys()&pred.keys()):
                    raise ValueError("总入口发动机输入修复交接身份不符")
                data.update(pred)
            if data.keys()!=expected.keys() or max(abs(v-expected[k]) for k,v in data.items())>1e-8:
                raise ValueError("集成发动机输出与独立复算不符")
        else:
            old,_=paths(name,name)
            regular=read(old/"run.json")
            for r,b in zip(actual["rounds"],regular["rounds"],strict=True):
                p=read(folder/name/r["submitted"])["payload"]["predictions"]
                q=read(old/b["submitted"])["payload"]["predictions"]
                if p.keys()!=q.keys() or max(abs(v-q[k]) for k,v in p.items())>1e-8 or data.keys()&p.keys():
                    raise ValueError("集成后的寿命任务与原默认输出不符")
                data.update(p)
        if len(data)!=entry["predictions"] or entry["rounds"]!=10:
            raise ValueError("总入口发放数与实际预测不一致")
        score=metric(name,truth,data)
        scores[name]={"metric":"macro_f1" if name=="gas" else "rmse","score":score,"rows":len(data),
                      "policy":entry["policy"],"run_sha256":sha256((folder/entry["report"]).read_bytes()).hexdigest()}
    atomic_json(folder/"scores.json",{"receipt_sha256":sha256((folder/"public-run-all.json").read_bytes()).hexdigest(),"tasks":scores})
    print(scores)


if __name__=="__main__":
    from argparse import ArgumentParser
    from pathlib import Path
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path)
    main(parser.parse_args().folder)
