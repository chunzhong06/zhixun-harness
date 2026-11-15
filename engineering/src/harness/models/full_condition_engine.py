"""按源设备开工工况准备输入，在全部合法训练行上续训赛方权重，联合发布输入变换与权重。"""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.preprocessing import StandardScaler

from harness.contracts import digest
from harness.models.condition_processing import ConditionStandardScaler
from harness.models.model_artifacts import ModelArtifacts
from harness.models.source_combinations import SourceCombinations
from harness.models.training_tools import SklearnBackend


class FullConditionEngine:
    def __init__(self,task,sampled,package,checksum,folder,trainer):
        self.repair = SourceCombinations([r.values for r in sampled])
        backend = SklearnBackend(execution_enabled=True)
        spec = {"policy":"full-condition-native","package":checksum,"repair":self.repair.identity,
                "parent":sha256(package.checkpoint).hexdigest(),"anchor_rows":digest([r.sample_id for r in sampled]),
                "decimals":list(task.condition_decimals),"epochs":10,"backend":backend.specification,
                "implementation":sha256(Path(__file__).read_bytes()).hexdigest()}

        def train():
            devices = defaultdict(list)
            for r in package.train:
                if r.role!="fit" or r.label is None:
                    raise ValueError("发动机续训只接受有答案的源训练记录")
                devices[r.unit_id].append(r)
            if not devices or any(len(rows)<8 for rows in devices.values()):
                raise ValueError("缺少源设备开工参考行")
            first = [r.values for rows in devices.values() for r in sorted(rows,key=lambda r:r.time_end)[:8]]
            condition = ConditionStandardScaler((21,22,23),tuple(task.condition_decimals)).fit(first)
            center = StandardScaler().fit(condition.transform([r.values for r in sampled]))
            values = np.asarray([r.values for r in package.train])
            with TemporaryDirectory() as tmp:
                parent_path = Path(tmp)/"parent.npz"
                parent_path.write_bytes(package.checkpoint)
                model = trainer.Model.load(parent_path)
                model.fit(self.inputs(condition,center,model,values),np.asarray([r.label for r in package.train]),10)
                output = Path(tmp)/"candidate.npz"
                model.save(output)
                return backend.dump((condition,center,output.read_bytes()))

        store = ModelArtifacts(folder)
        self.model_ref = store.create(spec,train)
        header,blob = store.read(self.model_ref)
        self.identity = digest([spec,header["sha256"]])
        self.condition,self.center,self.model,self.native_sha256 = self.load_bundle(blob,trainer)

    @staticmethod
    def load_bundle(blob,trainer):
        condition,center,weights = SklearnBackend(execution_enabled=True).load(blob)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"candidate.npz"
            path.write_bytes(weights)
            model = trainer.Model.load(path)
        return condition,center,model,sha256(weights).hexdigest()

    @staticmethod
    def inputs(condition,center,model,x):
        # 还原到原检查点规定的输入单位，权重、优化器和原模型标准化状态都可按赛方格式保存。
        return center.transform(condition.transform(x))*model.scale+model.mean

    def advance_current(self,model,explore,confirm,serve):
        ids = [r["sample_id"] for r in serve]
        if not ids or len(set(ids))!=len(ids) or any(r["label"] is not None for r in serve):
            raise ValueError("发动机交付需要无答案且独立的服务测量")
        x = np.asarray([r["values"] for r in serve])
        values = self.inputs(self.condition,self.center,self.model,self.repair.transform(x))
        return (dict(zip(ids,self.model.predict(values).tolist(),strict=True)),
                dict(zip(ids,model.predict(x).tolist(),strict=True)),
                {"chosen":"full-condition-native","native_sha256":self.native_sha256})
