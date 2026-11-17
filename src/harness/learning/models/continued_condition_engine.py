"""从十遍版自己的原生检查点继续五十遍，发布权重时保留原输入刻度与优化器状态。"""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from harness.core.contracts import digest
from harness.learning.models.full_condition_engine import FullConditionEngine
from harness.learning.models.model_artifacts import ModelArtifacts
from harness.learning.models.training_tools import SklearnBackend


class ContinuedConditionEngine(FullConditionEngine):
    def __init__(self,task,sampled,package,checksum,folder,trainer):
        super().__init__(task,sampled,package,checksum,folder,trainer)
        parent = self.native_sha256
        store = ModelArtifacts(folder)
        backend = SklearnBackend(execution_enabled=True)
        spec = {"policy":"continued-condition-native","repair":self.repair.identity,"package":checksum,
                "parent_bundle":self.model_ref,"parent":parent,"additional_epochs":50,
                "implementation":sha256(Path(__file__).read_bytes()).hexdigest(),"backend":backend.specification}

        def train():
            x = self.inputs(self.condition,self.center,self.model,[r.values for r in package.train])
            self.model.fit(x,np.asarray([r.label for r in package.train]),50)
            with TemporaryDirectory() as tmp:
                file = Path(tmp)/"continued.npz"
                self.model.save(file)
                return backend.dump((self.condition,self.center,file.read_bytes()))

        self.model_ref = store.create(spec,train)
        header,blob = store.read(self.model_ref)
        self.identity = digest([spec,header["sha256"]])
        self.condition,self.center,self.model,self.native_sha256 = self.load_bundle(blob,trainer)

    def advance_current(self,model,explore,confirm,serve):
        output,base,decision = super().advance_current(model,explore,confirm,serve)
        decision["chosen"] = "continued-condition-native"
        return output,base,decision
