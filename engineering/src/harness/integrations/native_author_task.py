"""两种控制器共同使用的真实父模型训练接口；任务目录没有评分标签。"""
from hashlib import sha256
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import io
import json
import sys
import tempfile
import cloudpickle
import numpy as np
from harness.models.complete_checkpoint import load_checkpoint
from harness.models.native_continuation import ContinuationSession, run_program, tensor_identity


class NativeCandidate:
    def __init__(self, model, checkpoint, report):
        self.model, self.checkpoint, self.report = model, checkpoint, report

    def predict(self, frame):
        return self.model.predict(np.asarray(frame, dtype=float))


def train_candidate(program):
    root = Path('input')
    contract = json.loads((root/'native-contract.json').read_bytes())
    for name, checksum in contract['files'].items():
        if sha256((root/name).read_bytes()).hexdigest() != checksum:
            raise ValueError('共同任务输入身份变化')
    spec = spec_from_file_location('_native_official_base', root/'baseline-model.py')
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cloudpickle.register_pickle_by_value(module)
    parent = root/'initial.npz'
    model = load_checkpoint(parent, module.Model)
    with np.load(root/'train.npz', allow_pickle=False) as data:
        session = ContinuationSession(model, data['x'], data['y'],groups=data['groups'],max_epochs=60)
    report = run_program(program, session)
    model = session._model
    folder = Path('artifacts');folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        path = Path(temp)/'checkpoint.npz';model.save(path)
        checkpoint = path.read_bytes()
        restored = load_checkpoint(path,module.Model)
        if tensor_identity(restored) != report['final_tensors']:
            raise ValueError('交付状态未完整保存')
    report.update(parent_checkpoint_sha256=sha256(parent.read_bytes()).hexdigest(),
                  checkpoint_sha256=sha256(checkpoint).hexdigest(),training_source=program)
    return NativeCandidate(restored,checkpoint,report)
