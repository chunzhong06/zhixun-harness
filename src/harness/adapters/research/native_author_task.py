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
from harness.learning.inference.complete_checkpoint import load_checkpoint
from harness.learning.training.native_continuation import ContinuationSession, run_program, tensor_identity


class NativeCandidate:
    def __init__(self, model, checkpoint, report):
        self.model, self.checkpoint, self.report = model, checkpoint, report

    def predict(self, frame):
        prediction = self.model.predict(np.asarray(frame, dtype=float))
        if self.model.meta['task'] == 'classification':
            return np.asarray(prediction).argmax(axis=1) + 1
        return prediction

    @property
    def classes_(self):
        if self.model.meta['task'] != 'classification':
            raise AttributeError('回归任务没有分类标签')
        return np.arange(1, len(self.model.meta['classes']) + 1)

    def predict_proba(self, frame):
        """分类分数来自实际重载的原生模型，列顺序与官方类别编号一致。"""
        if self.model.meta['task'] != 'classification':
            raise ValueError('回归任务不提供分类概率')
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
    # 新对照显式冻结本次训练预算；缺省60仅兼容既有回放，不当作比赛规定。
    max_epochs = contract.get('max_parent_epochs', 60)
    if type(max_epochs) is not int or not 1 <= max_epochs <= 10000:
        raise ValueError('共同任务训练预算必须在参考执行器支持的1至10000轮内')
    with np.load(root/'train.npz', allow_pickle=False) as data:
        session = ContinuationSession(model, data['x'], data['y'],groups=data['groups'],max_epochs=max_epochs)
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
