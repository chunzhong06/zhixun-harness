"""用独立评分与时序/恢复变形核对控制器，不生成科研性能数字。"""
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'engineering/src'))
from harness.models.cumulative_classification_control import MODES, CumulativeClassificationControl, confusion, macro_f1


def main():
    rng = np.random.default_rng(913)
    for classes in [2, 3, 6]:
        for count in [1, 17, 103]:
            y, p = rng.integers(1, classes+1, (2, count))
            expected = f1_score(y, p, labels=list(range(1, classes+1)), average='macro', zero_division=0)
            assert abs(float(macro_f1(confusion(y, p, classes)))-expected) < 1e-12
    prefix_y = np.array([1, 1, 2, 1, 2, 2]); prefix_p = np.array([1, 2, 2, 1, 1, 2])
    batches_y = [np.array([1, 2, 2, 1]), np.array([2, 2, 1]), np.array([1, 1, 2, 2, 2])]
    batches_p = [rng.integers(1, 3, (len(y), 5)) for y in batches_y]
    for mode in MODES:
        model = CumulativeClassificationControl(mode, 5, 3, confusion(prefix_y, prefix_p), 3)
        try:
            model.observe(batches_y[0], batches_p[0])
            raise AssertionError('未交付就接收标签')
        except ValueError:
            pass
        full_y, full_p = prefix_y.copy(), prefix_p.copy()
        for y, predictions in zip(batches_y, batches_p):
            restored = CumulativeClassificationControl.restore(model.state())
            a, pa = model.choose(len(y)); b, pb = restored.choose(len(y))
            assert a == b and pa == pb
            for illegal in [lambda: model.choose(len(y)), model.state]:
                try:
                    illegal(); raise AssertionError('未反馈批次被覆盖或持久化')
                except ValueError:
                    pass
            model.observe(y, predictions); restored.observe(y, predictions)
            assert model.state() == restored.state()
            full_y = np.r_[full_y, y]; full_p = np.r_[full_p, predictions[:, a]]
            assert abs(float(macro_f1(model.actual))-f1_score(full_y, full_p, labels=[1, 2], average='macro', zero_division=0)) < 1e-12
        # 改变尚未到期的后续标签，不应改变之前的模型选择。
        paths = []
        for changed in [False, True]:
            check = CumulativeClassificationControl(mode, 5, 3, confusion(prefix_y, prefix_p), 3)
            path = []
            for index, (y, predictions) in enumerate(zip(batches_y, batches_p)):
                path.append(check.choose(len(y))[0])
                check.observe(3-y if changed and index == 2 else y, predictions)
            paths.append(path)
        assert paths[0] == paths[1]
    print('独立Macro-F1、真实累计轨迹、非法时序、未来标签变形和逐批恢复均通过；0科研性能比较')


if __name__ == '__main__':
    main()
