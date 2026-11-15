"""用数值优化独立核验镜像步，并检查发布时序与状态恢复；不评价模型精度。"""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from harness.models.agile_selector import mirror_step
from harness.models.publication_control import MODES, SourceSupport, stream
from harness.runtime.storage import atomic_json


def run(output):
    rng = np.random.default_rng(17)
    comparisons = []
    for active in (np.array([True, True, True]), np.array([True, False, True])):
        rates = np.array([.1, .3, 1.])
        prior = rng.uniform(.1, 1, (2, 3));prior /= prior.sum()
        cost = rng.normal(size=(2, 3));floor = 1e-4
        actual = mirror_step(prior, cost, rates, active, floor)
        def objective(flat):
            w = flat.reshape(2, 3)
            return np.sum(cost*w+(w*np.log(w/prior)-w+prior)/rates)
        start = np.full((2, 3), floor)
        start[:, active] = (1.-(~active).sum()*2*floor)/(2*active.sum())
        bounds = [(floor, None) if ok else (floor, floor) for ok in np.tile(active, 2)]
        result = minimize(objective, start.ravel(), method='SLSQP', bounds=bounds,
                          constraints={'type': 'eq', 'fun': lambda w: w.sum()-1.},
                          options={'ftol': 1e-12, 'maxiter': 1000})
        assert result.success, result.message
        assert abs(objective(actual.ravel())-result.fun) < 1e-9
        np.testing.assert_allclose(actual.ravel(), result.x, rtol=0, atol=2e-6)
        comparisons.append(float(abs(objective(actual.ravel())-result.fun)))
    clock = np.arange(72)
    x = rng.normal(size=(72, 4));x[40:] *= 20
    values = np.column_stack((2+np.sin(clock), 3+np.cos(clock)))
    values[40:, 1] += 10
    # 首个无反馈时刻强制覆盖组合后的限幅分支；后段选择器可能已将候选权重降到近零。
    x[0] = 20.
    values[0, 1] += 10
    ref = np.where(clock % 6 == 0, 2+np.cos(clock/3), np.nan)
    support = SourceSupport(rng.normal(size=(100, 4)), np.ones(100), np.zeros(100))
    checks = {}
    for mode in MODES:
        issued, weights, limited, state = stream(clock, values, ref, x, mode=mode, support=support)
        assert np.isfinite(issued).all() and np.all((weights >= 0) & (weights <= 1))
        np.testing.assert_allclose(weights.sum(axis=1), 1., rtol=0, atol=1e-12)
        changed = ref.copy();changed[42] += 100
        altered = stream(clock, values, changed, x, mode=mode, support=support)[0]
        np.testing.assert_array_equal(issued[:43], altered[:43])
        future_x = x.copy();future_x[42:] *= 100
        altered_x = stream(clock, values, ref, future_x, mode=mode, support=support)[0]
        np.testing.assert_array_equal(issued[:42], altered_x[:42])
        a, wa, la, saved = stream(clock[:37], values[:37], ref[:37], x[:37], mode=mode, support=support)
        b, wb, lb, restored = stream(clock[37:], values[37:], ref[37:], x[37:], mode=mode,
                                      state=json.loads(json.dumps(saved)))
        np.testing.assert_array_equal(issued, np.r_[a, b])
        np.testing.assert_array_equal(weights, np.r_[wa, wb])
        assert restored == state
        checks[mode] = {'future_reference_prefix': True, 'future_input_prefix': True, 'exact_restart': True,
                        'guard_triggered': bool(limited.any())}
    assert checks['support_guard']['guard_triggered'] and checks['agile_guard']['guard_triggered']
    report = {'scope': '合成工程核验，不是模型能力实验', 'mirror_step_objective_differences': comparisons,
              'modes': checks, 'new_model_calls': 0, 'new_training': 0}
    atomic_json(output, report);print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser();p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
