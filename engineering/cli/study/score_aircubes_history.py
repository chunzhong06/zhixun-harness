"""从夏季训练向前看的多时长电极响应，随后只用六小时末的真实参考维护。"""
from score_aircubes_precision import DEST as PRIOR, evaluate, extend, LocalStore, PRIMARY as PREVIOUS
from harness.models.aircubes_history import HistoryRelease, HistoryMaintenance, PRIMARY

DEST = PRIOR.parent / "aircubes-history"


def train(tools, clocks, raw, truth):
    tools["history"] = HistoryRelease().fit(clocks, raw, truth)
    return tools


if __name__ == "__main__":
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("连续历史响应已发放")
        evaluate(DEST, PRIOR, train, HistoryMaintenance, PRIMARY, {PREVIOUS, PRIMARY}, extend)
