"""从七列原始读数核对差分与连续电极响应，通过已接收的实际预测验收。"""
from verify_aircubes_extended import verify, LocalStore
from score_aircubes_history import DEST as PRIOR, PRIMARY, PREVIOUS
from harness.models.aircubes_history import BENCHMARK, INCUMBENT

DEST = PRIOR.parent / "aircubes-history-entry"

if __name__ == "__main__":
    with LocalStore(DEST / "lock").exclusive():
        if (DEST / "scores.json").exists():
            raise ValueError("已完成四台原始入口的重放")
        verify(PRIOR, DEST, PRIMARY, (PREVIOUS, BENCHMARK, INCUMBENT, "论文组合与单步浓度变化占0.25",
            "论文组合与仅当前读数占0.5", "论文组合与半小时及两小时电极历史占0.5",
            "论文组合与加入六小时电极历史占0.5", "保持装机时的连续响应等权选择"))
