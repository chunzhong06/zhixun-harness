# 智训引擎 0.3.3

比赛宿主逐轮发放数据和父模型。气体任务识别六种气体，轴承与发动机任务分别预测剩余分钟数和循环数。三项通过同一个 [SDK 入口](src/harness/contest/submission.py) 训练、选择并交接模型。

本版修复了准备子进程被强制终止时的交接。直接收到 Linux 退出状态 −9 或 shell 状态 137 时，系统提交已确认的父权重及其状态。已用实际 SDK 核对下一轮恢复和普通错误的处理。三项预测策略沿用 0.3.2。

| 任务 | 十轮终选公开指标 | 各阶段平均改善 | 最差阶段改善 | 十轮回合耗时 |
|---|---:|---:|---:|---:|
| 气体 | Macro-F1 0.94642 | +51.79% | +0.00% | 5 分 34 秒 |
| 轴承 | RMSE 280.04957 分钟 | +14.60% | -67.13% | 24 分 36 秒 |
| 发动机 | RMSE 46.37674 循环 | +10.06% | +4.02% | 7 分 04 秒 |

4 核 CPU、4 GiB，研究程序本地回放。三项串行含阶段及终选全量计分共 48 分 00 秒。阶段改善相对官方起始模型计算。已知正式总分 62.89 对应 0.2.32。[逐项结果和故障验证](release/verification.json) · [实验条件](release/protocol.json)

气体早期将两份概率平均，首个留出阶段 F1 持平，另一阶段从 0.76984 升到 0.78571。两次都只识别出 109 条乙醛中的 3 条，继续沿用能在第五轮恢复高分模型的方案。

轴承首轮在三台设备上的 RMSE 分别从 217、402、715 变为 362、205、281 分钟。按短寿命反馈否决整个模型会丢掉后两台的改善。发动机同机配对使整体 RMSE 从 46.38 降到 46.19，但 FD002 中期从 41.27 升到 41.84 个循环，本版保留原预测器。

## 从源码运行

需要 Docker BuildKit 和已取得的 0.3.2 比赛镜像归档。父镜像提供官方 SDK 1.0.4、CPU 依赖、离线教师权重和许可文件。它的归档 SHA-256 为 `c24215d6a4c6ae98237033b8043a98b6e68185eafac7e3c70927bcf88ab5729e`。在仓库根目录执行以下命令，第一行路径换为实际位置。

```sh
docker load -i /path/to/0.3.2/harness.tar
docker buildx build --network=none --pull=false --provenance=false --load -t zhixun-harness:source-0.3.3 .
docker run --rm --network none zhixun-harness:source-0.3.3 describe
```

构建时使用本次检出的完整业务源码和 [SHA-256 清单](release/training-sources.json)，逐一核对 399 个 Python 模块及安装副本。取得赛方任一任务公开 ZIP 后，可独立调用比赛入口和官方预测器，逐行比较并按官方函数计算指标。以下使用 C 包作例，适用于 POSIX shell，省略 `--samples 256` 即检验全部公开行。

```sh
docker run --rm --network none --entrypoint python3 \
  -v "$PWD/release:/checks:ro" \
  -v "/path/to/C.zip:/inputs/C.zip:ro" \
  zhixun-harness:source-0.3.3 -B /checks/verify_public_checkpoint.py /inputs/C.zip --samples 256
```

已保存逐行预测时，可在 Python 3.11 以上、装有 NumPy 的环境中运行 `python release/score_public.py /path/to/task.zip /path/to/stream-predictions.npz`。程序验证官方包摘要、样本 ID 与顺序、输出形状，直接调用包内评分函数。A 的六列按官方检查点的类别顺序排列；B、C 每行提供一个寿命值。[本轮四项取舍条件](release/protocol.json)

训练和推理分别位于 [续训实现](src/harness/learning/training) 和 [检查点预测](src/harness/learning/inference)。比赛入口位于仓库根目录。第三方依赖及教师权重的使用条件见[来源与许可](NOTICE.md)。
