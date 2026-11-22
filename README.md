# 智训引擎 0.3.1

智训引擎接手每轮发放的工业模型和数据，在 CPU 上续训、比较候选并向比赛宿主交付选中的检查点。A 识别气体类别，B 预测轴承剩余寿命（分钟），C 预测发动机剩余寿命（循环）。宿主的 `train` 和 `predict` 进入同一个 [SDK 实现](src/harness/contest/submission.py)。

本版移除了 C 在所有检查点输出后统一施加的 235 循环上限。官方初始权重的全部 23,049 行公开预测与原版逐行一致，其中 480 行高于 235。检查点自己定义的寿命范围仍在模型内处理。三项本地十轮结果、计时范围及镜像标识见[验收记录](release/verification.json)。队伍已知的 62.89 分属于此前在平台评测的 0.2.32。

## 本地基线

| 任务 | 终选公开指标 | 十轮耗时 | 最差留出阶段相对改善 |
|---|---:|---:|---:|
| A | Macro-F1 0.71403 | 3 分 38 秒 | +0.00% |
| B | RMSE 280.050 分钟 | 33 分 51 秒 | -67.13% |
| C | RMSE 46.377 循环 | 9 分 14 秒 | +4.02% |

三十个回合合计 46 分 43 秒。含阶段复测和终选全量评分的整场墙钟为 61 分 51 秒。三项使用 4 核 CPU、4 GiB 内存及本地回放网关。

A 的 109 条乙醛全部漏判，其中 108 条被认作乙醇。B 的短上下文模型全公开 RMSE 为 396.19 分钟。保留当前轴承交付模型，关闭短上下文模型直接整体替换的方案。C 的终选整体 RMSE 从 46.541 降到 46.377，最差阶段相对改善从 4.82% 降到 4.02%。

## 从源码运行

需要 Docker BuildKit 和已取得的 0.3.0 比赛镜像归档。父镜像提供官方 SDK 1.0.4、CPU 依赖、离线教师权重和许可文件。它的归档 SHA-256 为 `16057e0f91140acf911b5eef4080dd8d2b4315f2c963c05dd48966126fc1c1fd`。在仓库根目录执行以下命令，第一行路径换为实际位置。

```sh
docker load -i /path/to/0.3.0/harness.tar
docker buildx build --network=none --pull=false --provenance=false --load -t zhixun-harness:source-0.3.1 .
docker run --rm --network none zhixun-harness:source-0.3.1 describe
```

构建时使用本次检出的完整业务源码和 [SHA-256 清单](release/training-sources.json)，逐一核对 399 个 Python 模块及安装副本。取得赛方任一任务公开 ZIP 后，可独立调用比赛入口和官方预测器，逐行比较并按官方函数计算指标。以下使用 C 包作例，适用于 POSIX shell，省略 `--samples 256` 即检验全部公开行。

```sh
docker run --rm --network none --entrypoint python3 \
  -v "$PWD/release:/checks:ro" \
  -v "/path/to/C.zip:/inputs/C.zip:ro" \
  zhixun-harness:source-0.3.1 -B /checks/verify_public_checkpoint.py /inputs/C.zip --samples 256
```

已保存逐行预测时，可在 Python 3.11 以上、装有 NumPy 的环境中运行 `python release/score_public.py /path/to/task.zip /path/to/stream-predictions.npz`。程序验证官方包摘要、样本 ID 与顺序、输出形状，直接调用包内评分函数。A 的六列按官方检查点的类别顺序排列；B、C 每行提供一个寿命值。[三项验收条件](release/protocol.json)

训练和推理分别位于 [续训实现](src/harness/learning/training) 和 [检查点预测](src/harness/learning/inference)。比赛入口位于仓库根目录。第三方依赖及教师权重的使用条件见[来源与许可](NOTICE.md)。
