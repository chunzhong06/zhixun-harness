# 智训引擎 0.3.2

智训引擎接手比赛宿主逐轮发放的工业模型。A 识别气体种类，B 预测轴承剩余寿命，C 预测发动机剩余寿命。三项使用同一个 [SDK 入口](src/harness/contest/submission.py)。

本版调整气体训练。前两轮以后，现有模型遇到尚未试过的一批训练数据时可以再训练。候选在当轮反馈上既提高 Macro-F1 又增加正确条数时，才交给宿主选择。

| 气体方案 | 全公开 Macro-F1 | 八阶段平均改善 | 最差阶段改善 | 十轮用时 |
|---|---:|---:|---:|---:|
| 0.3.1 | 0.71403 | 0.00% | 0.00% | 3 分 38 秒 |
| 仅恢复训练 | 0.69465 | -3.72% | -28.37% | 6 分 47 秒 |
| 0.3.2 | 0.94642 | +51.79% | +0.00% | 5 分 33 秒 |

0.3.2 在第 5 轮换模。三个方案使用相同 4 核 CPU、4 GiB、合法训练发放及公开反馈，使用已有研究程序回放。各阶段以当时的权重计分，最后用全部 1,244 行核对终选。B、C 延用 0.3.1 的方法，本次从 Git 源码重建后三项分别核对了 256 行官方初始预测。队伍已有的正式总分 62.89 属于 0.2.32。[核验与对照](release/verification.json)

八个公开阶段中，前三阶段持平，后五阶段提升。表中阶段改善指相对官方起始模型补回的 Macro-F1 缺口比例。另在两个已知外部批次上，0.3.2 的 F1 为 0.30307 和 0.38275，均优于官方起点；仅恢复训练的对照在这两批上分别达到 0.37197 和 0.43718。

## 从源码运行

需要 Docker BuildKit 和已取得的 0.3.1 比赛镜像归档。父镜像提供官方 SDK 1.0.4、CPU 依赖、离线教师权重和许可文件。它的归档 SHA-256 为 `d130cb985cb3e3bdbd27e123a470c40dc2cc0cf1ae2a3ed61ebed69ec132c41d`。在仓库根目录执行以下命令，第一行路径换为实际位置。

```sh
docker load -i /path/to/0.3.1/harness.tar
docker buildx build --network=none --pull=false --provenance=false --load -t zhixun-harness:source-0.3.2 .
docker run --rm --network none zhixun-harness:source-0.3.2 describe
```

构建时使用本次检出的完整业务源码和 [SHA-256 清单](release/training-sources.json)，逐一核对 399 个 Python 模块及安装副本。取得赛方任一任务公开 ZIP 后，可独立调用比赛入口和官方预测器，逐行比较并按官方函数计算指标。以下使用 C 包作例，适用于 POSIX shell，省略 `--samples 256` 即检验全部公开行。

```sh
docker run --rm --network none --entrypoint python3 \
  -v "$PWD/release:/checks:ro" \
  -v "/path/to/C.zip:/inputs/C.zip:ro" \
  zhixun-harness:source-0.3.2 -B /checks/verify_public_checkpoint.py /inputs/C.zip --samples 256
```

已保存逐行预测时，可在 Python 3.11 以上、装有 NumPy 的环境中运行 `python release/score_public.py /path/to/task.zip /path/to/stream-predictions.npz`。程序验证官方包摘要、样本 ID 与顺序、输出形状，直接调用包内评分函数。A 的六列按官方检查点的类别顺序排列；B、C 每行提供一个寿命值。[气体候选取舍条件](release/protocol.json)

训练和推理分别位于 [续训实现](src/harness/learning/training) 和 [检查点预测](src/harness/learning/inference)。比赛入口位于仓库根目录。第三方依赖及教师权重的使用条件见[来源与许可](NOTICE.md)。
