# 智训引擎 Harness

本仓库提供校赛提交镜像 `zhixun-harness:sdk-0.2.4-runtime-gzip` 对应的 0.2.4 源码。镜像归档 `harness-runtime-gzip.tar` 为 349,112,832 字节（332.94 MiB），SHA-256 为 `3b3b299e6006b5d86439969674a11a9c511ab7bf5f461432cada4d2fec168052`。

## 运行入口

在已加载的提交镜像上执行以下命令可以查看 SDK 声明。

```sh
docker run --rm --network none zhixun-harness:sdk-0.2.4-runtime-gzip describe
```

镜像使用 `contest_harness 1.0.2` 和 `contest.harness.workspace.v2`。平台调用 `python3 /app/harness.py train` 进入训练流程，调用 `python3 /app/harness.py predict` 进行预测。队伍选用 PyTRiO 的 `Qwen/Qwen3.8-27B`，由宿主网关处理调用与认证。

| 路径 | 对应内容 |
| --- | --- |
| [harness.py](harness.py)、[contest_submission.py](contest_submission.py) | 官方 SDK 入口与本项目声明的训练、预测函数 |
| [src/harness](src/harness)、[baselines](baselines) | 任务适配、父模型续训、候选比较、预测与兼容入口 |
| [author-runtime](author-runtime) | 镜像内随附的第三方运行源码及许可证 |
| [assets/tabular-resources.json](assets/tabular-resources.json) | 气体、轴承模型资源的容器路径及摘要 |
| [release/gzip-runtime](release/gzip-runtime) | 基于已验收前版镜像的精简配方、实际依赖清单和验证记录 |

模型二进制随赛事镜像交付。资源路径、大小和 SHA-256 见[镜像验证记录](release/gzip-runtime/verification.json)。精简配方使用已在本机加载的前版镜像作为构建输入，正式运行可直接加载队伍提交的 tar。

## 验证结果

同一检查点在精简前后复算全部官方公开验证数据，气体 1,244 条、轴承 1,679 条、发动机 23,049 条，每条预测完全一致。在精简版上另行完成的发动机验收使用 113,089 条官方公开训练数据和 4 次真实 Qwen 调用，公开验证 RMSE 从 52.8686 降至 46.5633。[SwanLab 验收记录](https://swanlab.cn/@chunzhong06/zhixun/runs/2y782a4g)

仓库保留的 `engineering/` 是早期平台试运行版本的工程目录。对应的原版源码和说明见[历史提交 6385f8f](https://github.com/chunzhong06/zhixun-harness/tree/6385f8f4c0c9d71f8c9e0e0605cd24c2bb0150ad)。
