# 来源与许可

模型训练与预测沿用赛事提供的 baseline 和 contest_harness 1.0.4 接口；项目实现父模型续训、候选比较及交付。官方数据与隐藏评测由赛方管理。

Dockerfile 从指定摘要的 CPU 运行镜像继承官方 SDK、第三方运行库及随附许可证。A、B 使用量化的表格教师，资源路径与 SHA-256 见 [资源清单](assets/tabular-resources.json)。模型二进制由已配置的比赛镜像提供。TabPFN、TabM、NumPy、PyTorch 等依赖及权重各自适用其随附条款。
