# 来源说明

0.2.4 使用赛方的 baseline 与 SDK 接口。TabPFN 及其他随附的第三方实现保留在 `author-runtime/`，各发行包的版权和许可证见相应源码及其 `.dist-info/` 目录。项目使用 SINQ 对教师模型进行量化。镜像版为 skrub 的 HTML 报告延后导入汇总模块，改动可在 `release/gzip-runtime/prune.py` 核对。

## 早期试运行版

容器baselines中的原始MLP、数据读取和预测协议源于赛事官方cpu-c-guide-20261110-v1.zip；完整组件加载器、训练和评价发布由本项目实现。

Qwen文本模板依据 https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja 的文本分支实现，与官方模板输出逐字核对。接口依据 https://docs.pytrio.com/docs/advanced/openai 。

复用NumPy、SciPy、scikit-learn、PyTorch、Pandas等软件，其许可随安装包保留。CoMind作者代码仅用于本地论文对照，未装入比赛镜像；仓库中的对照适配器需要另行取得作者源码和研究数据。
