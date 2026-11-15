# 智训引擎 Harness

比赛试运行源码。使用 PyTRiO Qwen/Qwen3.8-27B 驱动真实工业父模型续训，保留权重、Adam与标准化状态；经验证采用候选或保留父版，保存训练和决策记录。已完成本地及Linux容器验证，网站隐藏评测尚未验收。

## 构建

```sh
python engineering/cli/release/prepare_contest_wheels.py
docker build --platform linux/amd64 -f engineering/configs/container/Dockerfile.contest -t zhixun-harness:trial .
docker save -o harness.tar zhixun-harness:trial
```

依赖从PyPI和PyTorch官方CPU源下载，按固定SHA256核对。源码和镜像不含数据集、权重、凭据。

## 单轮训练

```sh
docker run --rm -e PYTRIO_API_KEY -v /your/input:/input:ro -v /your/output:/output zhixun-harness:trial --profile C --train /input/train.jsonl --validation /input/validation.jsonl --from-baseline /input/checkpoint.npz --output /output/round-1 --stage-id round-1
```

后续轮次使用 --resume 指定上一轮实际采用的检查点。命令参数沿官方原包；实际平台挂载和调用方式由试运行反馈核实。每轮最多一次模型请求、预留0.25元；按三任务各10轮，一次试运行最多预留7.5元调用费用，实际扣费以服务账单为准。

## 独立预测

```sh
docker run --rm -i -v /your/output/round-1:/model:ro zhixun-harness:trial --predict /model
```

标准输入为官方JSONL任务及inputs，标准输出为逐条outputs。发布目录自带baselines，仅需NumPy，也可执行 python -m baselines.predict --checkpoint checkpoint.npz。

实验记录：https://swanlab.cn/@chunzhong06/zhixun
