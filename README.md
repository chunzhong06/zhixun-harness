# 智训引擎 0.3.17

比赛宿主逐轮发放父模型和数据。Harness 维护气体分类、轴承与发动机寿命预测三项模型。当前检出的完整源码用于构建比赛镜像。源码清单位于 [release/training-sources.json](release/training-sources.json)，第三方资产见 [NOTICE](NOTICE.md)。

## 使用提交镜像

```sh
docker load -i harness.tar
docker run --rm --network none --read-only zhixun-harness:sdk-0.3.17-gzip describe
```

提交物为一份可独立运行的 Docker-save 归档，389.13 MiB。正式训练由宿主发放 SDK 数据、父模型和模型网关。A 的首个强候选可使用本轮分给它的时间，额外研究程序保留原有上限。A 的训练教师在已核对的数据上复用一份 10,808 字节的概率矩阵。B 按本轮教师、线程及有序输入的摘要交接训练缓存。对照协议及本地验收见 [协议](release/protocol.json) 和 [回执](release/verification.json)。

## 从源码构建

Dockerfile 的基础镜像是离线 CPU 运行时，提供固定 SDK、数值库、两份量化教师资源及轴承的拟合状态。团队构建机已经加载这个运行时。如果另一台机器要改代码并重建，可使用标准 Docker 命令传递一次构建依赖。在现有构建机执行 `docker save zhixun-cpu-runtime:1-gzip -o cpu-runtime.tar`，在新机器执行 `docker load -i cpu-runtime.tar`。运行已导出的提交镜像只需上节的 `harness.tar`。

```sh
docker buildx build --network=none --pull=false --platform linux/amd64 --provenance=false --output type=image,name=zhixun-harness:sdk-0.3.17-gzip,compression=gzip,compression-level=9 .
docker save -o harness.tar zhixun-harness:sdk-0.3.17-gzip
```

常规发布完整安装 `src/harness`。从原 0.3.14 迁移依赖的 [CPU 运行时配方](release/Dockerfile.cpu-runtime) 与业务发布分开保存。Python 源码与安装位置引用同一个目录。轴承候选携带压缩教师，独立预测时在进程私有目录展开一次，经内容校验后交给作者接口加载。公开行的测量可复用 [原包计分脚本](release/score_public.py)。
