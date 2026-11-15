# 公开包只读挂载在 /data，默认用本地规则和官方工业权重维护三个任务。
FROM python:3.13-slim
ARG INSTALL_TRAINING=0
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY engineering/src ./engineering/src
COPY engineering/cli ./engineering/cli
COPY engineering/tests/fixtures ./engineering/tests/fixtures
COPY engineering/configs ./engineering/configs
COPY engineering/__init__.py ./engineering/__init__.py
# 默认路径需要官方 NumPy 模型、作业监督和 SwanLab 离线记录。
RUN python -m pip install --no-cache-dir numpy -c engineering/configs/environment/requirements-lock-training.txt \
      -r engineering/configs/environment/requirements-engineering.txt && \
    if [ "$INSTALL_TRAINING" = "1" ]; then python -m pip install --no-cache-dir \
      -r engineering/configs/environment/requirements-lock-training.txt; fi
ENTRYPOINT ["python", "-B", "engineering/cli/workflow/run_workflow.py"]
CMD ["public-run", "all", "--data-root", "/data", "--output", "/output"]
