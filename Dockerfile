ARG BASE=zhixun-harness:sdk-0.3.0-gzip@sha256:433dd83e13fd9f64e111a1119c3509faff79ee14f0e8566a5b15daf2282f63f3
FROM ${BASE}
WORKDIR /app
# 父层提供官方 SDK、CPU 依赖和量化权重。业务代码完整取自本次检出。
RUN --mount=type=bind,target=/input \
    rm -rf /app/src/harness /app/baselines /app/build /app/src/zhixun_harness.egg-info \
           /usr/local/lib/python3.13/site-packages/harness /usr/local/lib/python3.13/site-packages/zhixun_harness-*.dist-info \
    && cp -R /input/src/harness /app/src/ \
    && cp -R /input/baselines /app/ \
    && cp /input/pyproject.toml /input/harness.py /input/contest_submission.py /app/ \
    && cp /input/assets/tabular-resources.json /app/assets/ \
    && cp /input/release/training-sources.json /input/release/verify_training_runtime.py /opt/ \
    && python3 -m pip install --no-deps --no-build-isolation --no-index --no-compile --no-cache-dir --force-reinstall /app \
    && rm -rf /app/build /app/src/zhixun_harness.egg-info \
    && python3 -B /opt/verify_training_runtime.py
LABEL org.opencontainers.image.version="0.3.1"
