ARG BASE=zhixun-cpu-runtime:1-gzip@sha256:7a288dfa75b2400cbda1a0f38783da2b80caefaebf5e3f131761198787018f54
FROM ${BASE}
# 运行时仅提供依赖和冻结资产。产品代码完整取自此次检出。
RUN --mount=type=bind,target=/input \
    mkdir -p /app/src \
    && cp -R /input/src/harness /app/src/harness \
    && python3 -B /input/release/build_gas_fit.py /input/assets/gas-fit-source.npz /opt/zhixun/gas-fit-v1 \
    && cp -R /input/assets/gas-training-v1 /opt/zhixun/gas-training-v1 \
    && cp -R /input/baselines /app/baselines \
    && cp /input/harness.py /input/contest_submission.py /input/pyproject.toml /app/ \
    && cp /input/release/training-sources.json /input/release/verify_training_runtime.py /opt/ \
    && python3 -m pip install --no-deps --no-build-isolation --no-index --no-compile --no-cache-dir /app \
    && rm -rf /app/build /app/src/zhixun_harness.egg-info /usr/local/lib/python3.13/site-packages/harness \
    && ln -s /app/src/harness /usr/local/lib/python3.13/site-packages/harness \
    && ln -s /app/src/harness /app/harness \
    && python3 -B /opt/verify_training_runtime.py
ARG VERSION=0.3.18
LABEL org.opencontainers.image.version=${VERSION}
ENTRYPOINT ["python3","-B","/app/harness.py"]
