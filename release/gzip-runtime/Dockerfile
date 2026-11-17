FROM zhixun-harness:sdk-0.2.4-compatible@sha256:81aa8c265aab53c66c1e2e5c0268360d33691056af49b4e68c31def8049d1653 AS trim
COPY prune.py /tmp/prune-runtime.py
RUN python3 -B /tmp/prune-runtime.py && rm /tmp/prune-runtime.py
FROM scratch
COPY --from=trim / /
ENV PATH="/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV LANG="C.UTF-8"
ENV GPG_KEY="7169605F62C751356D054A26A821E680E5FA6305"
ENV PYTHON_VERSION="3.13.18"
ENV PYTHON_SHA256="abb0102e6d4ae027509bcb82327d9dd85515b44a3ba8ad538cfc5e9847029f68"
ENV PYTHONDONTWRITEBYTECODE="1"
ENV PYTHONUNBUFFERED="1"
ENV BASELINE_CPU_THREADS="1"
ENV OMP_NUM_THREADS="1"
ENV OPENBLAS_NUM_THREADS="1"
ENV MKL_NUM_THREADS="1"
ENV PYTHONPATH="/app/src:/app/author-runtime"
ENV HARNESS_TABULAR_RESOURCES="/app/assets/tabular-resources.json"
ENV CONTEST_SUBMISSION="contest_submission"
ENV HF_HUB_OFFLINE="1"
ENV TABPFN_NO_BROWSER="1"
ENV CONTEST_WORK="/workspace"
WORKDIR /app
ENTRYPOINT ["python3", "-B", "/app/harness.py"]
