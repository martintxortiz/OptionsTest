FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV BLIS_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md run_backtest.py run_search.py train_heavy_model.py ./
COPY src ./src

RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "train_heavy_model.py", "--cpu-workers", "1"]
