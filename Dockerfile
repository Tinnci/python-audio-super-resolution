FROM ghcr.io/astral-sh/uv:0.8.15 AS uv

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=uv /uv /uvx /usr/local/bin/
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN uv pip install --system --no-cache .

WORKDIR /workdir

ENTRYPOINT ["audio-super-res"]
