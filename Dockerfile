FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

COPY uv.lock /app/uv.lock
COPY pyproject.toml /app/pyproject.toml

RUN uv sync --frozen --no-install-project

COPY . /app

RUN chmod +x /app/run.sh

RUN uv sync --frozen

ENV ENVIRONMENT=production

ENTRYPOINT /app/run.sh --port ${PORT}
