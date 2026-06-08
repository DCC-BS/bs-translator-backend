# Stage 1: Builder
FROM python:3.13-alpine AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install build toolchain (gcc, musl-dev, make, etc.)
# protobuf-dev includes standard proto files (google/protobuf/*.proto)
# rust/cargo are required for temporalio's Rust components
RUN apk add --no-cache build-base git protoc protobuf-dev rust cargo

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
# --locked: Sync with lockfile
# --no-dev: Exclude development dependencies
# --no-install-project: Install dependencies only (caching layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Copy application code
COPY . /app

# Sync project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Stage 2: Runtime
FROM python:3.13-alpine

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies for varlock (libstdc++) and sklearn/OpenMP (libgomp)
RUN apk add --no-cache libstdc++ libgomp

# Create non-root user (Alpine syntax)
RUN addgroup -S app && adduser -S app -G app

# Copy the environment, but not the source code
COPY --from=builder --chown=app:app /app /app
COPY --chown=app:app --chmod=755 entrypoint.sh /app/entrypoint.sh
COPY --from=ghcr.io/dmno-dev/varlock:latest --chown=app:app /usr/local/bin/varlock /usr/local/bin/varlock

# Enable virtual environment
ENV PATH="/app/.venv/bin:$PATH"

USER app

ENV APP_MODE=prod

ENTRYPOINT ["/app/entrypoint.sh"]
