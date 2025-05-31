FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ADD . /app
EXPOSE 8501

WORKDIR /app
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --compile-bytecode

CMD ["uv", "run", "streamlit", "run", "app.py"]
