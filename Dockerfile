FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

CMD ["neurocvr", "--help"]
