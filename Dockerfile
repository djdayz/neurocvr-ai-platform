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

EXPOSE 8000

CMD ["uvicorn", "neurocvr.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
