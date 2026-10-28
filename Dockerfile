# CPU-only image for the VisIA API and Streamlit demo.
# Model weights are downloaded from Hugging Face on first use (or mount a cache volume at
# /root/.cache/huggingface to avoid repeated downloads). No weights are baked into the image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN pip install --no-cache-dir uv==0.6.14

WORKDIR /app
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY app ./app

RUN uv sync --frozen --no-dev --extra api --extra app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501

CMD ["uvicorn", "visia_spatial_intelligence.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
