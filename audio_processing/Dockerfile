# use https://docs.astral.sh/uv/guides/integration/docker/#available-images
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libsm6 libxext6 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . /app
WORKDIR /app

RUN uv sync --locked
ENV PATH="/app/.venv/bin:$PATH"
RUN uv add tensorflow-cpu-aws

# RUN uv pip install tensorflow-cpu-aw
# # Expose FastAPI port
# EXPOSE 8000

# Start FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7777", "--log-level", "debug", "--workers", "4"]
# CMD ["tail", "-f", "/dev/null"]