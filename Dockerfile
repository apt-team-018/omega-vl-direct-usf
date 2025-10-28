# syntax=docker/dockerfile:1.6
# CUDA 12.1 + cuDNN DEVEL (includes nvcc compiler for Flash Attention compilation)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (include ninja-build for faster Flash Attention compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git build-essential curl ca-certificates ninja-build \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch with CUDA 12.1 from official index (pins to 2.4 series)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
 && python3 -m pip install packaging ninja \
 && (python3 -m pip install --prefer-binary --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple torch==2.4.1 || \
     python3 -m pip install --prefer-binary --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple torch) \
 # CRITICAL: Install custom transformers FIRST to prevent accelerate from installing standard transformers
 && python3 -m pip install transformers-usf-om-vl-exp-v0==0.0.1.post1 \
 # Now install other packages (they won't reinstall transformers)
 && python3 -m pip install fastapi==0.115.0 uvicorn[standard]==0.30.6 \
 && python3 -m pip install accelerate==0.34.2 hf_transfer \
 && python3 -m pip install safetensors>=0.4.4 sentencepiece>=0.2.0 einops>=0.7.0 \
 && python3 -m pip install Pillow>=10.0.0 aiohttp>=3.9.0 orjson>=3.9.0 \
 # Install Flash Attention 2 with --no-build-isolation (lets it see torch during build)
 && python3 -m pip install flash-attn==2.6.1 --no-build-isolation

# Copy server code
COPY server.py /app/server.py
COPY scripts/entrypoint.sh /app/scripts/entrypoint.sh
RUN chmod +x /app/scripts/entrypoint.sh

# Runtime env defaults (can be overridden at docker run)
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/root/.cache/usinc \
    HF_HUB_CACHE=/root/.cache/usinc/models \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    UVICORN_WORKERS=1 \
    TRUST_REMOTE_CODE=1 \
    DTYPE=bf16 \
    ATTN_IMPL=flash_attention_2 \
    MODEL_PATH=5techlab-research/test_iter3

# Expose FastAPI port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# GPU-aware entrypoint; uvicorn invoked inside the script
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
