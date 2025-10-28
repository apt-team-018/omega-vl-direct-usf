# syntax=docker/dockerfile:1.6
# CUDA 12.1 + cuDNN runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install PyTorch with CUDA 12.1 from official index (pins to 2.4 series)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
 && (python3 -m pip install --prefer-binary --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple torch==2.4.1 torchvision || \
     python3 -m pip install --prefer-binary --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple torch torchvision) \
 # Install deps (flash-attn may fail depending on CUDA/driver/toolchain; we tolerate failure)
 && (python3 -m pip install -r /app/requirements.txt || true) \
 # Ensure core deps are present even if flash-attn failed to build
 && python3 -m pip install fastapi uvicorn[standard] accelerate hf_transfer safetensors sentencepiece einops orjson Pillow aiohttp \
 && python3 -m pip install --upgrade "git+https://github.com/apt-team-018/transformers-usf-exp.git"

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
