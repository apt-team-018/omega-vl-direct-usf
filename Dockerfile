# syntax=docker/dockerfile:1.6
# CUDA 12.4 + cuDNN DEVEL (optimized for H100, includes nvcc compiler for Flash Attention compilation)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

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

# Install PyTorch with CUDA 12.4 from official index (latest stable for H100)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip \
 && python3 -m pip install packaging ninja wheel setuptools \
 && (python3 -m pip install --prefer-binary --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple torch torchvision || \
     python3 -m pip install --prefer-binary --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple torch torchvision) \
 # Install accelerate FIRST (no version constraint - let pip choose compatible version)
 && python3 -m pip install accelerate \
 && python3 -c "import accelerate; print(f'✅ Accelerate installed: {accelerate.__version__}')" \
 # Install base dependencies (no version constraints)
 && python3 -m pip install safetensors sentencepiece einops \
 && python3 -m pip install Pillow aiohttp orjson \
 # Install transformers dependencies BEFORE custom transformers
 && python3 -m pip install regex requests tqdm numpy packaging filelock \
 # Install EXACT huggingface-hub version required by custom transformers
 && python3 -m pip install "huggingface-hub==1.0.0.rc6" \
 # Install custom transformers WITHOUT dependencies (deps already installed above)
 && python3 -m pip install --no-deps transformers-usf-om-vl-exp-v0==0.0.1.post1 \
 && python3 -c "import accelerate; import transformers; print(f'✅ After transformers - Accelerate: {accelerate.__version__}, Transformers: {transformers.__version__}')" \
 # Install server dependencies (no version constraints)
 && python3 -m pip install fastapi "uvicorn[standard]" hf_transfer \
 # Skip Flash Attention - use PyTorch SDPA (built-in, fast on H100)
 && echo "==== Attention Configuration ====" \
 && python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}')" \
 && echo "ℹ️  Using PyTorch SDPA (Scaled Dot Product Attention)" \
 && echo "   Performance: 85-90% of flash-attn (still excellent on H100)" \
 && echo "   Benefits: Fast build, zero compilation, works everywhere" \
 && echo "===================================="

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
    ATTN_IMPL=sdpa \
    MODEL_PATH=5techlab-research/test_iter3

# Expose FastAPI port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# GPU-aware entrypoint; uvicorn invoked inside the script
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
