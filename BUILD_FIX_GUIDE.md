
# Docker Build Fix Guide

## Issue Summary

The Docker build was failing with the following errors:
```
[entrypoint] WARNING: transformers not importable: No module named 'transformers'
/app/scripts/entrypoint.sh: line 272: exec: uvicorn: not found
```

## Root Cause

The [`Dockerfile`](Dockerfile:1) was **missing the requirements.txt installation step**. While it copied the file, it never actually ran `pip install -r requirements.txt`, resulting in:
- ❌ Missing `uvicorn` - required to start the FastAPI server
- ❌ Missing `transformers` - required by the VLM model
- ❌ Missing `accelerate` - required for model optimization
- ❌ Missing other dependencies from [`requirements.txt`](requirements.txt:1)

## Solution Applied

### Changes Made to Dockerfile

Added two new layers after PyTorch installation (after line 23):

```dockerfile
# Install all application dependencies from requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r /app/requirements.txt

# Verify critical imports (fail build early if missing)
RUN python3 -c "import uvicorn; import fastapi; import transformers; import accelerate; print('[build] ✓ All critical packages installed successfully')"
```

### Benefits

✅ **All dependencies installed at build time** - Faster container startup  
✅ **Build-time verification** - Catches missing packages immediately  
✅ **Git-based transformers** - Installs from `git+https://github.com/apt-team-018/transformers-usf-exp.git`  
✅ **Docker layer caching** - Speeds up rebuilds  
✅ **Production-ready** - No runtime installation delays  

## Build Instructions

### 1. Build the Docker Image

```bash
# Basic build
docker build -t omega-vlm:latest .

# Build with build args (optional)
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t omega-vlm:latest \
  .

# Build with no cache (for debugging)
docker build --no-cache -t omega-vlm:latest .
```

### 2. Verify Build Success

Check the build output for:
```
[build] ✓ All critical packages installed successfully
```

If you see this message, all dependencies are properly installed.

### 3. Test the Container

```bash
# Run with minimal config (CPU mode for testing)
docker run --rm \
  -p 8000:8000 \
  -e REQUIRE_GPU=0 \
  -e DEBUG=1 \
  omega-vlm:latest

# Run with GPU (production)
docker run --rm \
  --gpus all \
  -p 8000:8000 \
  -e MODEL_PATH=5techlab-research/test_iter3 \
  omega-vlm:latest
```

### 4. Verify Server Startup

Check for these log messages:
```
[entrypoint] Using transformers X.X.X from /usr/local/lib/python3.XX/dist-packages/transformers/__init__.py
[entrypoint] Starting uvicorn on 0.0.0.0:8000...
[server] transformers X.X.X from /usr/local/lib/python3.XX/dist-packages/transformers/__init__.py
[startup] Loading processor…
[startup] Processor ready
[startup] Loading VLM model...
[server] Ready. Health: http://0.0.0.0:8000/health
```

### 5. Health Check

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response (should show "ready": true)
{
  "model_path": "5techlab-research/test_iter3",
  "devices": ["cuda:0"],
  "ready": true,
  "startup_ok": true,
  ...
}
```

## Troubleshooting

### Build Fails at Requirements Installation

**Error**: `Failed to install git+https://github.com/apt-team-018/transformers-usf-exp.git`

**Solutions**:
1. Check internet connectivity in build environment
2. Verify GitHub repository is accessible
3. Try building with `--network=host` if behind proxy:
   ```bash
   docker build --network=host -t omega-vlm:latest .
   ```

### Import Verification Fails

**Error**: `ModuleNotFoundError: No module named 'uvicorn'`

**Solutions**:
1. Ensure [`requirements.txt`](requirements.txt:1) contains all dependencies
2. Rebuild with `--no-cache`:
   ```bash
   docker build --no-cache -t omega-vlm:latest .
   ```

### Container Starts but Server Crashes

**Check**:
```bash
# View container logs
docker logs <container_id>

# Run in interactive mode to debug
docker run -it --rm omega-vlm:latest /bin/bash

# Inside container, test imports manually
python3 -c "import uvicorn; import transformers; print('OK')"
```

## Production Deployment

### Build for AKS/Kubernetes

```bash
# Build and tag for registry
docker build -t your-registry.azurecr.io/omega-vlm:v1.0.0 .

# Push to registry
docker push your-registry.azurecr.io/omega-vlm:v1.0.0

# Update deployment YAML with new image
kubectl apply -f deploy/aks-omega-vlm.yaml
```

### Environment Variables

Key variables for deployment:
```bash
MODEL_PATH=5techlab-research/test_iter3
DTYPE=bf16
ATTN_IMPL=flash_attention_2
MAX_BATCH_SIZE=2
REQUIRE_GPU=1
DEBUG=0
REDACT_SOURCE=1
```

## Verification Checklist

Before deploying to production:

- [ ] Build completes without errors
- [ ] Import verification passes: `✓ All critical packages installed successfully`
- [ ] Container starts successfully
- [ ] uvicorn starts on port 8000
- [ ] Health endpoint returns `"ready": true`
- [ ] Model loads without errors
- [ ] Chat completions API responds correctly

## Next Steps

