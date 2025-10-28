# Flash Attention 2 Build Guide for H100

## Changes Made to Enable Flash Attention 2

### 1. Updated Base Image (Line 3)

**Before:**
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
```

**After:**
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
```

**Why**: The `devel` image includes:
- CUDA compiler (`nvcc`)
- CUDA development headers
- Required for compiling Flash Attention from source

### 2. Added Build Tools (Line 11)

**Added**: `ninja-build`

**Why**: Ninja is a fast build system that significantly speeds up Flash Attention compilation.

### 3. Updated Flash Attention Installation (Line 31)

**Before:**
```dockerfile
&& (python3 -m pip install flash-attn==2.6.1 || echo "Warning...")
```

**After:**
```dockerfile
&& python3 -m pip install packaging ninja \
&& python3 -m pip install flash-attn==2.6.1 --no-build-isolation
```

**Why**:
- `packaging` and `ninja`: Required Python packages for build
- `--no-build-isolation`: Allows Flash Attention's setup.py to import torch during compilation

## Build on H100 Machine (Recommended)

### Option 1: Build Directly on H100

```bash
# 1. SSH to your H100 machine
ssh user@your-h100-machine

# 2. Install Docker if not already installed
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 3. Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 4. Clone/copy your project
git clone <your-repo-url>
cd omega-vl-usf

# 5. Login to Docker Hub
docker login -u arpitsh018

# 6. Build the image (native H100 compilation)
docker build -t arpitsh018/omega-vlm-inference-engine:v0.0.3 .

# 7. Tag and push
docker tag arpitsh018/omega-vlm-inference-engine:v0.0.3 arpitsh018/omega-vlm-inference-engine:latest
docker push arpitsh018/omega-vlm-inference-engine:v0.0.3
docker push arpitsh018/omega-vlm-inference-engine:latest
```

### Option 2: Use Docker Buildx from Local Machine to H100

If you can access H100 remotely:

```bash
# 1. Create a buildx builder that uses your H100 machine
docker buildx create --name h100-builder \
  --driver docker-container \
  --driver-opt network=host \
  ssh://user@your-h100-machine

# 2. Use the H100 builder
docker buildx use h100-builder

# 3. Build and push (compiles on H100)
docker buildx build --platform linux/amd64 \
  -t arpitsh018/omega-vlm-inference-engine:v0.0.3 \
  -t arpitsh018/omega-vlm-inference-engine:latest \
  --push .
```

## Build Time Expectations

### With Flash Attention 2 Compilation
- **On H100**: 15-25 minutes (includes FA2 compile)
- **Flash Attention build**: ~5-10 minutes of the total time

### Build Progress Indicators
You'll see:
```
Collecting flash-attn==2.6.1
  Downloading flash_attn-2.6.1.tar.gz (2.6 MB)
Installing build dependencies: started
Installing build dependencies: finished
Preparing metadata (pyproject.toml): started
Building wheels for collected packages: flash-attn
  Building wheel for flash-attn (setup.py): started
  [... compilation output ...]
  Building wheel for flash-attn (setup.py): finished with status 'done'
Successfully installed flash-attn-2.6.1
```

## Verify Flash Attention After Build

```bash
# Run container and check
docker run --rm --gpus all \
  arpitsh018/omega-vlm-inference-engine:v0.0.3 \
  python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
try:
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
except ImportError as e:
    print(f'❌ Flash Attention not available: {e}')
"
```

## Alternative: Pre-built Flash Attention Wheel

If building from source is problematic, use a pre-built wheel:

```dockerfile
# Replace line 31 with:
&& python3 -m pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.1/flash_attn-2.6.1+cu123torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

## Build Command Summary

### On H100 Machine (Best):
```bash
docker build -t arpitsh018/omega-vlm-inference-engine:v0.0.3 .
docker push arpitsh018/omega-vlm-inference-engine:v0.0.3
```

### From Local via Buildx to H100:
```bash
docker buildx build --builder h100-builder --platform linux/amd64 \
  -t arpitsh018/omega-vlm-inference-engine:v0.0.3 --push .
```

## Runtime Verification

Once deployed:
```bash
# Check /health endpoint
curl http://your-server:8000/health | jq '.attn'

# Should show:
"attn": "flash_attention_2"
```

The Dockerfile is now configured to successfully compile Flash Attention 2 on H100!