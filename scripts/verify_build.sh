#!/bin/bash
# Flash Attention Build Verification Script
# Tests Docker image for proper flash-attn installation and H100 compatibility

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

IMAGE="${1:-omega-vlm:latest}"
RUN_GPU_TESTS="${2:-yes}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Flash Attention Build Verification${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Image: $IMAGE"
echo "GPU Tests: $RUN_GPU_TESTS"
echo ""

# Check if image exists
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo -e "${RED}❌ Image not found: $IMAGE${NC}"
    echo "   Build it first: docker build -t $IMAGE ."
    exit 1
fi

echo -e "${GREEN}✅ Docker image found${NC}"
echo ""

# Test 1: Python and PyTorch versions
echo -e "${BLUE}Test 1: Python & PyTorch versions${NC}"
docker run --rm "$IMAGE" python3 -c "
import sys
import torch
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'cuDNN: {torch.backends.cudnn.version()}')
" || { echo -e "${RED}❌ Failed${NC}"; exit 1; }
echo -e "${GREEN}✅ Passed${NC}"
echo ""

# Test 2: Flash Attention installation
echo -e "${BLUE}Test 2: Flash Attention installation${NC}"
FLASH_STATUS=$(docker run --rm "$IMAGE" python3 -c "
try:
    import flash_attn
    print(f'INSTALLED:{flash_attn.__version__}')
except ImportError:
    print('NOT_INSTALLED')
" 2>&1)

if echo "$FLASH_STATUS" | grep -q "INSTALLED:"; then
    VERSION=$(echo "$FLASH_STATUS" | grep "INSTALLED:" | cut -d':' -f2)
    echo -e "${GREEN}✅ Flash Attention installed: $VERSION${NC}"
    echo "   Mode: flash_attention_2 (maximum performance)"
    FLASH_INSTALLED=true
elif echo "$FLASH_STATUS" | grep -q "NOT_INSTALLED"; then
    echo -e "${YELLOW}⚠️  Flash Attention not installed${NC}"
    echo "   Mode: SDPA fallback (85-90% performance)"
    FLASH_INSTALLED=false
else
    echo -e "${RED}❌ Error checking Flash Attention${NC}"
    echo "$FLASH_STATUS"
    exit 1
fi
echo ""

# Test 3: Required packages
echo -e "${BLUE}Test 3: Required packages${NC}"
docker run --rm "$IMAGE" python3 -c "
import transformers
import accelerate
import fastapi
import torch
print('✅ transformers:', transformers.__version__)
print('✅ accelerate:', accelerate.__version__)
print('✅ fastapi:', fastapi.__version__)
print('✅ torch:', torch.__version__)
" || { echo -e "${RED}❌ Failed${NC}"; exit 1; }
echo -e "${GREEN}✅ All required packages present${NC}"
echo ""

# Test 4: GPU tests (if requested)
if [ "$RUN_GPU_TESTS" = "yes" ]; then
    echo -e "${BLUE}Test 4: GPU availability${NC}"
    
    # Check if nvidia-docker is available
    if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  GPU not available or nvidia-docker not installed${NC}"
        echo "   Skipping GPU tests"
        echo "   To enable GPU tests: install nvidia-container-toolkit"
    else
        GPU_INFO=$(docker run --rm --gpus all "$IMAGE" python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA_DEVICES:{torch.cuda.device_count()}')
    print(f'GPU_0:{torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'MEMORY:{props.total_memory / 1024**3:.1f}GB')
    print(f'COMPUTE:{props.major}.{props.minor}')
else:
    print('NO_GPU')
" 2>&1)
        
        if echo "$GPU_INFO" | grep -q "CUDA_DEVICES:"; then
            echo -e "${GREEN}✅ GPU detected${NC}"
            echo "$GPU_INFO" | grep "GPU_0:" | sed 's/GPU_0:/   GPU:/'
            echo "$GPU_INFO" | grep "MEMORY:" | sed 's/MEMORY:/   Memory:/'
            echo "$GPU_INFO" | grep "COMPUTE:" | sed 's/COMPUTE:/   Compute:/'
            
            # Check for H100
            if echo "$GPU_INFO" | grep -q "H100"; then
                echo -e "${GREEN}   ✅ H100 detected - optimal performance expected${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  No GPU detected in container${NC}"
        fi
    fi
    echo ""
fi

# Test 5: Server startup (quick test)
echo -e "${BLUE}Test 5: Server startup test${NC}"
echo "Starting container..."

# Start container in background
CONTAINER_ID=$(docker run -d --rm \
    $([ "$RUN_GPU_TESTS" = "yes" ] && echo "--gpus all" || echo "") \
    -p 18000:8000 \
    -e PROGRESS_LOGS=0 \
    -e REDACT_SOURCE=1 \
    "$IMAGE" 2>&1)

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start container${NC}"
    exit 1
fi

echo "Container ID: ${CONTAINER_ID:0:12}"
echo "Waiting for server to start (max 60s)..."

# Wait for health endpoint
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:18000/health > /dev/null 2>&1; then
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo -n "."
done
echo ""

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}❌ Server failed to start within ${MAX_WAIT}s${NC}"
    echo "Container logs:"
    docker logs "$CONTAINER_ID" 2>&1 | tail -20
    docker stop "$CONTAINER_ID" > /dev/null 2>&1
    exit 1
fi

# Check health endpoint
HEALTH=$(curl -s http://localhost:18000/health)
READY=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('ready', False))")
ATTN=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('attn', 'unknown'))")

if [ "$READY" = "True" ]; then
    echo -e "${GREEN}✅ Server started successfully${NC}"
    echo "   Attention mode: $ATTN"
    echo "   Health endpoint: http://localhost:18000/health"
else
    echo -e "${RED}❌ Server not ready${NC}"
    echo "$HEALTH" | python3 -m json.tool
fi

# Cleanup
docker stop "$CONTAINER_ID" > /dev/null 2>&1
echo -e "${GREEN}✅ Container stopped${NC}"
echo ""

# Final summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Verification Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$FLASH_INSTALLED" = true ]; then
    echo -e "${GREEN}✅ Flash Attention: INSTALLED${NC}"
    echo "   Expected performance: 100% (baseline)"
    echo "   H100 tokens/sec: 90-110 (with batching)"
else
    echo -e "${YELLOW}⚠️  Flash Attention: NOT INSTALLED${NC}"
    echo "   Using SDPA fallback"
    echo "   Expected performance: 85-90% of flash-attn"
    echo "   H100 tokens/sec: 70-90 (with batching)"
fi

echo ""
echo -e "${GREEN}✅ Build verification complete${NC}"
echo ""
echo "Next steps:"
echo "  1. Deploy: kubectl apply -f deploy/aks-omega-vlm.yaml"
echo "  2. Test: curl http://your-server:8000/health"
echo "  3. Monitor: kubectl logs -f deployment/omega-vlm"
echo ""