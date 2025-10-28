# Docker Build and Push Instructions

## Build and Push VLM Server to Docker Hub

### Step 1: Login to Docker Hub

```bash
echo "dckr_pat_vOxYclwU10qRcbKFGADeid2WcPg" | docker login -u arpiths018 --password-stdin
```

### Step 2: Build the Docker Image

```bash
docker build -t arpiths018/omega-vlm-inference-engine:v0.0.1 .
```

### Step 3: Tag as Latest (Optional)

```bash
docker tag arpiths018/omega-vlm-inference-engine:v0.0.1 arpiths018/omega-vlm-inference-engine:latest
```

### Step 4: Push to Docker Hub

```bash
# Push version tag
docker push arpiths018/omega-vlm-inference-engine:v0.0.1

# Push latest tag
docker push arpiths018/omega-vlm-inference-engine:latest
```

## Complete One-Liner

```bash
echo "dckr_pat_vOxYclwU10qRcbKFGADeid2WcPg" | docker login -u arpiths018 --password-stdin && \
docker build -t arpiths018/omega-vlm-inference-engine:v0.0.1 . && \
docker tag arpiths018/omega-vlm-inference-engine:v0.0.1 arpiths018/omega-vlm-inference-engine:latest && \
docker push arpiths018/omega-vlm-inference-engine:v0.0.1 && \
docker push arpiths018/omega-vlm-inference-engine:latest
```

## Verify Image

```bash
# Check local images
docker images | grep omega-vlm

# Pull and test (on another machine)
docker pull arpiths018/omega-vlm-inference-engine:v0.0.1
docker run --rm arpiths018/omega-vlm-inference-engine:v0.0.1 --help
```

## Update Kubernetes Deployment

After pushing, update the image in [`deploy/aks-omega-vlm.yaml`](deploy/aks-omega-vlm.yaml):

```yaml
spec:
  containers:
  - name: server
    image: docker.io/arpiths018/omega-vlm-inference-engine:v0.0.1
    imagePullPolicy: Always
```

## Available Images

After pushing, these images will be available:

- `docker.io/arpiths018/omega-vlm-inference-engine:v0.0.1`
- `docker.io/arpiths018/omega-vlm-inference-engine:latest`

## Quick Run Test

```bash
# Test the pushed image
docker run -d --name vlm-test \
  --gpus "device=0" \
  -p 8000:8000 \
  -e API_KEY="test-token" \
  arpiths018/omega-vlm-inference-engine:v0.0.1 \
  --gpu-count 1 \
  --model-path 5techlab-research/test_iter3

# Check health
sleep 30
curl http://localhost:8000/health

# Cleanup
docker stop vlm-test && docker rm vlm-test
```

## Troubleshooting

### Build Failed
```bash
# Clean build with no cache
docker build --no-cache -t arpiths018/omega-vlm-inference-engine:v0.0.1 .
```

### Push Failed
```bash
# Re-login
docker logout
echo "dckr_pat_vOxYclwU10qRcbKFGADeid2WcPg" | docker login -u arpiths018 --password-stdin

# Retry push
docker push arpiths018/omega-vlm-inference-engine:v0.0.1
```

### Large Image Size
```bash
# Check image size
docker images arpiths018/omega-vlm-inference-engine:v0.0.1

# If too large, consider multi-stage build or smaller base image
```

## Security Note

⚠️ **IMPORTANT**: The Docker Hub token shown above should be kept secure. After deployment:

1. Revoke this token in Docker Hub settings
2. Generate a new token for future deployments
3. Store tokens securely (e.g., GitHub Secrets, Kubernetes Secrets)
4. Never commit tokens to version control