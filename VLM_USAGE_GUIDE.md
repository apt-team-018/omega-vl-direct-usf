# Omega VLM Inference Server - Usage Guide

## Quick Start

### Build and Run

```bash
# Build the Docker image
docker build -t omega-vlm-server .

# Run with 1 GPU
docker run -d --name omega-vlm \
  --gpus "device=0" \
  -p 8000:8000 \
  -e API_KEY="your-secure-token" \
  -v /path/to/model:/models/omega-vlm:ro \
  omega-vlm-server \
  --gpu-count 1 \
  --model-path /models/omega-vlm \
  --max-model-length 8192

# Run with 8 GPUs (multi-GPU)
docker run -d --name omega-vlm \
  --gpus "device=0,1,2,3,4,5,6,7" \
  -p 8000:8000 \
  -e API_KEY="your-secure-token" \
  -v /path/to/model:/models/omega-vlm:ro \
  omega-vlm-server \
  --gpu-count 8 \
  --model-path /models/omega-vlm \
  --max-model-length 8192 \
  --max-batch-size 2 \
  --dtype bf16
```

## Environment Variables

### VLM-Specific Configuration

```bash
MAX_MODEL_LENGTH=8192           # Maximum sequence length
MAX_IMAGE_SIZE=1024             # Max image dimension (width/height)
MAX_IMAGES_PER_REQUEST=10       # Maximum images per request
IMAGE_CACHE_SIZE=256            # Image cache size
ALLOW_REMOTE_IMAGES=1           # Allow loading images from URLs

# Batching (VLM-optimized)
MAX_BATCH_SIZE=2                # Lower for VLM (vs 8 for text-only)
BATCH_TIMEOUT_MS=10             # Allow time for image loading
MAX_QUEUE_SIZE=128              # Lower for VLM (vs 256 for text-only)

# Generation defaults
MAX_NEW_TOKENS_DEFAULT=512      # Higher for VLM responses
TEMPERATURE_DEFAULT=0.8
TOP_P_DEFAULT=0.9
TOP_K_DEFAULT=50
REPETITION_PENALTY_DEFAULT=1.0
DO_SAMPLE_DEFAULT=1
SEED_DEFAULT=42

# GPU Configuration
GPU_COUNT=8                     # Use first 8 GPUs
GPU_IDS="0,1,2,3,4,5,6,7"      # Or explicit GPU IDs
```

## API Usage Examples

### 1. Text + Image Request (URL)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secure-token" \
  -d '{
    "model": "omega",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "image": "https://example.com/image.jpg"
          },
          {
            "type": "text",
            "text": "Describe this image in detail."
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95
  }'
```

### 2. Text + Image Request (Base64)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secure-token" \
  -d '{
    "model": "omega",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
          },
          {
            "type": "text",
            "text": "What objects are in this image?"
          }
        ]
      }
    ],
    "max_tokens": 256
  }'
```

### 3. Multiple Images Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secure-token" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "image": "https://example.com/image1.jpg"
          },
          {
            "type": "image",
            "image": "https://example.com/image2.jpg"
          },
          {
            "type": "text",
            "text": "Compare these two images."
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.8
  }'
```

### 4. Text-Only Request (Backward Compatible)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secure-token" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "max_tokens": 128
  }'
```

### 5. Advanced Parameters Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secure-token" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "image": "https://example.com/photo.jpg"
          },
          {
            "type": "text",
            "text": "Analyze this image."
          }
        ]
      }
    ],
    "max_tokens": 1024,
    "min_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "length_penalty": 1.0,
    "seed": 42,
    "stop": ["END", "STOP"]
  }'
```

## Python Client Examples

### Basic VLM Request

```python
import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Prepare request
url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secure-token"
}

# Option 1: Use image URL
data = {
    "model": "omega",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://example.com/image.jpg"
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ]
        }
    ],
    "max_tokens": 512,
    "temperature": 0.7
}

# Option 2: Use base64-encoded local image
image_b64 = encode_image("local_image.jpg")
data = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{image_b64}"
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                }
            ]
        }
    ],
    "max_tokens": 256
}

# Send request
response = requests.post(url, headers=headers, json=data)
result = response.json()

print(result["choices"][0]["message"]["content"])
print(f"Tokens used: {result['usage']['total_tokens']}")
print(f"Latency: {result['latency_seconds']}s")
```

### OpenAI-Compatible Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-secure-token"
)

response = client.chat.completions.create(
    model="omega",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg"
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }
            ]
        }
    ],
    max_tokens=512,
    temperature=0.7
)

print(response.choices[0].message.content)
```

## Request Parameters

### Message Content Types

- **Text**: `{"type": "text", "text": "Your text here"}`
- **Image URL**: `{"type": "image", "image": "https://..."}`
- **Image Base64**: `{"type": "image", "image": "data:image/jpeg;base64,..."}`
- **OpenAI Format**: `{"type": "image_url", "image_url": {"url": "..."}}`

### Generation Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `max_tokens` | int | 512 | 1-∞ | Maximum tokens to generate |
| `min_tokens` | int | 1 | 0-∞ | Minimum tokens to generate |
| `temperature` | float | 0.8 | 0.0-2.0 | Sampling temperature |
| `top_p` | float | 0.9 | 0.0-1.0 | Nucleus sampling probability |
| `top_k` | int | 50 | 0-∞ | Top-k sampling |
| `repetition_penalty` | float | 1.0 | 1.0-2.0 | Repetition penalty |
| `length_penalty` | float | 1.0 | -2.0-2.0 | Length penalty for beam search |
| `seed` | int | 42 | 0-∞ | Random seed for reproducibility |
| `stop` | list[str] | null | - | Stop sequences |

## Performance Optimization

### GPU Configuration Matrix

#### Single GPU (Low Latency)
```bash
GPU_COUNT=1
MAX_BATCH_SIZE=1
BATCH_TIMEOUT_MS=0
MAX_NEW_TOKENS_DEFAULT=256
```

#### 4 GPUs (Balanced)
```bash
GPU_COUNT=4
MAX_BATCH_SIZE=2
BATCH_TIMEOUT_MS=10
MAX_NEW_TOKENS_DEFAULT=512
MAX_QUEUE_SIZE=128
```

#### 8 GPUs (High Throughput)
```bash
GPU_COUNT=8
MAX_BATCH_SIZE=2
BATCH_TIMEOUT_MS=15
MAX_NEW_TOKENS_DEFAULT=512
MAX_QUEUE_SIZE=256
```

### Memory Requirements

- **Single GPU**: 80GB A100/H100 recommended
- **Multi-GPU**: 80GB per GPU for optimal performance
- **VLM models require 3-4x more memory than text-only models**

### Image Optimization

```bash
# Limit image size to reduce memory
MAX_IMAGE_SIZE=1024

# Disable remote images for security
ALLOW_REMOTE_IMAGES=0

# Limit concurrent images
MAX_IMAGES_PER_REQUEST=5
```

## Health Check

```bash
curl -s http://localhost:8000/health | jq
```

Response:
```json
{
  "model_path": null,
  "devices": ["cuda:0", "cuda:1", ...],
  "device_names": ["NVIDIA A100-SXM4-80GB", ...],
  "visible_cuda_devices": 8,
  "queues": [0, 1, 0, 2, ...],
  "ready": true,
  "startup_ok": true,
  "startup_error": null,
  "overloaded": false,
  "max_queue_size": 128,
  "dtype": "torch.bfloat16",
  "attn": "flash_attention_2",
  "batch": {
    "max_batch_size": 2,
    "batch_timeout_ms": 10
  },
  "vlm_config": {
    "max_model_length": 8192,
    "max_image_size": 1024,
    "max_images_per_request": 10,
    "allow_remote_images": true
  }
}
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: omega-vlm
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: server
        image: omega-vlm-server:latest
        resources:
          limits:
            nvidia.com/gpu: 8
            memory: "256Gi"
        env:
        - name: GPU_COUNT
          value: "8"
        - name: MAX_MODEL_LENGTH
          value: "8192"
        - name: MAX_BATCH_SIZE
          value: "2"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: vlm-secret
              key: apiKey
        args:
        - "--gpu-count"
        - "8"
        - "--model-path"
        - "/models/omega-vlm"
        - "--max-model-length"
        - "8192"
        volumeMounts:
        - name: model-store
          mountPath: /models
      volumes:
      - name: model-store
        persistentVolumeClaim:
          claimName: model-pvc
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
MAX_BATCH_SIZE=1

# Reduce max tokens
MAX_NEW_TOKENS_DEFAULT=256

# Reduce image size
MAX_IMAGE_SIZE=512

# Reduce images per request
MAX_IMAGES_PER_REQUEST=5
```

### Slow Performance

```bash
# Increase batch size (if memory allows)
MAX_BATCH_SIZE=4

# Increase timeout for batching
BATCH_TIMEOUT_MS=20

# Use more GPUs
GPU_COUNT=8
```

### Image Loading Errors

```bash
# Check image URL accessibility
# Verify base64 encoding is correct
# Ensure MAX_IMAGE_SIZE is not too restrictive
# Check ALLOW_REMOTE_IMAGES setting
```

## Production Best Practices

1. **Use BF16 precision** for best quality/speed balance
2. **Set MAX_MODEL_LENGTH** at startup to match your use case
3. **Configure GPU_COUNT** based on your hardware (1-8 GPUs per node)
4. **Enable Flash Attention** when available for memory efficiency
5. **Use API_KEY** for authentication in production
6. **Monitor queue depths** via `/health` endpoint
7. **Set appropriate MAX_BATCH_SIZE** (2-4 for VLM)
8. **Limit MAX_IMAGES_PER_REQUEST** to prevent abuse
9. **Use persistent volumes** for model weights
10. **Configure resource limits** in Kubernetes

## Complete Example: Multi-Turn Conversation

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Authorization": "Bearer your-secure-token",
    "Content-Type": "application/json"
}

# Turn 1: Analyze image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://example.com/chart.png"},
            {"type": "text", "text": "What does this chart show?"}
        ]
    }
]

response = requests.post(url, headers=headers, json={
    "messages": messages,
    "max_tokens": 512
}).json()

assistant_reply = response["choices"][0]["message"]["content"]
print(f"Assistant: {assistant_reply}")

# Turn 2: Follow-up question
messages.append({"role": "assistant", "content": assistant_reply})
messages.append({
    "role": "user",
    "content": "What are the key insights from this data?"
})

response = requests.post(url, headers=headers, json={
    "messages": messages,
    "max_tokens": 512
}).json()

print(f"Assistant: {response['choices'][0]['message']['content']}")