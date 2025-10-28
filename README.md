# GPU-Optimized LLM FastAPI Server (Dockerized)

Production-grade FastAPI server for Transformers and local model folders with:
- Local model folder or registry-style model ID via MODEL_PATH
- Multi-GPU routing (one replica per selected GPU, router dispatches per request)
- Micro-batching for throughput
- BF16 default compute, togglable to FP16/FP32
- FlashAttention 2 default-on (falls back to SDPA automatically; togglable off)
- Healthcheck, overload backpressure, and safe defaults
- GPU-aware Docker entrypoint accepting GPU count or explicit device IDs
- OpenAI-compatible API with configurable defaults, authentication, and model-name contract
- Protected Swagger UI and OpenAPI with Bearer auth support

Endpoints:
- OpenAI-compatible: POST /v1/chat/completions (protected when API_KEY is set)
- Simple generation: POST /generate
- Apply chat template: POST /v1/apply_template
- Models list: GET /v1/models (protected when API_KEY is set)
- Health: GET /health

Image Tags
- docker.io/arpitsh018/omega-vlm-inference-engine:latest
- docker.io/arpitsh018/omega-vlm-inference-engine:v0.2.13

Notes
- The app listens on 8000 inside the container.
- Map a host port of your choice, e.g. -p 8000:8000 or -p 80:8000.
## 🚀 Flash Attention 2 for H100 (NEW)

This image now includes **optimized Flash Attention 2 installation** using pre-built wheels:

- ✅ **30-second build** (no compilation required)
- ✅ **H100 native support** (compute_90 kernels)
- ✅ **Automatic SDPA fallback** if wheel unavailable
- ✅ **Build anywhere, deploy to H100**

### Quick Start
```bash
# Build with automatic flash-attn detection
docker build -t omega-vlm:latest .

# Flash Attention status shown during build:
# ✅ Flash Attention: 2.6.1 (if wheel available)
# ℹ️  SDPA fallback (if wheel unavailable)

# Verify after build
docker run --rm omega-vlm:latest python3 -c "import flash_attn; print(flash_attn.__version__)"
```

### Performance
- **Flash-Attn mode**: 90-110 tokens/sec on H100
- **SDPA fallback**: 70-90 tokens/sec on H100 (85-90% performance)

📖 **Complete Guide**: See [FLASH_ATTENTION_BUILD.md](FLASH_ATTENTION_BUILD.md) for:
- Pre-built wheel installation details
- Building on H100 machine
- Troubleshooting compilation issues
- Performance benchmarks
- Alternative strategies


## Requirements

- NVIDIA drivers installed on host
- nvidia-container-toolkit (for Docker GPU support)
- Docker 24+ and Docker Compose v2.20+ (recommended)
- Internet access (only required if using a remote model ID)

## Transformers implementation

This image installs a forked Transformers package during build:
- Source: git+https://github.com/apt-team-018/transformers-omega3.git
- Reason: Ensures compatibility with this server’s generation path.
- To revert to upstream: edit the Dockerfile and replace the git install with `pip install transformers` (then rebuild).

### Override Transformers at runtime (optional)

You can override the Transformers source at container start and the entrypoint will print exactly what is installed and where it’s loaded from.

New CLI flag/env:
- Flag: `--transformers-path PATH_OR_URL_OR_WHEEL`
- Env: `TRANSFORMERS_INSTALL_PATH=/path/or/url/or/wheel`

Behavior:
- If provided, the entrypoint runs: `pip install -U "$TRANSFORMERS_INSTALL_PATH"` before starting the server.
- It then prints:
  - `[entrypoint] transformers X.Y.Z from /path/to/site-packages/transformers/__init__.py`
  - `[entrypoint] Using transformers X.Y.Z from ...`
- The server also prints on startup:
  - `[server] transformers X.Y.Z from ...`

Examples:
```bash
# Install from a local path mounted into the container
docker run --rm --gpus "count=1" -p 8000:8000 \
  -v /opt/vendors/transformers:/vendors/transformers:ro \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --transformers-path /vendors/transformers

# Install from a Git URL (branch)
docker run --rm --gpus "count=1" -p 8000:8000 \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --transformers-path "git+https://github.com/your-org/transformers.git@your-branch"

# Install from a prebuilt wheel
docker run --rm --gpus "count=1" -p 8000:8000 \
  -v /opt/wheels:/wheels:ro \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --transformers-path /wheels/transformers-4.x.y-py3-none-any.whl
```

Kubernetes (optional):
- Add to container args:
```yaml
args:
  - "--transformers-path"
  - "git+https://github.com/your-org/transformers.git@your-branch"
```
- Or mount a volume with your local checkout/wheel and point `--transformers-path` to that path inside the container.

## Quickstart (GPU VM)

```bash
docker pull docker.io/arpitsh018/omega-vlm-inference-engine:latest

docker run -d --name omega-api --restart unless-stopped \
  --gpus all \
  -p 8000:8000 \
  -e API_KEY="YOUR_SECURE_TOKEN" \  # optional; protects docs and /v1/* endpoints
  -v /path/omega-vlm-prod-v0.1:/models/omega-vlm-prod-v0.1:ro \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --gpu-count 1 \
  --model-path /models/omega-vlm-prod-v0.1 \
  --model-name "model" \
  --dtype bf16 \
  --attn-impl flash_attention_2 \
  --max-batch-size 8 \
  --batch-timeout-ms 6 \
  --default-temperature 0.8 \
  --default-top-p 0.9 \
  --default-max-tokens 128 \
  --default-seed 42 \
  --default-sampling true \
  --host 0.0.0.0 \
  --port 8000
```

Health check
```bash
curl http://YOUR_VM_IP:8000/health
```

## Low-latency defaults (v0.2.13)

This image ships with low-latency settings enabled by default to reduce overhead on the /v1/chat/completions and /generate paths.

Defaults:
- MAX_BATCH_SIZE=1 (micro-batching off)
- BATCH_TIMEOUT_MS=0 (no collect window)
- PROGRESS_LOGS=0 (progress and per-request metrics printing off)
- FAST_USAGE=1 (use worker-provided token counts, avoiding re-tokenization in the endpoint)
- JSON responses use ORJSON by default (faster serialization)

Throughput vs latency:
- For minimal latency (default): keep MAX_BATCH_SIZE=1 and BATCH_TIMEOUT_MS=0.
- For higher throughput: increase micro-batching and a small window, e.g.:
  - MAX_BATCH_SIZE=8
  - BATCH_TIMEOUT_MS=4

Usage accounting:
- FAST_USAGE=1 (default): the worker returns prompt/completion token counts; the endpoint avoids extra tokenization work.
- Set FAST_USAGE=0 if you need endpoint-side tokenization (slower).

Examples
```bash
# Low-latency defaults (already the default):
docker run --rm --gpus "count=1" -p 8000:8000 \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --model-name "model"

# Higher throughput (micro-batching on):
docker run --rm --gpus "count=1" -p 8000:8000 \
  -e MAX_BATCH_SIZE=8 -e BATCH_TIMEOUT_MS=4 \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --model-name "model"

# Re-enable progress logs if desired:
docker run --rm --gpus "count=1" -p 8000:8000 \
  -e PROGRESS_LOGS=1 \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --model-name "model"

# Turn off fast usage (forces endpoint tokenization; slower):
docker run --rm --gpus "count=1" -p 8000:8000 \
  -e FAST_USAGE=0 \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --model-name "model"
```

## CPU-only VM

Works on non-GPU VMs; uses standard attention and FP32.

```bash
docker pull docker.io/arpitsh018/omega-vlm-inference-engine:latest

docker run -d --name omega-api --restart unless-stopped \
  -p 8000:8000 \
  -e API_KEY="YOUR_SECURE_TOKEN" \  # optional
  -v /path/omega-vlm-prod-v0.1:/models/omega-vlm-prod-v0.1:ro \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path /models/omega-vlm-prod-v0.1 \
  --disable-fa2 \
  --dtype fp32 \
  --host 0.0.0.0 \
  --port 8000
```

## Build (local)

```bash
docker build -t llm-fastapi:latest .
```

## Key Configuration (env or CLI flags)

Model
- MODEL_PATH: local directory or HF repo id
  - Local path example: /models/omega-vlm-prod-v0.1
  - Model ID example: model/omega-vlm-prod-v0.1
- TRUST_REMOTE_CODE: 1 (default) to allow custom model code; set 0 to disable

Compute / Kernels
- DTYPE: bf16 (default) | fp16 | fp32
- ATTN_IMPL: flash_attention_2 (default) | sdpa | eager
  - CLI toggles: --enable-fa2 / --disable-fa2
  - Force eager: set --attn-impl eager (or env ATTN_IMPL=eager)
  - Strict mode (no fallback): set env ATTN_STRICT=true (or CLI via env)

Batching / Limits / Backpressure
- MAX_INPUT_TOKENS (alias: MAX_CONTEXT_TOKENS): default 16384
- MAX_NEW_TOKENS_DEFAULT: default 128
- MAX_BATCH_SIZE: default 1
- BATCH_TIMEOUT_MS: default 0
- MAX_QUEUE_SIZE: default 256 (server returns 503 when queue is full)

GPU Selection
- Default: single GPU (GPU_COUNT=1) if any visible
- GPU_COUNT: N (use first N visible GPUs)
- GPU_IDS: "i,j,..." (use explicit devices by index)
- Do not set CUDA_VISIBLE_DEVICES in the container; let Docker/K8s restrict devices.

Auth (optional but recommended)
- API_KEY: when set to a non-empty value, protects:
  - GET /openapi.json, GET /docs, GET /redoc
  - GET /v1/models
  - POST /v1/chat/completions
- Clients must send the key via:
  - Authorization: Bearer YOUR_KEY
- When API_KEY is unset or set to "EMPTY", auth is disabled.

Server
- HOST: default 0.0.0.0
- PORT: default 8000
- LOG_LEVEL: default info

Precedence: CLI flags > environment variables > defaults.

## Configuring Inference Defaults and Model Name

These flags control the default sampling behavior and the display model name used by the API and docs.

- --default-temperature F (env TEMPERATURE_DEFAULT; default 0.8)
- --default-top-p F (env TOP_P_DEFAULT; default 0.9)
- --default-max-tokens N (env MAX_NEW_TOKENS_DEFAULT; default 128; requests are clamped to this)
- --default-seed N (env SEED_DEFAULT; default 42; used for sampling reproducibility)
- --default-sampling true|false (env DO_SAMPLE_DEFAULT; default true)
- --model-name STR (env MODEL_NAME; default "model")
  - The Swagger app title is set to: "MODEL NAME - Docomentsio"
  - The OpenAI-like endpoint enforces the model name contract:
    - If "model" field is provided in the request, it MUST equal the configured MODEL_NAME
    - If omitted, the server assumes MODEL_NAME
  - The /v1/models endpoint returns the configured MODEL_NAME

Examples
```bash
# Deterministic defaults (greedy decoding)
--default-sampling false

# Sampling defaults
--default-sampling true --default-temperature 0.7 --default-top-p 0.95 --default-seed 123

# Custom model display name (impacts Swagger title and API contract)
--model-name "omega"
```

## Run (Docker)

Notes:
- Local model path: mount a read-only volume with -v /host/models/omega-vlm-prod-v0.1:/models/omega-vlm-prod-v0.1:ro and set --model-path /models/omega-vlm-prod-v0.1.
- Model ID: set MODEL_PATH to a model ID (e.g., model/omega-vlm-prod-v0.1) and ensure network access and any required token.

### Example: Local model folder, 1 GPU (default)

```bash
docker run --rm \
  --gpus "count=1" \
  -p 8000:8000 \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  -v /host/models/omega-vlm-prod-v0.1:/models/omega-vlm-prod-v0.1:ro \
  llm-fastapi:latest \
  --model-path /models/omega-vlm-prod-v0.1 \
  --model-name "model"
```

### Example: Model ID (public), 1 GPU, BF16 (default), FlashAttention 2 (default)

```bash
docker run --rm \
  --gpus "count=1" \
  -p 8000:8000 \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  llm-fastapi:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --model-name "omega"
```

### Example: Model ID (private) with token

```bash
docker run --rm \
  --gpus "count=1" \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  llm-fastapi:latest \
  --model-path model/omega-vlm-prod-v0.1
```

### Example: Toggle dtype and FA2

```bash
# FP16 with FA2 disabled
docker run --rm \
  --gpus "count=1" \
  -p 8000:8000 \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  llm-fastapi:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --dtype fp16 --disable-fa2
```

### Example: Force eager attention (with optional strict mode)

```bash
# Force eager (fallback allowed: server tries eager, then fallback to SDPA if needed)
docker run --rm \
  --gpus "count=1" \
  -p 8000:8000 \
  -e ATTN_IMPL=eager \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  llm-fastapi:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --attn-impl eager

# Force eager strictly (no fallback; startup fails if eager isn't available)
docker run --rm \
  --gpus "count=1" \
  -p 8000:8000 \
  -e ATTN_IMPL=eager \
  -e ATTN_STRICT=true \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  llm-fastapi:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --attn-impl eager
```

Compose/Kubernetes:
- Compose: set ATTN_IMPL: eager and optionally ATTN_STRICT: "true" under environment:.
- Kubernetes: add env:
  ```yaml
  - name: ATTN_IMPL
    value: "eager"
  - name: ATTN_STRICT
    value: "true"   # optional strict mode
  ```

### Example: Multi-GPU by explicit ids (replicate per GPU)

```bash
docker run --rm \
  --gpus "device=0,1" \
  -p 8000:8000 \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  llm-fastapi:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --gpu-ids "0,1"
```

### Example: Throughput tuning and backpressure

```bash
docker run --rm \
  --gpus "count=1" \
  -p 8000:8000 \
  -e MAX_BATCH_SIZE=8 \
  -e BATCH_TIMEOUT_MS=6 \
  -e MAX_QUEUE_SIZE=256 \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  llm-fastapi:latest \
  --model-path model/omega-vlm-prod-v0.1
```

## docker-compose (Compose v2)

Minimal compose that builds locally and uses 1 GPU by default:

```yaml
# compose.yaml
services:
  llm:
    image: llm-fastapi:latest
    build: .
    ports:
      - "8000:8000"
    environment:
      MODEL_PATH: model/omega-vlm-prod-v0.1
      DTYPE: bf16
      ATTN_IMPL: flash_attention_2
      MAX_BATCH_SIZE: "8"
      BATCH_TIMEOUT_MS: "6"
      MAX_QUEUE_SIZE: "256"
      API_KEY: ${API_KEY:-}   # optional
      # HUGGING_FACE_HUB_TOKEN: ${HF_TOKEN:-}   # optional
    # Option A: request N GPUs (NVIDIA runtime required)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: ["gpu"]
    command: ["--gpu-count","1","--model-name","model"]
```

Local model path with volume:

```yaml
services:
  llm:
    image: llm-fastapi:latest
    build: .
    ports: ["8000:8000"]
    volumes:
      - /host/models/omega-vlm-prod-v0.1:/models/omega-vlm-prod-v0.1:ro
    environment:
      MODEL_PATH: /models/omega-vlm-prod-v0.1
      API_KEY: ${API_KEY:-}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: ["gpu"]
```

Notes:
- Some Compose setups ignore deploy: outside Swarm. If so, prefer docker run --gpus ... or use a CLI wrapper. Alternatively, newer Compose supports gpus: but behavior varies by version.
- Explicit device selection can be done at runtime, e.g., --gpus "device=0,1" and command: ["--gpu-ids","0,1"].

## Kubernetes (optional)

With the NVIDIA device plugin:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-fastapi
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-fastapi
  template:
    metadata:
      labels:
        app: llm-fastapi
    spec:
      containers:
        - name: server
          image: llm-fastapi:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: "model/omega-vlm-prod-v0.1"
            - name: DTYPE
              value: "bf16"
            - name: ATTN_IMPL
              value: "flash_attention_2"
            - name: MAX_BATCH_SIZE
              value: "8"
            - name: BATCH_TIMEOUT_MS
              value: "6"
            - name: MAX_QUEUE_SIZE
              value: "256"
            - name: API_KEY
              valueFrom:
                secretKeyRef:
                  name: omega-api
                  key: apiKey
          args: ["--gpu-count","1","--model-name","model"]
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 20
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 20
          resources:
            limits:
              nvidia.com/gpu: 1
          # Optional: mount a local model PVC instead of a remote model ID
          # volumeMounts:
          #   - name: model
          #     mountPath: /models/omega-vlm-prod-v0.1
      # volumes:
      #   - name: model
      #     persistentVolumeClaim:
      #       claimName: omega-vlm-prod-v0.1-pvc
```

Kubernetes automatically constrains visible GPUs; no need to set CUDA_VISIBLE_DEVICES.

## API Usage

Health:

```bash
curl -s http://localhost:8000/health | jq
```

### Multi-Modal Queries (Vision-Language Model)

This server supports **OpenAI-compatible multi-modal requests** with text and images. For complete documentation and examples, see [MULTIMODAL_USAGE.md](MULTIMODAL_USAGE.md).

**Quick Example:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/photo.jpg"
            }
          }
        ]
      }
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

**Supported Image Formats:**
- Remote URLs: `https://example.com/image.jpg`
- Base64 data URLs: `data:image/jpeg;base64,...`
- Formats: JPEG, PNG, GIF, WEBP
- Max images per request: 10 (configurable via `MAX_IMAGES_PER_REQUEST`)
- Max image dimension: 1024px (auto-resized via `MAX_IMAGE_SIZE`)

**Python Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    },
    json={
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/image.jpg"
                        }
                    }
                ]
            }
        ]
    }
)
print(response.json())
```

See `examples/multimodal_examples.py` for more examples.

OpenAI-compatible (Text-Only):

Important:
- The model field must equal the configured display model name (MODEL_NAME). If omitted, the server assumes MODEL_NAME.
- stream=false and n=1 are supported in this version.
- Response includes latency_seconds.

```bash
# Example where MODEL_NAME is set to "model" (default)
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_SECURE_TOKEN" \
  -d '{
    "model": "model",
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "temperature": 0.8,
    "stream": false
  }' | jq
```

Defaults when omitted:
- top_p: 0.9
- max_tokens: 128 (clamped to server default)
- n: 1
- stream: false
- Sampling default is controlled by DO_SAMPLE_DEFAULT (see flags)
- Seed default: 42 (used when sampling for reproducibility)

Simple generate:

```bash
curl -s http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "Write a haiku about GPUs",
    "parameters": {"max_new_tokens": 64}
  }' | jq
```

Apply chat template:

```bash
curl -s http://localhost:8000/v1/apply_template \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }' | jq
```

List models (protected when API_KEY is set):

```bash
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer YOUR_SECURE_TOKEN" | jq
```

## Performance Tuning

- MAX_BATCH_SIZE: Increase for higher throughput, at cost of latency and memory.
- BATCH_TIMEOUT_MS: Micro-batch collection window. 2–8 ms typical.
- MAX_INPUT_TOKENS / MAX_CONTEXT_TOKENS and MAX_NEW_TOKENS_DEFAULT: Clamp to prevent extreme requests.
- DTYPE: bf16 (balanced), fp16 (smaller memory, may underflow some models), fp32 (largest memory).
- ATTN_IMPL=flash_attention_2 when installed; otherwise server auto-falls back to SDPA, or is strict when ATTN_STRICT=true.
- torch.compile is attempted automatically to reduce Python overhead; falls back if unsupported.
- Backpressure: When all queues reach MAX_QUEUE_SIZE, server returns 503 to shed load.

## Security and Tokens

- If your model ID requires authentication, provide a token via a suitable environment variable (e.g., HUGGING_FACE_HUB_TOKEN).
- TRUST_REMOTE_CODE defaults to ON for maximum compatibility with custom modeling code. Set TRUST_REMOTE_CODE=0 to hard-disable.
- API_KEY (when set) protects /docs, /openapi.json, /redoc, /v1/models, and /v1/chat/completions. In Swagger, use the "Authorize" button (Bearer).

## Troubleshooting

- No CUDA GPUs are visible
  - Ensure nvidia-container-toolkit is installed and Docker is configured with the NVIDIA runtime.
  - Verify with: docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
- flash-attn build fails during pip install
  - The image tolerates failure and falls back to SDPA automatically (unless strict via ATTN_STRICT=true).
- OOM / CUDA out of memory
  - Lower MAX_BATCH_SIZE and/or MAX_INPUT_TOKENS; reduce max tokens per request.
  - Limit GPU count via --gpu-count or use a smaller model.

## Development notes

- Single uvicorn worker (workers=1) is intentional to avoid duplicate model loads; asyncio + micro-batching handles concurrency.
- One model replica per selected GPU is created inside the process; router dispatches to the least-queued worker.
- Missing-first-token issues are mitigated: decoding returns only the generated tail tokens after the true prompt length.

## Progress logs and per-request metrics

Disabled by default. To enable, pass `--progress true` or set `PROGRESS_LOGS=1`.

- Startup heartbeat:
  - Prints a dot "." every 0.5s during model load and a line like "[startup] loading… 15s" every ~5s, so the terminal never looks idle.
  - Structured step logs per device:
    - "[startup][cuda:0] Loading tokenizer…", "Tokenizer ready"
    - "Loading model (attn=…, dtype=…)", "Falling back to sdpa attention" when needed
    - "torch.compile=on|off", "Warmup…", "Ready."
- Readiness line:
  - On success: "[server] Ready. Health: http://HOST:PORT/health"
  - On degraded startup (load failure): "[server] Not ready (degraded). Health: … Error: …"
- vLLM-like request metrics (after each request completes):
  - Chat: "[req] chat 200 id=abcd123 latency=0.532s gen=0.471s model=model tokens: prompt=45 completion=128 total=173 tps=271.7 do_sample=true temp=0.8 top_p=0.9 seed=42"
  - Generate: "[req] generate 200 id=wxyz789 gen=0.095s tokens: completion=64 tps=673.2 do_sample=false"
  - Errors: "[req] chat 500 id=abcd123 error="generation failed: …""
  - tps = completion_tokens / generation_seconds; prompt/completion token counts are estimated via tokenizer.
- Control:
  - Enable/disable globally with `--progress true|false` (entrypoint) or `PROGRESS_LOGS=1|0`.

## DEBUG mode

Enable deep debug logging, including:
- Root logger, uvicorn access/error logs set to DEBUG
- transformers and huggingface_hub verbosity set to DEBUG
- Full exception tracebacks printed to the terminal from handlers and the global exception handler

How to enable:
- CLI: pass --debug true to the entrypoint
- Env: set DEBUG=1

Notes:
- In DEBUG mode, the entrypoint automatically bumps --log-level to debug unless you explicitly set --log-level yourself.
- Progress logs remain enabled by default; you can still disable them with --progress false.

Example:
```bash
docker run -d --name omega-api --restart unless-stopped \
  --gpus "count=1" \
  -p 8000:8000 \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --model-name "model" \
  --debug true
```

## Redacting model source and download logs

To prevent revealing where the model is coming from and to keep the terminal clean during first-run downloads:

- Suppresses HF/transformers download progress bars and incidental prints.
- Redacts model path and revision in startup logs and /health payload.
- Keeps startup heartbeat and generic “loading model …” lines.

Usage:
- CLI: --redact-source true   (default)
- Env: REDACT_SOURCE=1

Note: DEBUG still prints stack traces for errors, but model identifiers remain redacted.

## Cache location

Default cache directory inside the container:
- /root/.cache/usinc/models

Override cache location:
- CLI: --cache-dir /path/to/cache
- Env: set HF_HOME and/or HF_HUB_CACHE to your desired path

Example (persist cache on host):
```bash
docker run -d --name omega-api --restart unless-stopped \
  --gpus "count=1" \
  -p 8000:8000 \
  -e API_KEY="YOUR_SECURE_TOKEN" \
  -v /data/models-cache:/models-cache \
  docker.io/arpitsh018/omega-vlm-inference-engine:latest \
  --model-path model/omega-vlm-prod-v0.1 \
  --model-name "model" \
  --cache-dir /models-cache
```

Notes:
- When redaction is enabled (default), cache paths are not printed in logs or /health.
- Hugging Face libraries honor these envs automatically.
- TRANSFORMERS_CACHE is deprecated by Transformers and is not used here; prefer HF_HOME and/or HF_HUB_CACHE.

## Full deploy parameter reference

Precedence: CLI flags > environment variables > built-in defaults.

### GPU selection
- --gpu-ids / GPU_IDS (default: unset): Comma-separated CUDA device indices to use inside the container (e.g., "0,1"); overrides GPU_COUNT.
- --gpu-count / GPU_COUNT (default: 1 if GPUs visible): Number of GPUs to use (first N visible); ignored when GPU_IDS is set.

### Model
- --model-path / MODEL_PATH (default: 5techlab-research/test_iter3): Path or Hugging Face repo ID of the model to load.
- --model-id (alias of --model-path) (default: same as MODEL_PATH): Backward-compatible alias; prefer --model-path.
- --model-name / MODEL_NAME (default: "model"): Display model name used by the API/docs; OpenAI requests must use this exact name.
- MODEL_REVISION (default: unset): Optional HF branch/tag/commit to pin a specific model revision.
- TRUST_REMOTE_CODE (default: 1): Allow custom modeling code from the repo; required by many community models.

### Attention kernels
- --attn-impl / ATTN_IMPL (default: flash_attention_2): Attention implementation: flash_attention_2 | sdpa | eager; may fall back to sdpa.
- --enable-fa2 (no env; sets ATTN_IMPL=flash_attention_2): Shortcut to force FlashAttention 2 (takes precedence over --attn-impl).
- --disable-fa2 (no env; sets ATTN_IMPL=sdpa): Shortcut to force SDPA (takes precedence over --attn-impl).
- ATTN_STRICT (default: false): When true, disables fallback; startup fails if the selected attention implementation is unavailable.

### Compute
- --dtype / DTYPE (default: bf16): Compute dtype for model weights/activations: bf16 | fp16 | fp32.

### Batching, limits, backpressure
- --max-batch-size / MAX_BATCH_SIZE (default: 8): Maximum requests per micro-batch for throughput.
- --batch-timeout-ms / BATCH_TIMEOUT_MS (default: 6): Micro-batch collection window in milliseconds.
- MAX_INPUT_TOKENS or MAX_CONTEXT_TOKENS (default: 16384): Maximum input context tokens; longer inputs are left-truncated.
- --default-max-tokens / MAX_NEW_TOKENS_DEFAULT (default: 128): Maximum new tokens per request; higher request values are clamped.
- MAX_QUEUE_SIZE (default: 256): Queue depth cap per worker before 503 backpressure is returned.

### Generation defaults
- --default-sampling / DO_SAMPLE_DEFAULT (default: true): Enable sampling by default; when false, decoding is greedy/deterministic.
- --default-temperature / TEMPERATURE_DEFAULT (default: 0.8): Temperature used when sampling if not provided by the request.
- --default-top-p / TOP_P_DEFAULT (default: 0.9): Nucleus sampling p used when sampling if not provided.
- --default-seed / SEED_DEFAULT (default: 42): Default RNG seed used when sampling for reproducible outputs.

### Server process
- --host / HOST (default: 0.0.0.0): Address to bind the HTTP server on.
- --port / PORT (default: 8000): Port to bind the HTTP server on.
- --log-level / LOG_LEVEL (default: info): Uvicorn log level: critical | error | warning | info | debug | trace.
- --progress / PROGRESS_LOGS (default: false): Enable startup heartbeat (dot animation + "[startup] loading… Xs") and per-request metrics logs (latency, gen time, tokens, tokens/sec); set true to enable.
- --cache-dir / CACHE_DIR (default: /root/.cache/usinc/models): Sets HF_HUB_CACHE to control where model weights are stored.
- --fast-usage / FAST_USAGE (default: true): Use worker-provided token counts to avoid re-tokenization for usage/TPS accounting.
- --redact-source / REDACT_SOURCE (default: true/1): Hide Hugging Face download progress and redact model path/revision from logs and /health.
- --debug / DEBUG (default: false/0): Enable deep debug logging and full tracebacks; also bumps uvicorn log-level to debug unless explicitly set via --log-level.
- UVICORN_WORKERS (fixed to 1): Single worker enforced to avoid duplicate model loads; concurrency handled via asyncio + batching.
- JSON responses use ORJSON for faster serialization by default.

### Auth and external tokens
- API_KEY (default: EMPTY): When set, protects /docs, /openapi.json, /redoc, /v1/models, and /v1/chat/completions (via Authorization: Bearer).
- HUGGING_FACE_HUB_TOKEN or HF_TOKEN (default: unset): HF access token for private models/repos; either variable is accepted.

### Advanced runtime (usually leave at defaults)
- PYTORCH_CUDA_ALLOC_CONF (default: expandable_segments:True): Improves CUDA memory allocator behavior for long-running servers.
- TOKENIZERS_PARALLELISM (default: false): Avoids excessive tokenizer threading in server workloads.
- HF_HUB_ENABLE_HF_TRANSFER (default: 1): Enables faster artifact transfer when hf_transfer is installed.
- CUDA_DEVICE_MAX_CONNECTIONS (default: 1): Recommended setting for Flash/SDPA kernel performance.

Notes
- Model name contract: If "model" is omitted in OpenAI requests, the server assumes MODEL_NAME; if provided, it must equal MODEL_NAME.
- Attention strict mode: Set ATTN_STRICT=true to fail fast rather than falling back to SDPA when Flash/eager is unavailable.

## Operate on AKS (quick commands)

Prereqs:
- az aks get-credentials -g YOUR_RESOURCE_GROUP -n YOUR_AKS_CLUSTER
- Verify GPU support on nodes:
  ```bash
  kubectl get nodes -o wide
  kubectl get ds -n kube-system | grep -i nvidia
  ```

Deploy and get external IP:
```bash
# Create API key secret (change the token)
kubectl create secret generic omega-llm-secret --from-literal=apiKey=YOUR_SECURE_TOKEN

# Apply the manifest provided in this repo
kubectl apply -f deploy/aks-omega-vlm.yaml

# Watch rollout and service
kubectl rollout status deployment/omega-llm
kubectl get svc omega-llm -w
```

Enable logs (defaults are low-latency with logs off):
```bash
# Turn on progress logs and debug without editing YAML
kubectl set env deployment/omega-llm PROGRESS_LOGS=1 DEBUG=1
kubectl rollout status deployment/omega-llm
```

Fetch logs and events:
```bash
POD=$(kubectl get pods -l app=omega-llm -o jsonpath='{.items[0].metadata.name}')
kubectl logs -f "$POD"

# If no logs, check events/describe for scheduling or image issues
kubectl get events --sort-by=.metadata.creationTimestamp | tail -n 100
kubectl describe pod "$POD" | sed -n '/Events/,$p'
```

Connectivity checks:
```bash
# Port-forward to test locally if LB isn't ready yet
kubectl port-forward deploy/omega-llm 8000:8000
curl -s http://localhost:8000/health | jq
```

Change runtime flags on the fly (optional):
```bash
# Update CLI flags without editing YAML
kubectl set args deployment/omega-llm -- \
  --model-name omega \
  --gpu-count 8 \
  --model-path /models/model_path \
  --dtype bf16 \
  --attn-impl flash_attention_2 \
  --max-batch-size 8 \
  --batch-timeout-ms 6 \
  --host 0.0.0.0 \
  --port 8000 \
  --default-temperature 0.8 \
  --default-top-p 0.95 \
  --default-seed 42

kubectl rollout status deployment/omega-llm
```

Notes:
- If your node has fewer GPUs, lower both --gpu-count and resources.limits.nvidia.com/gpu accordingly and re-apply.
- For private HF repos, add HUGGING_FACE_HUB_TOKEN as an env referencing a Secret.
- By default, PROGRESS_LOGS=0 for best latency. Enable it temporarily while debugging.

## Changelog

v0.2.1
- Add configurable inference defaults via flags/env: temperature, top_p, max_new_tokens, seed, and default sampling
- Add model display name configuration (--model-name) and enforce model name in OpenAI-like endpoint
- Swagger/UI title: "MODEL NAME - Docomentsio"; Swagger shows Bearer auth
- Protect /docs, /openapi.json, /redoc, and /v1/models with API key
- Seeded sampling with torch.Generator for reproducibility
- Fix first-token drop by slicing with true input lengths and decoding only generated tails
