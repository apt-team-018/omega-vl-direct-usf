# Deployment Validation Checklist

## 📋 Pre-Deployment Checklist

### Docker Build Validation

- [ ] **Build completes successfully**
  ```bash
  docker build -t omega-vlm:latest .
  ```
  Expected time: ~2-3 minutes (with pre-built wheel)

- [ ] **Flash Attention status confirmed**
  ```bash
  docker run --rm omega-vlm:latest python3 -c "
  try:
      import flash_attn
      print('✅ Flash Attention installed')
  except ImportError:
      print('ℹ️ SDPA fallback mode')
  "
  ```

- [ ] **Image size reasonable**
  ```bash
  docker images omega-vlm:latest
  ```
  Expected: 10-15 GB (includes model cache if pre-downloaded)

- [ ] **Run verification script**
  ```bash
  chmod +x scripts/verify_build.sh
  ./scripts/verify_build.sh omega-vlm:latest
  ```
  All tests should pass ✅

### Container Registry

- [ ] **Login to registry**
  ```bash
  docker login your-registry.io
  ```

- [ ] **Tag image**
  ```bash
  docker tag omega-vlm:latest your-registry.io/omega-vlm:v0.0.3
  docker tag omega-vlm:latest your-registry.io/omega-vlm:latest
  ```

- [ ] **Push to registry**
  ```bash
  docker push your-registry.io/omega-vlm:v0.0.3
  docker push your-registry.io/omega-vlm:latest
  ```

- [ ] **Verify image in registry**
  ```bash
  docker pull your-registry.io/omega-vlm:v0.0.3
  ```

---

## 🚀 Deployment Checklist

### Kubernetes/AKS Setup

- [ ] **Update deployment manifest**
  ```yaml
  # deploy/aks-omega-vlm.yaml
  image: your-registry.io/omega-vlm:v0.0.3
  imagePullPolicy: Always
  ```

- [ ] **Verify GPU node availability**
  ```bash
  kubectl get nodes -l accelerator=nvidia-h100
  ```

- [ ] **Check GPU resource quota**
  ```bash
  kubectl describe nodes | grep -A 5 "Allocated resources"
  ```

- [ ] **Apply deployment**
  ```bash
  kubectl apply -f deploy/aks-omega-vlm.yaml
  ```

- [ ] **Verify pod creation**
  ```bash
  kubectl get pods -l app=omega-vlm
  ```
  Status should be `Running` within 2-3 minutes

### Initial Health Checks

- [ ] **Check pod logs**
  ```bash
  kubectl logs -f deployment/omega-vlm --tail=50
  ```
  Look for:
  - `✅ Flash Attention:` or `ℹ️ SDPA fallback`
  - `[server] Ready. Health: http://...`
  - No error messages

- [ ] **Verify GPU allocation**
  ```bash
  kubectl logs deployment/omega-vlm | grep -i gpu
  ```
  Should show: `GPU 0: NVIDIA H100 80GB HBM3`

- [ ] **Test health endpoint**
  ```bash
  kubectl port-forward deployment/omega-vlm 8000:8000 &
  curl http://localhost:8000/health | jq
  ```
  
  Expected response:
  ```json
  {
    "ready": true,
    "attn": "flash_attention_2",  // or "sdpa"
    "devices": ["cuda:0"],
    "device_names": ["NVIDIA H100 80GB HBM3"]
  }
  ```

---

## ✅ Post-Deployment Validation

### Functional Testing

- [ ] **Test basic chat completion**
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
      "model": "model",
      "messages": [{"role": "user", "content": "Hello, test message"}],
      "max_tokens": 50
    }' | jq
  ```

- [ ] **Verify response structure**
  - `id`: present
  - `choices[0].message.content`: non-empty
  - `usage.prompt_tokens`: > 0
  - `usage.completion_tokens`: > 0

- [ ] **Test with image (VLM mode)**
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d '{
      "model": "model",
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image", "image": "https://example.com/image.jpg"}
        ]
      }],
      "max_tokens": 100
    }' | jq
  ```

- [ ] **Check response latency**
  - Small requests (<100 tokens): <100ms
  - Medium requests (100-500 tokens): <500ms
  - Large requests (500+ tokens): <2s

### Performance Validation

- [ ] **Monitor token generation speed**
  ```bash
  kubectl logs deployment/omega-vlm | grep "tps="
  ```
  
  Expected tokens/second on H100:
  - **Flash-Attn mode**: 80-110 tps
  - **SDPA mode**: 60-85 tps

- [ ] **Check GPU utilization**
  ```bash
  # From pod
  kubectl exec deployment/omega-vlm -- nvidia-smi
  ```
  
  During inference:
  - GPU utilization: 70-95%
  - Memory usage: 20-40 GB (depends on model)

- [ ] **Verify batch processing**
  ```bash
  # Send concurrent requests
  for i in {1..5}; do
    curl -X POST http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $API_KEY" \
      -d '{"model":"model","messages":[{"role":"user","content":"Test"}]}' &
  done
  wait
  ```

- [ ] **Check queue metrics**
  ```bash
  curl http://localhost:8000/health | jq '.queues'
  ```
  - Queues should process efficiently
  - No persistent backlogs

### Memory & Resource Monitoring

- [ ] **Monitor pod memory usage**
  ```bash
  kubectl top pod -l app=omega-vlm
  ```
  Expected: 30-50 GB (depends on model and batch size)

- [ ] **Check for OOM events**
  ```bash
  kubectl get events --field-selector involvedObject.name=<pod-name>
  ```
  No `OOMKilled` events

- [ ] **Verify HuggingFace cache**
  ```bash
  kubectl exec deployment/omega-vlm -- du -sh /root/.cache/usinc
  ```
  Model should be cached (8-15 GB)

---

## 🔍 Attention Mode Validation

### Flash Attention Mode

If `/health` shows `"attn": "flash_attention_2"`:

- [ ] **Confirm flash-attn version**
  ```bash
  kubectl exec deployment/omega-vlm -- python3 -c "import flash_attn; print(flash_attn.__version__)"
  ```
  Expected: `2.6.1` or newer

- [ ] **Performance baseline**
  - Single request: >80 tokens/sec
  - Batched (2-4): >90 tokens/sec

- [ ] **Memory efficiency**
  - Peak memory: ~30-40 GB for typical VLM
  - Stable during sustained load

### SDPA Fallback Mode

If `/health` shows `"attn": "sdpa"`:

- [ ] **Verify fallback logged**
  ```bash
  kubectl logs deployment/omega-vlm | grep -i "sdpa\|fallback"
  ```
  Should see: `Falling back to sdpa attention` or `Mode: SDPA fallback`

- [ ] **Performance acceptable**
  - Single request: >60 tokens/sec (85-90% of flash-attn)
  - Still very performant on H100

- [ ] **No errors in logs**
  ```bash
  kubectl logs deployment/omega-vlm | grep -i error
  ```

---

## 📊 Load Testing

### Basic Load Test

- [ ] **Run load test script**
  ```bash
  # Install dependencies
  pip install aiohttp tqdm

  # Run load test
  python scripts/load_test_chat.py \
    --url http://localhost:8000 \
    --api-key $API_KEY \
    --requests 100 \
    --concurrency 10
  ```

- [ ] **Verify metrics**
  - Success rate: >95%
  - Average latency: <500ms
  - P95 latency: <1000ms
  - Throughput: >10 requests/sec

### Sustained Load

- [ ] **30-minute sustained test**
  ```bash
  python scripts/load_test_chat.py \
    --url http://localhost:8000 \
    --api-key $API_KEY \
    --duration 1800 \
    --concurrency 5
  ```

- [ ] **Monitor for issues**
  - No memory leaks (stable memory usage)
  - No degrading performance
  - No error spikes
  - GPU temperature stable (<85°C)

### Spike Test

- [ ] **Burst traffic test**
  ```bash
  # Send 50 concurrent requests
  python scripts/load_test_chat.py \
    --url http://localhost:8000 \
    --api-key $API_KEY \
    --requests 50 \
    --concurrency 50
  ```

- [ ] **Verify graceful handling**
  - Queue backpressure working
  - 503 responses if overloaded (expected)
  - Recovery after spike

---

## 🔒 Security & Access

- [ ] **API key authentication working**
  ```bash
  # Should fail without key
  curl http://localhost:8000/v1/chat/completions
  # Expected: 401 Unauthorized
  ```

- [ ] **TLS/HTTPS configured** (if applicable)
  ```bash
  curl https://your-domain.com/health
  ```

- [ ] **Network policies applied**
  ```bash
  kubectl get networkpolicy
  ```

- [ ] **RBAC configured**
  ```bash
  kubectl get rolebindings
  ```

---

## 📈 Monitoring & Alerts

### Metrics Collection

- [ ] **Prometheus scraping configured**
  ```bash
  kubectl get servicemonitor omega-vlm
  ```

- [ ] **Grafana dashboard imported**
  - GPU utilization panel
  - Request latency histogram
  - Token generation rate
  - Queue depth

### Alert Rules

- [ ] **High error rate alert**
  - Threshold: >5% errors in 5 minutes

- [ ] **High latency alert**
  - Threshold: P95 >2s for 5 minutes

- [ ] **GPU memory alert**
  - Threshold: >90% memory usage

- [ ] **Pod restart alert**
  - Trigger: Pod restart detected

### Logging

- [ ] **Centralized logging configured**
  ```bash
  kubectl logs deployment/omega-vlm | grep -i "req\|error\|startup"
  ```

- [ ] **Log retention set**
  - Minimum: 7 days
  - Recommended: 30 days

---

## 🔄 Rollback Plan

### Preparation

- [ ] **Previous version tagged**
  ```bash
  docker tag omega-vlm:v0.0.2 your-registry.io/omega-vlm:rollback
  ```

- [ ] **Rollback command ready**
  ```bash
  kubectl set image deployment/omega-vlm \
    server=your-registry.io/omega-vlm:v0.0.2
  ```

### Triggers for Rollback

- [ ] **Error rate >10%**
- [ ] **Latency >5s sustained**
- [ ] **Memory leaks detected**
- [ ] **Crash loop backoff**

---

## ✅ Sign-off Criteria

### Must-Pass Criteria

- [x] Docker build successful with flash-attn or SDPA
- [ ] Health endpoint returns `ready: true`
- [ ] GPU detected and accessible
- [ ] Basic inference request succeeds
- [ ] Token generation speed meets baseline
- [ ] No critical errors in logs
- [ ] Load test passes (>95% success rate)
- [ ] Memory usage stable

### Nice-to-Have

- [ ] Flash Attention mode active (vs SDPA fallback)
- [ ] <100ms latency for small requests
- [ ] >100 tokens/sec throughput
- [ ] Zero 5xx errors during testing

---

## 📝 Deployment Notes

### Deployment Date
```
Date: _______________
Deployed by: _______________
```

### Configuration
```
Image: _______________
Flash Attention: [ ] Yes [ ] No (SDPA)
GPU Type: _______________
Instance Count: _______________
```

### Performance Baseline
```
Tokens/sec: _______________
P50 Latency: _______________
P95 Latency: _______________
Memory Usage: _______________
```

### Issues Encountered
```
(None / List issues and resolutions)
_______________________________________________
_______________________________________________
```

### Sign-off
```
Technical Lead: _______________ Date: ________
DevOps Lead: _______________ Date: ________
```

---

## 🔗 Quick Reference

### Key Commands
```bash
# Health check
curl http://localhost:8000/health | jq

# Logs
kubectl logs -f deployment/omega-vlm --tail=100

# GPU status
kubectl exec deployment/omega-vlm -- nvidia-smi

# Metrics
curl http://localhost:8000/health | jq '{attn, ready, queues, devices}'

# Restart
kubectl rollout restart deployment/omega-vlm

# Rollback
kubectl rollout undo deployment/omega-vlm
```

### Troubleshooting Links
- [Flash Attention Build Guide](FLASH_ATTENTION_BUILD.md)
- [Docker Deployment Guide](DOCKER_DEPLOYMENT.md)
- [VLM Usage Guide](VLM_USAGE_GUIDE.md)

---

**Deployment Status: [ ] PENDING [ ] IN PROGRESS [ ] COMPLETE [ ] FAILED**