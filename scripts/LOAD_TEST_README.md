# Omega LLM Load Test Guide

This document shows how to run the load testing script against an OpenAI-compatible `/v1/chat/completions` endpoint with large-scale settings.

## Prerequisites

- Python 3 available at `/opt/homebrew/bin/python3` (macOS/Homebrew default path).
- The repository checked out locally with `scripts/load_test_chat.py` present.
- Network access to the target endpoint.

Optional (macOS): if you encounter "Too many open files", raise the file descriptor limit before running in the same terminal:
```
ulimit -n 4096
```

## Sample Command (10,000 total requests, 350-way parallel)

Run:
```
/opt/homebrew/bin/python3 scripts/load_test_chat.py --url "https://jin3kqdhslwqn5-8000.proxy.runpod.net/v1/chat/completions" --api-key "sk-IrR7Bwxtin0haWagUnPrBgq5PurnUz86" --model "omega" --total 10000 --concurrency 350 --max-tokens 1024 --temperature 0.8 --top-p 0.95 --random-words 3 --timeout 300 --seed 42
```

Expected initial output header:
```
Dispatching 10000 requests to https://jin3kqdhslwqn5-8000.proxy.runpod.net/v1/chat/completions
  concurrency=350 timeout=300.0s random_words=3 random_suffix=True
  uniqueness: 10000/10000 unique prompts
  payload: model=omega max_tokens=1024 temperature=0.8 top_p=0.95
  seed=42
```

The script will then execute the load and, upon completion, print a summary including:
- Total requests, successes, failures
- Latency stats (mean, p50, p90, p95, p99, min, max)
- Throughput (req/s)
- Token usage (prompt/completion/total, averages)
- Token throughput (tokens/sec)
- Status code breakdown
- Top 5 slowest successful requests and sample errors

## Notes

- Long answers: `--max-tokens` requests larger responses, but actual length may be clamped by the server configuration.
- Uniqueness: Prompts are made unique using deterministic indices and randomized suffixes to avoid caching/WAF heuristics.
- Optional model: You can pass `--model "model_name"` to include a specific model field in the request payload (the server may validate it).
- Optional ramp-up: To smooth bursts, add `--ramp-up 350,100` (submit in waves of 350 every 100ms while maintaining 350 workers).

For additional run options:
```
/opt/homebrew/bin/python3 scripts/load_test_chat.py --help
```
