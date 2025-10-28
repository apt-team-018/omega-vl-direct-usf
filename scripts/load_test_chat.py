#!/usr/bin/env python3
import argparse
import concurrent.futures as cf
import json
import os
import random
import statistics
import sys
import time
import traceback
import urllib.request
import urllib.error
import uuid
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Base prompts (cycled as needed)
# -----------------------------
def base_prompts() -> List[str]:
    """
    Generate 700+ long-form prompts designed to elicit long answers.
    Each prompt explicitly asks for 800–1200 words and structured sections.
    """
    categories = [
        "Large Language Models", "Transformers", "Reinforcement Learning", "Computer Vision",
        "Natural Language Processing", "Speech Recognition", "Time Series Forecasting",
        "Graph Neural Networks", "Recommendation Systems", "Anomaly Detection",
        "Distributed Training", "Model Serving", "MLOps", "Feature Stores", "Data Versioning",
        "Prompt Engineering", "Retrieval Augmented Generation", "Vector Databases",
        "Evaluation and Benchmarks", "Few-shot Learning", "Fine-tuning Techniques",
        "Quantization", "Pruning and Distillation", "GPU Performance Tuning",
        "High-Performance Computing", "Parallel Computing", "CUDA Kernels", "Compilers",
        "Systems Design", "Microservices", "Event-Driven Architectures", "API Gateways",
        "Load Balancing", "Caching Strategies", "Database Sharding", "Consistency Models",
        "Streaming Systems", "Message Queues", "Data Lakes", "Data Warehouses", "ETL and ELT",
        "Observability", "Logging and Tracing", "SRE Practices", "Chaos Engineering",
        "Security Architecture", "Identity and Access Management", "Zero Trust Networks",
        "Network Protocols", "HTTP/3 and QUIC", "Web Performance Optimization",
        "Frontend Performance", "Mobile Performance", "Kubernetes", "Service Meshes",
        "Serverless Computing", "Edge Computing", "Content Delivery Networks",
        "Internet of Things", "Blockchain", "Smart Contracts", "Fintech Systems",
        "Healthcare Interoperability", "E-commerce Platforms", "Search Systems",
        "Analytics Platforms", "Privacy and Compliance", "GDPR Readiness",
        "A/B Testing", "Experimentation Platforms", "Cost Optimization",
        "Cloud Architecture", "Multi-Region Design", "Disaster Recovery",
        "Data Modeling", "OLTP vs OLAP", "Caching Layers", "API Design", "GraphQL vs REST",
        "Testing Strategies", "Continuous Delivery", "Canary Releases", "Blue-Green Deployments",
        "Feature Flagging", "Observability KPIs", "SLOs and Error Budgets",
        "Backpressure and Flow Control", "Idempotency and Exactly-Once", "Dead Letter Queues",
        "Circuit Breakers and Retries", "Rate Limiting", "Token Buckets",
    ]
    subthemes = [
        "architecture", "scaling strategies", "trade-offs", "security considerations",
        "performance tuning", "failure modes and resiliency", "monitoring and observability",
        "testing strategy", "deployment pipeline", "cost optimization", "data modeling",
        "consistency and availability", "benchmarking methodology", "best practices",
        "anti-patterns", "migration plan", "case study", "roadmap and phased rollout",
        "governance and compliance", "real-world examples"
    ]
    templates = [
        "Write a comprehensive, 800–1200 word guide on {topic}. Include definitions, background, an architecture diagram described in text, step-by-step implementation, code snippets, performance considerations, pitfalls, and best practices.",
        "Compose an in-depth technical deep dive (800–1200 words) about {topic}. Cover history, core concepts, algorithms, complexity, real-world case studies, security aspects, and a conclusion with actionable recommendations.",
        "Prepare a long-form tutorial (800–1200 words) on {topic} with sections: Overview, Design, Implementation, Benchmarks, Security, Failure Modes, Observability, Cost, and a Final Checklist.",
        "Draft an architectural blueprint (800–1200 words) for {topic}. Detail service boundaries, data flow, storage choices, scaling, caching, backpressure, error handling, disaster recovery, and operational playbooks.",
        "Write a thorough comparison and trade-off analysis (800–1200 words) focused on {topic}. Provide decision matrices, example scenarios, and guidance for different scales and constraints.",
        "Produce a practitioner's handbook (800–1200 words) on {topic}. Include tools, commands, code samples, runbooks, incident response steps, and postmortem lessons.",
        "Create a research-style survey (800–1200 words) summarizing {topic}. Include taxonomy, state of the art, evaluation criteria, datasets, benchmarks, and open challenges.",
        "Write a step-by-step implementation guide (800–1200 words) for {topic}. Include sample APIs, data schemas, deployment scripts, monitoring KPIs, and testing strategies.",
    ]

    # Build at least 700 prompts by combining categories, subthemes, and templates.
    prompts: List[str] = []
    for i, cat in enumerate(categories):
        for j, sub in enumerate(subthemes):
            t = templates[(i + j) % len(templates)]
            prompts.append(t.format(topic=f"{cat} — {sub}"))
            if len(prompts) >= 700:
                break
        if len(prompts) >= 700:
            break

    # If fewer than 700 (unlikely), fill by cycling combinations deterministically.
    idx = 0
    while len(prompts) < 700:
        t = templates[idx % len(templates)]
        cat = categories[idx % len(categories)]
        sub = subthemes[idx % len(subthemes)]
        prompts.append(t.format(topic=f"{cat} — {sub}"))
        idx += 1

    return prompts


# A small bag of random words to make prompts unique (avoids caches/WAF heuristics)
WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu", "amber", "azure", "crimson", "ember", "frost",
    "glimmer", "harbor", "ivory", "jade", "keystone", "lagoon", "magma",
    "nebula", "onyx", "pebble", "quartz", "raven", "saffron", "topaz",
    "umber", "velvet", "willow", "zephyr", "blossom", "cinder", "dawn",
    "ember", "flint", "grove", "hollow", "island", "jungle", "knoll",
    "lyric", "meadow", "nectar", "orchid", "prairie", "quiver", "rapids",
    "solace", "thicket", "upland", "vista", "wander", "yonder", "zenith"
]


def make_dynamic_prompts(total: int, random_words: int, add_suffix: bool, seed: Optional[int]) -> List[str]:
    rng = random.Random(seed) if seed is not None else random.Random()
    bases = base_prompts()
    prompts: List[str] = []
    for i in range(total):
        base = bases[i % len(bases)]
        words = " ".join(rng.choice(WORDS) for _ in range(max(0, random_words)))
        id8 = uuid.uuid4().hex[:8] if add_suffix else ""
        suffix_parts = []
        # Always include a deterministic index to guarantee uniqueness traceability
        suffix_parts.append(f"idx:{i}")
        if add_suffix:
            suffix_parts.append(f"id:{id8}")
        if random_words > 0 and words:
            suffix_parts.append(words)
        suffix = (" [" + " ".join(suffix_parts) + "]") if suffix_parts else ""
        prompts.append(base + suffix)
    return prompts


# -----------------------------
# Request building + HTTP call
# -----------------------------
def build_request_payload(prompt: str, max_tokens: Optional[int], temperature: Optional[float], top_p: Optional[float], model: Optional[str]) -> Dict[str, Any]:
    # OpenAI-style Chat Completions payload; omit "model" to use server default
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    if max_tokens is not None:
        try:
            payload["max_tokens"] = int(max_tokens)
        except Exception:
            pass
    if temperature is not None:
        try:
            payload["temperature"] = float(temperature)
        except Exception:
            pass
    if top_p is not None:
        try:
            payload["top_p"] = float(top_p)
        except Exception:
            pass
    if model is not None:
        try:
            payload["model"] = str(model)
        except Exception:
            pass
    return payload


def post_chat_completion(
    url: str,
    prompt: str,
    api_key: Optional[str],
    timeout: float = 120.0,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    model: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    payload = build_request_payload(prompt, max_tokens, temperature, top_p, model)
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        # Browser-like headers avoid some WAF/CDN heuristics (e.g., Cloudflare 1010)
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": url,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url=url, data=data, headers=headers, method="POST")

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            latency = time.perf_counter() - t0
            status = resp.status
            try:
                parsed = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                parsed = {"raw": body.decode("utf-8", errors="replace")}
            usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
            completion_tokens = int(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0
            prompt_tokens = int(usage.get("prompt_tokens", 0)) if isinstance(usage, dict) else 0
            total_tokens = int(usage.get("total_tokens", 0)) if isinstance(usage, dict) else 0
            # Extract text length (optional)
            text_len = 0
            try:
                choices = parsed.get("choices", [])
                if choices and isinstance(choices, list):
                    text = choices[0].get("message", {}).get("content", "") or ""
                    text_len = len(text)
            except Exception:
                text_len = 0
            result = {
                "ok": 200 <= status < 300,
                "status": status,
                "latency_s": latency,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "text_len": text_len,
                "error": None,
                "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
            }
            return (result["ok"], result)
    except urllib.error.HTTPError as e:
        latency = time.perf_counter() - t0
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return (False, {
            "ok": False,
            "status": e.code,
            "latency_s": latency,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "text_len": 0,
            "error": f"HTTPError {e.code}: {err_body}",
            "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        })
    except Exception as e:
        latency = time.perf_counter() - t0
        return (False, {
            "ok": False,
            "status": 0,
            "latency_s": latency,
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
            "text_len": 0,
            "error": f"{type(e).__name__}: {e}",
            "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
        })


# -----------------------------
# Metrics helpers
# -----------------------------
def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * p
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def summarize(results: List[Dict[str, Any]], started_at: float, ended_at: float) -> None:
    latencies = [r["latency_s"] for r in results if r["ok"]]
    total = len(results)
    successes = sum(1 for r in results if r["ok"])
    failures = total - successes

    total_wall = ended_at - started_at
    rps = successes / total_wall if total_wall > 0 else 0.0

    # Token aggregates from successful responses that reported usage
    comp_tokens = [r["completion_tokens"] for r in results if r["ok"] and r["completion_tokens"] > 0]
    prompt_toks = [r["prompt_tokens"] for r in results if r["ok"] and r["prompt_tokens"] > 0]
    total_toks = [r["total_tokens"] for r in results if r["ok"] and r["total_tokens"] > 0]
    sum_comp_tokens = sum(comp_tokens) if comp_tokens else 0
    sum_prompt_tokens = sum(prompt_toks) if prompt_toks else 0
    sum_total_tokens = sum(total_toks) if total_toks else 0
    avg_comp_tokens = (sum_comp_tokens / len(comp_tokens)) if comp_tokens else 0.0
    avg_prompt_tokens = (sum_prompt_tokens / len(prompt_toks)) if prompt_toks else 0.0
    avg_total_tokens = (sum_total_tokens / len(total_toks)) if total_toks else 0.0

    # Tokens/sec (wall clock)
    comp_tps_wall = (sum_comp_tokens / total_wall) if total_wall > 0 else 0.0
    total_tps_wall = (sum_total_tokens / total_wall) if total_wall > 0 else 0.0

    # Per-request completion tokens per latency (approx)
    tps_values = []
    for r in results:
        if r["ok"] and r["completion_tokens"] > 0 and r["latency_s"] > 0:
            tps_values.append(r["completion_tokens"] / r["latency_s"])
    avg_tps_req = statistics.mean(tps_values) if tps_values else 0.0

    # Status breakdown
    status_counts: Dict[int, int] = {}
    for r in results:
        status_counts[r.get("status", 0)] = status_counts.get(r.get("status", 0), 0) + 1

    print("\n=== Load Test Summary ===")
    print(f"Total requests: {total}")
    print(f"Successes:      {successes}")
    print(f"Failures:       {failures}")
    print(f"Total wall time: {total_wall:.3f}s")
    print(f"Throughput:      {rps:.2f} req/s")

    if status_counts:
        print("\nStatus breakdown:")
        for code in sorted(status_counts.keys()):
            print(f"  {code}: {status_counts[code]}")

    if latencies:
        print("\nLatency (successful requests):")
        print(f"  mean: {statistics.mean(latencies):.3f}s")
        print(f"  median (p50): {statistics.median(latencies):.3f}s")
        print(f"  p90: {percentile(latencies, 0.90):.3f}s")
        print(f"  p95: {percentile(latencies, 0.95):.3f}s")
        print(f"  p99: {percentile(latencies, 0.99):.3f}s")
        print(f"  min: {min(latencies):.3f}s")
        print(f"  max: {max(latencies):.3f}s")
    else:
        print("\nNo successful requests to report latency.")

    print("\nTokens (successful responses with usage):")
    print(f"  total prompt tokens:     {sum_prompt_tokens}")
    print(f"  total completion tokens: {sum_comp_tokens}")
    print(f"  total tokens:            {sum_total_tokens}")
    print(f"  avg prompt tokens:       {avg_prompt_tokens:.2f}")
    print(f"  avg completion tokens:   {avg_comp_tokens:.2f}")
    print(f"  avg total tokens:        {avg_total_tokens:.2f}")

    print("\nToken throughput (wall clock):")
    print(f"  completion tokens/sec:   {comp_tps_wall:.2f}")
    print(f"  total tokens/sec:        {total_tps_wall:.2f}")
    print(f"  avg tokens/sec per req (approx): {avg_tps_req:.2f}")

    # Top 5 slowest successful requests
    slowest = sorted([r for r in results if r["ok"]], key=lambda x: x["latency_s"], reverse=True)[:5]
    if slowest:
        print("\nTop 5 slowest successful requests:")
        for i, r in enumerate(slowest, 1):
            print(f"  {i}. {r['latency_s']:.3f}s | tokens={r['completion_tokens']} | prompt='{r['prompt']}'")

    # Print a few errors for visibility
    errs = [r for r in results if not r["ok"]]
    if errs:
        print("\nSample errors (up to 5):")
        for e in errs[:5]:
            print(f"  status={e['status']} latency={e['latency_s']:.3f}s error={e['error']} prompt='{e['prompt']}'")


# -----------------------------
# Runner
# -----------------------------
def run_load_test(
    url: str,
    api_key: Optional[str],
    total: int,
    concurrency: int,
    timeout: float,
    random_words: int,
    random_suffix: bool,
    seed: Optional[int],
    max_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    model: Optional[str],
    ramp_up: Optional[Tuple[int, int]],
) -> None:
    # Guardrails
    if concurrency < 1:
        concurrency = 1
    if concurrency > 1000:
        concurrency = 1000
    if total < 1:
        total = 1
    if concurrency > total:
        concurrency = total

    prompts = make_dynamic_prompts(total, random_words, random_suffix, seed)
    # Enforce uniqueness across all prompts (extremely unlikely to collide due to UUID, but guaranteed)
    seen: set = set()
    for i in range(len(prompts)):
        p = prompts[i]
        if p in seen:
            prompts[i] = p + f" [uniq:{uuid.uuid4().hex[:8]}]"
        seen.add(prompts[i])
    unique_count = len(seen)

    print(f"Dispatching {len(prompts)} requests to {url}")
    print(f"  concurrency={concurrency} timeout={timeout}s random_words={random_words} random_suffix={random_suffix}")
    print(f"  uniqueness: {unique_count}/{len(prompts)} unique prompts")
    if (max_tokens is not None) or (temperature is not None) or (top_p is not None) or (model is not None):
        print(f"  payload: model={model} max_tokens={max_tokens} temperature={temperature} top_p={top_p}")
    if seed is not None:
        print(f"  seed={seed}")
    if ramp_up:
        print(f"  ramp_up: batch_size={ramp_up[0]} sleep_ms={ramp_up[1]}")

    started_at = time.perf_counter()
    results: List[Dict[str, Any]] = []

    def task(p: str) -> Dict[str, Any]:
        ok, res = post_chat_completion(
            url,
            p,
            api_key,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            model=model,
        )
        return res

    with cf.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: List[cf.Future] = []

        def submit_batch(start_idx: int, batch_size: int) -> int:
            end = min(start_idx + batch_size, len(prompts))
            for i in range(start_idx, end):
                futures.append(executor.submit(task, prompts[i]))
            return end

        if ramp_up:
            batch_size, sleep_ms = ramp_up
            idx = 0
            while idx < len(prompts):
                idx = submit_batch(idx, batch_size)
                # Avoid sleeping after final batch
                if idx < len(prompts):
                    time.sleep(max(0.0, float(sleep_ms) / 1000.0))
        else:
            # Fire all at once (executor limits concurrency)
            submit_batch(0, len(prompts))

        for fut in cf.as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                results.append({
                    "ok": False,
                    "status": 0,
                    "latency_s": 0.0,
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                    "text_len": 0,
                    "error": f"WorkerError: {e}",
                    "prompt": "",
                })

    ended_at = time.perf_counter()
    summarize(results, started_at, ended_at)


def parse_ramp_up(val: Optional[str]) -> Optional[Tuple[int, int]]:
    if not val:
        return None
    try:
        parts = [p.strip() for p in val.split(",")]
        if len(parts) != 2:
            raise ValueError("must be 'BATCH,SLEEP_MS'")
        batch = int(parts[0])
        sleep_ms = int(parts[1])
        if batch <= 0 or sleep_ms < 0:
            raise ValueError("batch must be >0 and sleep_ms >=0")
        return (batch, sleep_ms)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"invalid --ramp-up: {e}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Load test for OpenAI-compatible /v1/chat/completions endpoint")
    ap.add_argument("--url", required=True, help="Endpoint URL, e.g. https://host/v1/chat/completions")
    ap.add_argument("--api-key", default=os.getenv("API_KEY"), help="Bearer API key (or set env API_KEY)")
    ap.add_argument("--total", type=int, default=50, help="Total number of requests to send (default: 50)")
    ap.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers (1..1000, default: 10)")
    ap.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds (default: 120)")
    ap.add_argument("--random-words", type=int, default=3, help="Number of random words to append to each prompt (default: 3)")
    ap.add_argument("--no-random-suffix", action="store_true", help="Disable adding a unique id suffix to each prompt")
    ap.add_argument("--seed", type=int, default=None, help="Seed for reproducible prompt randomization")
    ap.add_argument("--ramp-up", type=parse_ramp_up, default=None, metavar="BATCH,SLEEP_MS",
                    help="Send requests in batches, sleeping between batches (e.g., '50,250' for 50 every 250ms)")
    ap.add_argument("--max-tokens", type=int, default=None, help="Request up to this many new tokens per completion")
    ap.add_argument("--temperature", type=float, default=None, help="Sampling temperature (e.g., 0.8)")
    ap.add_argument("--top-p", dest="top_p", type=float, default=None, help="Nucleus sampling p (e.g., 0.95)")
    ap.add_argument("--model", type=str, default=None, help="Model name to include in the request payload")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run_load_test(
            url=args.url,
            api_key=args.api_key,
            total=args.total,
            concurrency=args.concurrency,
            timeout=args.timeout,
            random_words=max(0, int(args.random_words)),
            random_suffix=not args.no_random_suffix,
            seed=args.seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            model=args.model,
            ramp_up=args.ramp_up,
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
