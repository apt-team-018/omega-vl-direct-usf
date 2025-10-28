#!/usr/bin/env bash
set -euo pipefail

# GPU-aware entrypoint for the FastAPI LLM server
# Accepts args and maps them to env vars consumed by server.py

GPU_IDS=""
GPU_COUNT=""
MODEL_ID_ENV="${MODEL_ID:-5techlab-research/test_iter3}"
ATTN_IMPL_ENV="${ATTN_IMPL:-flash_attention_2}"
MAX_BATCH_SIZE_ENV="${MAX_BATCH_SIZE:-1}"
BATCH_TIMEOUT_MS_ENV="${BATCH_TIMEOUT_MS:-0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${LOG_LEVEL:-info}"
EXTRA_ARGS=()
MODEL_PATH_ENV="${MODEL_PATH:-${MODEL_ID_ENV}}"
DTYPE_ENV="${DTYPE:-bf16}"
FA2_TOGGLE=""
# Defaults for generation behavior (server.py reads these envs)
TEMP_DEFAULT_ENV="${TEMPERATURE_DEFAULT:-0.8}"
TOP_P_DEFAULT_ENV="${TOP_P_DEFAULT:-0.9}"
MAX_NEW_TOKENS_DEFAULT_ENV="${MAX_NEW_TOKENS_DEFAULT:-128}"
SEED_DEFAULT_ENV="${SEED_DEFAULT:-42}"
DO_SAMPLE_DEFAULT_ENV="${DO_SAMPLE_DEFAULT:-1}"  # 1=true, 0=false
# Progress / heartbeat logs (1=on, 0=off)
PROGRESS_LOGS_ENV="${PROGRESS_LOGS:-0}"
DEBUG_ENV="${DEBUG:-0}"
REDACT_SOURCE_ENV="${REDACT_SOURCE:-1}"
CACHE_DIR_ENV="${CACHE_DIR:-${HF_HUB_CACHE:-/root/.cache/usinc/models}}"
FAST_USAGE_ENV="${FAST_USAGE:-1}"
TRANSFORMERS_INSTALL_PATH="${TRANSFORMERS_INSTALL_PATH:-}"
HF_TOKEN_ARG="${HF_TOKEN:-}"

usage() {
  cat <<EOF
Usage: entrypoint.sh [options] [-- extra uvicorn args]

Options:
  --gpu-ids "i,j"            Comma-separated GPU indices to use inside container (e.g., "0,1")
  --gpu-count N              Number of GPUs to use (e.g., 1, 2). Prefer with: docker --gpus "count=N"
  --model-path STR           Local path or HF repo id (default: ${MODEL_PATH_ENV})
  --model-id STR             Alias of --model-path (for backward compat)
  --model-name STR           Display model name for API/docs (env MODEL_NAME; default: ${MODEL_NAME:-model})
  --hf-token STR             Hugging Face token for private models (env HF_TOKEN or HUGGING_FACE_HUB_TOKEN)
  --attn-impl NAME           Attention impl: flash_attention_2 | sdpa (default: ${ATTN_IMPL_ENV})
  --enable-fa2               Shortcut to set attention impl to flash_attention_2
  --disable-fa2              Shortcut to set attention impl to sdpa
  --dtype bf16|fp16|fp32     Compute dtype (default: ${DTYPE_ENV})
  --max-batch-size N         Micro-batch size (default: ${MAX_BATCH_SIZE_ENV})
  --batch-timeout-ms N       Collect window in ms (default: ${BATCH_TIMEOUT_MS_ENV})
  --default-temperature F    Default temperature (env TEMPERATURE_DEFAULT, default: ${TEMP_DEFAULT_ENV})
  --default-top-p F          Default top_p (env TOP_P_DEFAULT, default: ${TOP_P_DEFAULT_ENV})
  --default-max-tokens N     Default max_new_tokens (env MAX_NEW_TOKENS_DEFAULT, default: ${MAX_NEW_TOKENS_DEFAULT_ENV})
  --default-seed N           Default sampling seed (env SEED_DEFAULT, default: ${SEED_DEFAULT_ENV})
  --default-sampling BOOL    Default sampling on/off (env DO_SAMPLE_DEFAULT, default: ${DO_SAMPLE_DEFAULT_ENV}) [true|false]
  --progress BOOL            Enable progress/heartbeat logs (env PROGRESS_LOGS, default: ${PROGRESS_LOGS_ENV}) [true|false]
  --debug BOOL               Enable DEBUG mode (env DEBUG, default: ${DEBUG_ENV}) [true|false]
  --redact-source BOOL       Redact model source/logs (env REDACT_SOURCE, default: ${REDACT_SOURCE_ENV}) [true|false]
  --cache-dir PATH           Cache directory for model weights (sets HF_HUB_CACHE; default: ${CACHE_DIR_ENV})
  --fast-usage BOOL          Use worker-provided token counts to skip re-tokenization (env FAST_USAGE, default: ${FAST_USAGE_ENV}) [true|false]
  --transformers-path PATH   Install transformers from a path/URL/wheel at container start (env TRANSFORMERS_INSTALL_PATH)
  --host HOST                Bind host (default: ${HOST})
  --port PORT                Bind port (default: ${PORT})
  --log-level LEVEL          Uvicorn log level (default: ${LOG_LEVEL})
  -h, --help                 Show this help

Notes:
- Do not set CUDA_VISIBLE_DEVICES here; let Docker/K8s device plugins restrict devices.
- GPU selection inside the app is controlled via GPU_IDS or GPU_COUNT envs exported here.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-ids)
      GPU_IDS="${2:-}"; shift 2;;
    --gpu-count)
      GPU_COUNT="${2:-}"; shift 2;;
    --model-path)
      MODEL_PATH_ENV="${2:-}"; shift 2;;
    --model-id)
      MODEL_PATH_ENV="${2:-}"; shift 2;;
    --model-name)
      export MODEL_NAME="${2:-model}"; shift 2;;
    --hf-token)
      HF_TOKEN_ARG="${2:-}"; shift 2;;
    --attn-impl)
      ATTN_IMPL_ENV="${2:-}"; shift 2;;
    --max-batch-size)
      MAX_BATCH_SIZE_ENV="${2:-}"; shift 2;;
    --batch-timeout-ms)
      BATCH_TIMEOUT_MS_ENV="${2:-}"; shift 2;;
    --host)
      HOST="${2:-}"; shift 2;;
    --port)
      PORT="${2:-}"; shift 2;;
    --log-level)
      LOG_LEVEL="${2:-}"; shift 2;;
    --dtype)
      DTYPE_ENV="${2:-bf16}"; shift 2;;
    --enable-fa2)
      FA2_TOGGLE="enable"; shift 1;;
    --disable-fa2)
      FA2_TOGGLE="disable"; shift 1;;
    --default-temperature)
      TEMP_DEFAULT_ENV="${2:-${TEMP_DEFAULT_ENV}}"; shift 2;;
    --default-top-p)
      TOP_P_DEFAULT_ENV="${2:-${TOP_P_DEFAULT_ENV}}"; shift 2;;
    --default-max-tokens)
      MAX_NEW_TOKENS_DEFAULT_ENV="${2:-${MAX_NEW_TOKENS_DEFAULT_ENV}}"; shift 2;;
    --default-seed)
      SEED_DEFAULT_ENV="${2:-${SEED_DEFAULT_ENV}}"; shift 2;;
    --default-sampling)
      case "${2:-}" in
        true|True|TRUE|1) DO_SAMPLE_DEFAULT_ENV="1" ;;
        false|False|FALSE|0) DO_SAMPLE_DEFAULT_ENV="0" ;;
        *) echo "Invalid value for --default-sampling: ${2:-} (use true|false)"; exit 1 ;;
      esac
      shift 2;;
    --progress)
      case "${2:-}" in
        true|True|TRUE|1) PROGRESS_LOGS_ENV="1" ;;
        false|False|FALSE|0) PROGRESS_LOGS_ENV="0" ;;
        *) echo "Invalid value for --progress: ${2:-} (use true|false)"; exit 1 ;;
      esac
      shift 2;;
    --debug)
      case "${2:-}" in
        true|True|TRUE|1) DEBUG_ENV="1" ;;
        false|False|FALSE|0) DEBUG_ENV="0" ;;
        *) echo "Invalid value for --debug: ${2:-} (use true|false)"; exit 1 ;;
      esac
      shift 2;;
    --redact-source)
      case "${2:-}" in
        true|True|TRUE|1) REDACT_SOURCE_ENV="1" ;;
        false|False|FALSE|0) REDACT_SOURCE_ENV="0" ;;
        *) echo "Invalid value for --redact-source: ${2:-} (use true|false)"; exit 1 ;;
      esac
      shift 2;;
    --cache-dir)
      CACHE_DIR_ENV="${2:-/root/.cache/usinc/models}"; shift 2;;
    --fast-usage)
      case "${2:-}" in
        true|True|TRUE|1) FAST_USAGE_ENV="1" ;;
        false|False|FALSE|0) FAST_USAGE_ENV="0" ;;
        *) echo "Invalid value for --fast-usage: ${2:-} (use true|false)"; exit 1 ;;
      esac
      shift 2;;
    --transformers-path)
      TRANSFORMERS_INSTALL_PATH="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    --)
      shift; break;;
    *)
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

# Attention impl toggles take precedence over explicit --attn-impl
if [[ "${FA2_TOGGLE:-}" == "enable" ]]; then
  ATTN_IMPL_ENV="flash_attention_2"
elif [[ "${FA2_TOGGLE:-}" == "disable" ]]; then
  ATTN_IMPL_ENV="sdpa"
fi

# Default to one GPU if neither explicit ids nor count provided
if [[ -z "${GPU_IDS}" && -z "${GPU_COUNT}" ]]; then
  GPU_COUNT="1"
fi

# Export GPU selection envs for server.py
if [[ -n "${GPU_IDS}" ]]; then
  export GPU_IDS="${GPU_IDS}"
  unset GPU_COUNT || true
elif [[ -n "${GPU_COUNT}" ]]; then
  export GPU_COUNT="${GPU_COUNT}"
  unset GPU_IDS || true
fi

# Core runtime envs and defaults
export MODEL_PATH="${MODEL_PATH_ENV}"
export MODEL_ID="${MODEL_PATH_ENV}"
export DTYPE="${DTYPE_ENV}"
export ATTN_IMPL="${ATTN_IMPL_ENV}"
export MAX_BATCH_SIZE="${MAX_BATCH_SIZE_ENV}"
export BATCH_TIMEOUT_MS="${BATCH_TIMEOUT_MS_ENV}"
# Expose generation defaults to server.py
export TEMPERATURE_DEFAULT="${TEMP_DEFAULT_ENV}"
export TOP_P_DEFAULT="${TOP_P_DEFAULT_ENV}"
export MAX_NEW_TOKENS_DEFAULT="${MAX_NEW_TOKENS_DEFAULT_ENV}"
export SEED_DEFAULT="${SEED_DEFAULT_ENV}"
export DO_SAMPLE_DEFAULT="${DO_SAMPLE_DEFAULT_ENV}"
export PROGRESS_LOGS="${PROGRESS_LOGS_ENV}"
export DEBUG="${DEBUG_ENV}"
export REDACT_SOURCE="${REDACT_SOURCE_ENV}"
export HF_HUB_CACHE="${CACHE_DIR_ENV}"
export FAST_USAGE="${FAST_USAGE_ENV}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# Harmonize HF auth envs (priority: CLI arg > HF_TOKEN env > HUGGING_FACE_HUB_TOKEN env)
if [[ -n "${HF_TOKEN_ARG}" ]]; then
  export HF_TOKEN="${HF_TOKEN_ARG}"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN_ARG}"
elif [[ -n "${HF_TOKEN:-}" ]] && [[ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]] && [[ -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi

if [[ "${REDACT_SOURCE_ENV}" == "1" ]]; then
  echo "[entrypoint] MODEL=redacted DTYPE=$DTYPE ATTN_IMPL=$ATTN_IMPL MAX_BATCH_SIZE=$MAX_BATCH_SIZE BATCH_TIMEOUT_MS=$BATCH_TIMEOUT_MS"
else
  echo "[entrypoint] MODEL_PATH=$MODEL_PATH DTYPE=$DTYPE ATTN_IMPL=$ATTN_IMPL MAX_BATCH_SIZE=$MAX_BATCH_SIZE BATCH_TIMEOUT_MS=$BATCH_TIMEOUT_MS"
fi
echo "[entrypoint] Defaults: TEMPERATURE_DEFAULT=$TEMPERATURE_DEFAULT TOP_P_DEFAULT=$TOP_P_DEFAULT MAX_NEW_TOKENS_DEFAULT=$MAX_NEW_TOKENS_DEFAULT SEED_DEFAULT=$SEED_DEFAULT DO_SAMPLE_DEFAULT=$DO_SAMPLE_DEFAULT"
echo "[entrypoint] PROGRESS_LOGS=$PROGRESS_LOGS"
echo "[entrypoint] DEBUG=$DEBUG"
echo "[entrypoint] REDACT_SOURCE=$REDACT_SOURCE"
echo "[entrypoint] FAST_USAGE=$FAST_USAGE"
if [[ "${REDACT_SOURCE_ENV}" == "1" ]]; then
  echo "[entrypoint] CACHE_DIR=redacted"
else
  echo "[entrypoint] CACHE_DIR=${CACHE_DIR_ENV}"
fi
if [[ -n "${GPU_IDS:-}" ]]; then
  echo "[entrypoint] Using explicit GPU_IDS=${GPU_IDS}"
elif [[ -n "${GPU_COUNT:-}" ]]; then
  echo "[entrypoint] Using GPU_COUNT=${GPU_COUNT}"
else
  echo "[entrypoint] No GPU args passed; using all visible GPUs."
fi
# If DEBUG is enabled, default uvicorn log level to debug unless explicitly set otherwise
if [[ "${DEBUG_ENV}" == "1" ]]; then
  case "${LOG_LEVEL}" in
    info|INFO|Info|"") LOG_LEVEL="debug" ;;
  esac
fi
# Optional runtime override of transformers source
if [[ -n "${TRANSFORMERS_INSTALL_PATH:-}" ]]; then
  echo "[entrypoint] Installing transformers from ${TRANSFORMERS_INSTALL_PATH} ..."
  python3 -m pip install --no-cache-dir -U "${TRANSFORMERS_INSTALL_PATH}"
  python3 - <<'PY'
import importlib, sys
try:
    t = importlib.import_module("transformers")
    print(f"[entrypoint] transformers {t.__version__} from {t.__file__}")
except Exception as e:
    print(f"[entrypoint] WARNING: unable to import transformers after install: {e}", file=sys.stderr)
PY
fi

# Always show which transformers is in use
python3 - <<'PY'
import importlib, sys
try:
    t = importlib.import_module("transformers")
    print(f"[entrypoint] Using transformers {t.__version__} from {t.__file__}")
except Exception as e:
    print(f"[entrypoint] WARNING: transformers not importable: {e}", file=sys.stderr)
PY

echo "[entrypoint] Starting uvicorn on ${HOST}:${PORT}..."

# Single worker to avoid duplicate model loads; asyncio & batching handle concurrency
exec uvicorn server:app --host "${HOST}" --port "${PORT}" --workers 1 --log-level "${LOG_LEVEL}" "${EXTRA_ARGS[@]}"
