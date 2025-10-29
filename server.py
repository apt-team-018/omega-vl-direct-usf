import os
import time
import uuid
import asyncio
import logging
import io
import base64
import aiohttp
import json
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import List, Optional, Dict, Any, Tuple, Union, AsyncGenerator
from dataclasses import dataclass

import torch
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import JSONResponse, ORJSONResponse, StreamingResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from transformers import AutoProcessor, Omega17VLExpForConditionalGeneration, TextIteratorStreamer
from transformers.utils import logging as tf_logging
from PIL import Image
from io import BytesIO
from threading import Thread
import importlib as _importlib  # added to print transformers origin
try:
    _t = _importlib.import_module("transformers")
    print(f"[server] transformers {_t.__version__} from {_t.__file__}", flush=True)
except Exception:
    pass

# -------------------------
# Global speed/accuracy envs (safe defaults)
# -------------------------
# Faster model downloads (optional): pip install hf_transfer
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# Better GPU allocator behavior for long-running generation servers
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Tokenizers spawn many threads by default; keep this low for server workloads
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Flash/SDPA kernels benefit from this; recommended by FA2 docs
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")

# -------------------------
# Config (env overrides)
# -------------------------
MODEL_PATH = os.getenv("MODEL_PATH") or os.getenv("MODEL_ID", "5techlab-research/test_iter3")
MODEL_REVISION = os.getenv("MODEL_REVISION")
DTYPE_STR = os.getenv("DTYPE", "bf16").lower()
if DTYPE_STR in {"bf16", "bfloat16"}:
    DTYPE = torch.bfloat16
elif DTYPE_STR in {"fp16", "float16", "half"}:
    DTYPE = torch.float16
elif DTYPE_STR in {"fp32", "float32"}:
    DTYPE = torch.float32
else:
    DTYPE = torch.bfloat16
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "1").lower() in {"1", "true", "yes"}

# Attention kernels: try FlashAttention 2, fallback to SDPA if not available
ATTN_IMPL = os.getenv("ATTN_IMPL", "flash_attention_2")  # options: "flash_attention_2", "sdpa"

# VLM-specific configuration
MAX_MODEL_LENGTH = int(os.getenv("MAX_MODEL_LENGTH", "8192"))  # Max sequence length for model
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))  # Max image dimension
MAX_IMAGES_PER_REQUEST = int(os.getenv("MAX_IMAGES_PER_REQUEST", "10"))  # Total images across all messages
IMAGE_CACHE_SIZE = int(os.getenv("IMAGE_CACHE_SIZE", "256"))
ALLOW_REMOTE_IMAGES = os.getenv("ALLOW_REMOTE_IMAGES", "1").lower() in {"1", "true", "yes"}

# Supported image formats (common formats only)
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "GIF", "WEBP"}

# Micro-batching knobs (VLM-optimized defaults)
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "2"))  # Lower for VLM
BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", "10"))  # Allow time for image loading
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", os.getenv("MAX_CONTEXT_TOKENS", "8192")))  # hard clamp
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS_DEFAULT", "512"))  # Higher for VLM
# Backpressure: cap queued requests per worker before 503
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "128"))  # Lower for VLM
API_KEY = os.getenv("API_KEY", "EMPTY")
ATTN_STRICT = os.getenv("ATTN_STRICT", "false").lower() in {"1", "true", "yes", "on"}

# Generation defaults (latency-friendly, accuracy-preserving)
DO_SAMPLE_DEFAULT = bool(int(os.getenv("DO_SAMPLE_DEFAULT", "1")))  # 0 -> False, 1 -> True
TEMPERATURE_DEFAULT = float(os.getenv("TEMPERATURE_DEFAULT", "0.8"))
TOP_P_DEFAULT = float(os.getenv("TOP_P_DEFAULT", "0.9"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "50"))
REPETITION_PENALTY_DEFAULT = float(os.getenv("REPETITION_PENALTY_DEFAULT", "1.0"))
SEED_DEFAULT = int(os.getenv("SEED_DEFAULT", "42"))
# Display model name for API contract and docs (user-configurable)
MODEL_NAME = os.getenv("MODEL_NAME", "model")
# Server bind info (for startup log)
HOST_BIND = os.getenv("HOST", "0.0.0.0")
PORT_BIND = os.getenv("PORT", "8000")
# Progress logs toggle (1/true/on to enable)
PROGRESS_LOGS = (os.getenv("PROGRESS_LOGS", "0").lower() in {"1", "true", "yes", "on"})
DEBUG_MODE = (os.getenv("DEBUG", "0").lower() in {"1", "true", "yes", "on"})
REDACT_SOURCE = (os.getenv("REDACT_SOURCE", "1").lower() in {"1", "true", "yes", "on"})
FAST_USAGE = (os.getenv("FAST_USAGE", "1").lower() in {"1", "true", "yes", "on"})
REQUIRE_GPU = (os.getenv("REQUIRE_GPU", "1").lower() in {"1", "true", "yes", "on"})

def logp(msg: str):
    if PROGRESS_LOGS:
        print(msg, flush=True)

# Configure logging and library verbosity
try:
    if REDACT_SOURCE:
        # Silence HF progress bars and telemetry
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = os.getenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = os.getenv("HF_HUB_DISABLE_TELEMETRY", "1")
        # Keep upstream libraries quiet even if DEBUG is on
        tf_logging.set_verbosity_error()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("urllib3").setLevel(logging.ERROR)
        logging.getLogger("filelock").setLevel(logging.ERROR)
        # Application logs can still be DEBUG if requested
        if DEBUG_MODE:
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger("uvicorn").setLevel(logging.DEBUG)
            logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
            logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
    else:
        if DEBUG_MODE:
            logging.basicConfig(level=logging.DEBUG)
            tf_logging.set_verbosity_debug()
            logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)
            logging.getLogger("uvicorn").setLevel(logging.DEBUG)
            logging.getLogger("uvicorn.error").setLevel(logging.DEBUG)
            logging.getLogger("uvicorn.access").setLevel(logging.DEBUG)
        elif PROGRESS_LOGS:
            tf_logging.set_verbosity_info()
            logging.getLogger("huggingface_hub").setLevel(logging.INFO)
except Exception:
    pass

@contextmanager
def _silence_hf_io():
    """
    Temporarily silence stdout/stderr prints and set HF/transformers related
    loggers to ERROR to hide download progress and source details.
    """
    lvl_tr = logging.getLogger("transformers").level
    lvl_hf = logging.getLogger("huggingface_hub").level
    lvl_u3 = logging.getLogger("urllib3").level
    lvl_fl = logging.getLogger("filelock").level
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("filelock").setLevel(logging.ERROR)
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            yield
    finally:
        logging.getLogger("transformers").setLevel(lvl_tr)
        logging.getLogger("huggingface_hub").setLevel(lvl_hf)
        logging.getLogger("urllib3").setLevel(lvl_u3)
        logging.getLogger("filelock").setLevel(lvl_fl)

# Torch speedups: allow TF32 fallback for fp32 ops (does not change BF16 compute)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")  # TF32 when relevant ops fall back to fp32
torch.backends.cuda.matmul.allow_tf32 = True

# -------------------------
# GPU selection
# -------------------------
# Priority:
# 1) GPU_IDS (comma-separated), else
# 2) GPU_COUNT (use first N visible), else
# 3) All visible GPUs if >= 1
_gpu_ids_env = os.getenv("GPU_IDS")
_gpu_count_env = os.getenv("GPU_COUNT")

if torch.cuda.is_available():
    visible = torch.cuda.device_count()
else:
    visible = 0

if _gpu_ids_env:
    GPU_IDS = [int(x.strip()) for x in _gpu_ids_env.split(",") if x.strip() != ""]
elif _gpu_count_env:
    n = max(0, int(_gpu_count_env))
    n = min(n, visible) if visible > 0 else 0
    GPU_IDS = list(range(n))
else:
    # Default to single GPU if available (developer-friendly default)
    GPU_IDS = list(range(min(visible, 1))) if visible > 0 else []

# Enforce GPU requirement if configured
if REQUIRE_GPU:
    if not torch.cuda.is_available() or len(GPU_IDS) == 0:
        raise RuntimeError("CUDA GPU is required but not available; set REQUIRE_GPU=0 to allow CPU fallback")

# If no CUDA GPUs are visible, fall back to a single CPU worker to keep the server available
CPU_FALLBACK = False
if not GPU_IDS:
    CPU_FALLBACK = True
# When running on CPU, ensure safe defaults (fp32 compute, sdpa attention)
if CPU_FALLBACK:
    DTYPE = torch.float32
    if ATTN_IMPL != "sdpa":
        ATTN_IMPL = "sdpa"

# -------------------------
# OpenAI-compatible Chat API models (VLM-enhanced)
# -------------------------
class ContentPart(BaseModel):
    type: str = Field(..., description="Content type: 'text', 'image', or 'image_url'")
    text: Optional[str] = Field(None, description="Text content when type='text'")
    image: Optional[str] = Field(None, description="Image URL or base64 when type='image' (deprecated, use image_url)")
    image_url: Optional[Dict[str, str]] = Field(None, description="OpenAI format: {'url': '...'} when type='image_url'")


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentPart]]  # Support both text-only and multimodal


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    
    # Sampling parameters
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0)
    max_tokens: Optional[int] = Field(None, ge=1)
    min_tokens: Optional[int] = Field(None, ge=0)
    
    # Advanced parameters
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0)
    length_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    
    # Stop conditions
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    
    # Control
    seed: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    
    # VLM-specific
    max_image_size: Optional[int] = Field(None, description="Override default max image size")



# -------------------------
# Image Loading Utilities
# -------------------------
def validate_image_format(image: Image.Image, source: str = "image"):
    """Validate image format is in supported list."""
    image_format = image.format
    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format: {image_format}. Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
        )


async def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL asynchronously and validate format."""
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.read()
                image = Image.open(BytesIO(data))
                
                # Validate format before conversion
                validate_image_format(image, url)
                
                # Convert to RGB
                image = image.convert('RGB')
                
                # Resize if needed
                if MAX_IMAGE_SIZE > 0:
                    width, height = image.size
                    if max(width, height) > MAX_IMAGE_SIZE:
                        ratio = MAX_IMAGE_SIZE / max(width, height)
                        new_size = (int(width * ratio), int(height * ratio))
                        image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                return image
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {str(e)}")


def load_image_from_base64(data: str) -> Image.Image:
    """Load image from base64 string and validate format."""
    try:
        # Handle data URLs
        if data.startswith('data:image'):
            # Extract base64 part after comma
            if ',' in data:
                data = data.split(',', 1)[1]
        
        # Decode base64
        image_data = base64.b64decode(data)
        image = Image.open(BytesIO(image_data))
        
        # Validate format before conversion
        validate_image_format(image, "base64 image")
        
        # Convert to RGB
        image = image.convert('RGB')
        
        # Resize if needed
        if MAX_IMAGE_SIZE > 0:
            width, height = image.size
            if max(width, height) > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(width, height)
                new_size = (int(width * ratio), int(height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {str(e)}")


async def load_image(image_source: str) -> Image.Image:
    """Load image from URL or base64 with format validation."""
    if image_source.startswith('http://') or image_source.startswith('https://'):
        if not ALLOW_REMOTE_IMAGES:
            raise HTTPException(status_code=400, detail="Remote images are disabled")
        return await load_image_from_url(image_source)
    else:
        # Assume base64
        return load_image_from_base64(image_source)


# -------------------------
# Chat template helpers (VLM-enhanced)
# -------------------------
async def prepare_vlm_messages(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-style messages to VLM format with loaded images.
    
    IMPORTANT: ALL messages are converted to structured format [{"type": "text", ...}]
    even if the input is a simple string. This ensures consistency with VLM processor.
    
    Image counting is across ALL messages in the conversation (not per-message),
    matching OpenAI format behavior where MAX_IMAGES_PER_REQUEST is a total limit.
    """
    vlm_messages = []
    total_image_count = 0  # Track images across all messages
    
    for msg in messages:
        role = msg.role
        content = msg.content
        
        # ALWAYS convert to structured format [{"type": "text", ...}]
        if isinstance(content, str):
            # Simple string input -> convert to structured format
            vlm_messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}]
            })
        else:
            # Already structured or multimodal message
            content_parts = []
            
            for part in content:
                if isinstance(part, ContentPart):
                    if part.type == "text" and part.text:
                        content_parts.append({"type": "text", "text": part.text})
                    elif part.type == "image_url":
                        # OpenAI-compatible format: {"type": "image_url", "image_url": {"url": "..."}}
                        image_source = None
                        if part.image_url and isinstance(part.image_url, dict):
                            image_source = part.image_url.get("url")
                        
                        if image_source:
                            total_image_count += 1
                            if total_image_count > MAX_IMAGES_PER_REQUEST:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Maximum {MAX_IMAGES_PER_REQUEST} images allowed across all messages in request"
                                )
                            
                            # Load and validate image
                            image = await load_image(image_source)
                            content_parts.append({"type": "image", "image": image})
                    elif part.type == "image":
                        # Legacy format: {"type": "image", "image": "..."} or {"type": "image", "image_url": {"url": "..."}}
                        image_source = None
                        if part.image:
                            image_source = part.image
                        elif part.image_url and isinstance(part.image_url, dict):
                            image_source = part.image_url.get("url")
                        
                        if image_source:
                            total_image_count += 1
                            if total_image_count > MAX_IMAGES_PER_REQUEST:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Maximum {MAX_IMAGES_PER_REQUEST} images allowed across all messages in request"
                                )
                            
                            # Load and validate image
                            image = await load_image(image_source)
                            content_parts.append({"type": "image", "image": image})
                elif isinstance(part, dict):
                    # Handle raw dict format
                    if part.get("type") == "text":
                        text_value = part.get("text", "")
                        if text_value:  # Only add non-empty text
                            content_parts.append({"type": "text", "text": text_value})
                    elif part.get("type") == "image_url":
                        # OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
                        image_url_dict = part.get("image_url")
                        if isinstance(image_url_dict, dict):
                            image_source = image_url_dict.get("url")
                            if image_source:
                                total_image_count += 1
                                if total_image_count > MAX_IMAGES_PER_REQUEST:
                                    raise HTTPException(
                                        status_code=400,
                                        detail=f"Maximum {MAX_IMAGES_PER_REQUEST} images allowed across all messages in request"
                                    )
                                # Load and validate image
                                image = await load_image(image_source)
                                content_parts.append({"type": "image", "image": image})
                    elif part.get("type") == "image":
                        # Legacy format
                        image_source = part.get("image") or (part.get("image_url", {}).get("url") if isinstance(part.get("image_url"), dict) else None)
                        if image_source:
                            total_image_count += 1
                            if total_image_count > MAX_IMAGES_PER_REQUEST:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Maximum {MAX_IMAGES_PER_REQUEST} images allowed across all messages in request"
                                )
                            # Load and validate image
                            image = await load_image(image_source)
                            content_parts.append({"type": "image", "image": image})
            
            # Always use structured format, even if empty (shouldn't happen but defensive)
            if not content_parts:
                content_parts = [{"type": "text", "text": ""}]
            
            vlm_messages.append({
                "role": role,
                "content": content_parts
            })
    
    return vlm_messages


def _extract_text_from_content(content: Any) -> str:
    """Extract text from content (fallback for text-only paths)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("type")
                if t == "text" and "text" in part:
                    parts.append(str(part["text"]))
                elif "content" in part and isinstance(part["content"], str):
                    parts.append(part["content"])
            elif isinstance(part, ContentPart) and part.type == "text" and part.text:
                parts.append(part.text)
        return "".join(parts)
    return str(content)
# -------------------------
# Request envelope for batching (VLM-enhanced)
# -------------------------
@dataclass
class _PendingReq:
    messages: List[Dict[str, Any]]  # VLM messages with loaded images
    params: Dict[str, Any]
    future: asyncio.Future
    is_stream: bool = False


# -------------------------
# Worker (one per GPU)
# -------------------------
class GPUWorker:
    def __init__(self, device: str, model_id: str):
        self.device = device
        self.model_id = model_id
        self.queue: asyncio.Queue[_PendingReq] = asyncio.Queue()
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[Omega17VLExpForConditionalGeneration] = None
        self.ready = asyncio.Event()
        self.batch_task: Optional[asyncio.Task] = None

    async def start(self):
        # Ensure correct CUDA device context
        if str(self.device).startswith("cuda:"):
            try:
                idx = int(str(self.device).split(":")[1])
                if torch.cuda.is_available():
                    torch.cuda.set_device(idx)
                    try:
                        torch.set_default_device(self.device)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Load processor (combines tokenizer + image processor)
        rev_kw = {"revision": MODEL_REVISION} if MODEL_REVISION else {}
        logp(f"[startup][{self.device}] Loading processor…")
        if REDACT_SOURCE:
            with _silence_hf_io():
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    use_fast=True,
                    token=HF_TOKEN,
                    trust_remote_code=TRUST_REMOTE_CODE,
                    **rev_kw,
                )
        else:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                use_fast=True,
                token=HF_TOKEN,
                trust_remote_code=TRUST_REMOTE_CODE,
                **rev_kw,
            )
        
        logp(f"[startup][{self.device}] Processor ready")

        # Try preferred attention impl; optionally enforce strict (no fallback)
        attn_impl = ATTN_IMPL
        logp(f"[startup][{self.device}] Loading VLM model (attn={attn_impl}, dtype={str(DTYPE).replace('torch.','')}, max_len={MAX_MODEL_LENGTH})…")
        try:
            if REDACT_SOURCE:
                with _silence_hf_io():
                    self.model = Omega17VLExpForConditionalGeneration.from_pretrained(
                        self.model_id,
                        dtype=DTYPE,
                        device_map="cuda",
                        attn_implementation=attn_impl,
                        low_cpu_mem_usage=True,
                        token=HF_TOKEN,
                        trust_remote_code=TRUST_REMOTE_CODE,
                        **rev_kw,
                    )
            else:
                self.model = Omega17VLExpForConditionalGeneration.from_pretrained(
                    self.model_id,
                    dtype=DTYPE,
                    device_map="cuda",
                    attn_implementation=attn_impl,
                    low_cpu_mem_usage=True,
                    token=HF_TOKEN,
                    trust_remote_code=TRUST_REMOTE_CODE,
                    **rev_kw,
                )
        except Exception as e:
            # If strict mode enabled, or chosen impl already 'sdpa', re-raise
            if ATTN_STRICT or attn_impl == "sdpa":
                raise
            # Otherwise, fallback to SDPA
            logp(f"[startup][{self.device}] Falling back to sdpa attention")
            if REDACT_SOURCE:
                with _silence_hf_io():
                    self.model = Omega17VLExpForConditionalGeneration.from_pretrained(
                        self.model_id,
                        dtype=DTYPE,
                        device_map="cuda",
                        attn_implementation="sdpa",
                        low_cpu_mem_usage=True,
                        token=HF_TOKEN,
                        trust_remote_code=TRUST_REMOTE_CODE,
                        **rev_kw,
                    )
            else:
                self.model = Omega17VLExpForConditionalGeneration.from_pretrained(
                    self.model_id,
                    dtype=DTYPE,
                    device_map="cuda",
                    attn_implementation="sdpa",
                    low_cpu_mem_usage=True,
                    token=HF_TOKEN,
                    trust_remote_code=TRUST_REMOTE_CODE,
                    **rev_kw,
                )

        # Explicitly ensure model is on the target device
        if str(self.device).startswith("cuda:"):
            try:
                idx = int(str(self.device).split(":")[1])
                self.model = self.model.cuda(idx)
            except Exception:
                pass

        # Generation config hygiene
        self.model.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.generation_config.eos_token_id = self.processor.tokenizer.eos_token_id
        self.model.generation_config.use_cache = True
        self.model.generation_config.max_length = MAX_MODEL_LENGTH
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if hasattr(self.model.config, "max_position_embeddings"):
            self.model.config.max_position_embeddings = MAX_MODEL_LENGTH

        # Optional compilation (reduces Python overhead during decoding)
        compiled = False
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
            compiled = True
        except Exception:
            # Continue without compile if not supported
            compiled = False
        logp(f"[startup][{self.device}] torch.compile={'on' if compiled else 'off'}")

        # Warmup (build CUDA graphs, caches)
        logp(f"[startup][{self.device}] Warmup…")
        self._warmup()
        logp(f"[startup][{self.device}] Ready.")

        # Start micro-batching loop
        self.batch_task = asyncio.create_task(self._batch_loop())
        self.ready.set()

    def _warmup(self):
        """Warmup with VLM-style messages (text-only for simplicity)."""
        warmup_messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, test warmup."}]
                }
            ]
        ]
        
        with torch.inference_mode():
            try:
                inputs = self.processor.apply_chat_template(
                    warmup_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)
                
                # Short decode to warm up kernels and cache
                _ = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            except Exception as e:
                logp(f"[startup][{self.device}] Warmup warning: {e}")

    async def _batch_loop(self):
        """
        Collect requests for a few milliseconds and run as a single batch.
        """
        while True:
            try:
                first: _PendingReq = await self.queue.get()
                batch: List[_PendingReq] = [first]
                t0 = time.time()

                # For streaming requests, process immediately without batching
                if first.is_stream:
                    await self._run_streaming_batch([first])
                    continue

                # Collect up to MAX_BATCH_SIZE or BATCH_TIMEOUT_MS for non-streaming
                while len(batch) < MAX_BATCH_SIZE:
                    timeout_left = (BATCH_TIMEOUT_MS / 1000.0) - (time.time() - t0)
                    if timeout_left <= 0:
                        break
                    try:
                        nxt = await asyncio.wait_for(self.queue.get(), timeout=timeout_left)
                        # Don't mix streaming and non-streaming in the same batch
                        if nxt.is_stream:
                            # Put it back for the next iteration
                            await self.queue.put(nxt)
                            break
                        batch.append(nxt)
                    except asyncio.TimeoutError:
                        break

                # Run batch
                await self._run_batch(batch)
            except Exception as e:
                if DEBUG_MODE:
                    logging.exception("Batch loop error")
                # Fail-fast all waiting futures in case of unexpected errors
                for req in batch if "batch" in locals() else []:
                    if not req.future.done():
                        req.future.set_exception(e)

    def _sort_by_length(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        # Sort by true lengths (sum of attention_mask)
        lengths = attention_mask.sum(dim=1)
        sorted_idx = torch.argsort(lengths, descending=True)
        inv_idx = torch.empty_like(sorted_idx)
        inv_idx[sorted_idx] = torch.arange(sorted_idx.size(0), device=sorted_idx.device)
        return (
            input_ids.index_select(0, sorted_idx),
            attention_mask.index_select(0, sorted_idx),
            inv_idx.tolist(),
        )

    async def _run_batch(self, batch: List[_PendingReq]):
        """Process VLM batch with images and text."""
        assert self.model is not None and self.processor is not None

        messages_list = [r.messages for r in batch]
        params_list = [r.params for r in batch]

        # Extract generation parameters (use max across batch for conservative approach)
        max_new_tokens = min(
            max([int(p.get("max_new_tokens", MAX_NEW_TOKENS_DEFAULT)) for p in params_list]),
            MAX_NEW_TOKENS_DEFAULT,
        )
        
        min_new_tokens = max([int(p.get("min_new_tokens", 1)) for p in params_list])
        
        do_sample = any(bool(p.get("do_sample", DO_SAMPLE_DEFAULT)) for p in params_list)
        
        # Sampling parameters
        if do_sample:
            temperature = max([float(p.get("temperature", TEMPERATURE_DEFAULT)) for p in params_list])
            temperature = max(0.01, min(temperature, 2.0))
            
            top_p = max([float(p.get("top_p", TOP_P_DEFAULT)) for p in params_list])
            top_p = max(0.0, min(top_p, 1.0))
            
            top_k = max([int(p.get("top_k", TOP_K_DEFAULT)) for p in params_list])
            top_k = max(0, top_k)
            
            repetition_penalty = max([float(p.get("repetition_penalty", REPETITION_PENALTY_DEFAULT)) for p in params_list])
            repetition_penalty = max(1.0, min(repetition_penalty, 2.0))
            
            length_penalty = params_list[0].get("length_penalty", 1.0)
            length_penalty = max(-2.0, min(length_penalty, 2.0))
        else:
            temperature = 0.0
            top_p = 1.0
            top_k = 0
            repetition_penalty = 1.0
            length_penalty = 1.0

        # Apply chat template with processor (handles images automatically)
        try:
            inputs = self.processor.apply_chat_template(
                messages_list,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(
                        HTTPException(status_code=500, detail=f"Failed to process VLM inputs: {e}")
                    )
            return

        # Get input lengths for token counting
        lengths = inputs["attention_mask"].sum(dim=1) if "attention_mask" in inputs else torch.tensor([inputs["input_ids"].size(1)] * len(batch))

        # Seed for reproducibility
        batch_seed = int(params_list[0].get("seed", SEED_DEFAULT)) if len(params_list) > 0 else SEED_DEFAULT
        if do_sample:
            try:
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(batch_seed)
            except Exception:
                pass

        # Generate with VLM model
        with torch.inference_mode():
            try:
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "min_new_tokens": min_new_tokens,
                    "do_sample": do_sample,
                    "use_cache": True,
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                    "eos_token_id": self.processor.tokenizer.eos_token_id,
                }
                
                if do_sample:
                    gen_kwargs.update({
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "repetition_penalty": repetition_penalty,
                        "length_penalty": length_penalty,
                    })
                
                outputs = self.model.generate(**gen_kwargs)
            except Exception as e:
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(
                            HTTPException(status_code=500, detail=f"Generation failed: {e}")
                        )
                return

        # Decode outputs (trim prompt tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        
        decoded_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Set results for each request
        for i, req in enumerate(batch):
            text = decoded_texts[i]
            
            # Apply stop strings
            stops = req.params.get("stop") or []
            if isinstance(stops, str):
                stops = [stops]
            elif not isinstance(stops, list):
                stops = []
            
            cut_idx = None
            for s in stops:
                if not s:
                    continue
                j = text.find(s)
                if j != -1:
                    cut_idx = j if cut_idx is None else min(cut_idx, j)
            if cut_idx is not None:
                text = text[:cut_idx]

            # Token counts
            prompt_tok_count = int(lengths[i].item()) if i < len(lengths) else 0
            completion_tok_count = len(generated_ids_trimmed[i])

            if not req.future.done():
                req.future.set_result({
                    "text": text,
                    "prompt_tokens": prompt_tok_count,
                    "completion_tokens": completion_tok_count,
                })


    async def _run_streaming_batch(self, batch: List[_PendingReq]):
        """Process VLM batch with streaming support."""
        assert self.model is not None and self.processor is not None

        # Only support batch size of 1 for streaming
        if len(batch) != 1:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(
                        HTTPException(status_code=400, detail="Streaming only supports batch size of 1")
                    )
            return

        req = batch[0]
        messages_list = [req.messages]
        params = req.params

        # Extract generation parameters
        max_new_tokens = min(int(params.get("max_new_tokens", MAX_NEW_TOKENS_DEFAULT)), MAX_NEW_TOKENS_DEFAULT)
        min_new_tokens = int(params.get("min_new_tokens", 1))
        do_sample = bool(params.get("do_sample", DO_SAMPLE_DEFAULT))
        
        # Sampling parameters
        if do_sample:
            temperature = float(params.get("temperature", TEMPERATURE_DEFAULT))
            temperature = max(0.01, min(temperature, 2.0))
            top_p = float(params.get("top_p", TOP_P_DEFAULT))
            top_p = max(0.0, min(top_p, 1.0))
            top_k = int(params.get("top_k", TOP_K_DEFAULT))
            top_k = max(0, top_k)
            repetition_penalty = float(params.get("repetition_penalty", REPETITION_PENALTY_DEFAULT))
            repetition_penalty = max(1.0, min(repetition_penalty, 2.0))
            length_penalty = params.get("length_penalty", 1.0)
            length_penalty = max(-2.0, min(length_penalty, 2.0))
        else:
            temperature = 0.0
            top_p = 1.0
            top_k = 0
            repetition_penalty = 1.0
            length_penalty = 1.0

        # Apply chat template
        try:
            inputs = self.processor.apply_chat_template(
                messages_list,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)
        except Exception as e:
            if not req.future.done():
                req.future.set_exception(
                    HTTPException(status_code=500, detail=f"Failed to process VLM inputs: {e}")
                )
            return

        # Get prompt length for token counting
        prompt_length = inputs["input_ids"].shape[1]
        
        # Seed for reproducibility
        batch_seed = int(params.get("seed", SEED_DEFAULT))
        if do_sample:
            try:
                torch.manual_seed(batch_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(batch_seed)
            except Exception:
                pass

        # Create streamer
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Prepare generation kwargs
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        if do_sample:
            gen_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
            })

        # Start generation in a separate thread
        def generate_fn():
            with torch.inference_mode():
                try:
                    self.model.generate(**gen_kwargs)
                except Exception as e:
                    # Error will be caught by streamer iteration
                    pass

        generation_thread = Thread(target=generate_fn)
        generation_thread.start()

        # Set the streamer and metadata as result
        if not req.future.done():
            req.future.set_result({
                "streamer": streamer,
                "prompt_tokens": prompt_length,
                "generation_thread": generation_thread,
                "stop_strings": params.get("stop", []),
            })


# -------------------------
# API key dependency
# -------------------------
def require_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    # If API_KEY is 'EMPTY' (default), auth is disabled
    if API_KEY == "EMPTY":
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="missing authorization")
    auth_lower = authorization.lower()
    if not auth_lower.startswith("bearer "):
        raise HTTPException(status_code=401, detail="invalid authorization scheme")
    token = authorization.split(" ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

# -------------------------
# Router that picks a GPU
# -------------------------
class Router:
    def __init__(self, workers: List[GPUWorker]):
        self.workers = workers

    async def start(self):
        await asyncio.gather(*(w.start() for w in self.workers))

    def _pick_worker(self) -> GPUWorker:
        # Simple least-queue heuristic (effective for bursty loads)
        return min(self.workers, key=lambda w: w.queue.qsize())

    def is_overloaded(self) -> bool:
        # Backpressure: if all worker queues are at or above the cap, report overload
        try:
            return all(w.queue.qsize() >= MAX_QUEUE_SIZE for w in self.workers)
        except Exception:
            return False

    async def generate_vlm(self, messages: List[Dict[str, Any]], params: Dict[str, Any], is_stream: bool = False):
        """Generate response for VLM messages."""
        if self.is_overloaded():
            raise HTTPException(status_code=503, detail="server overloaded: queue full")
        worker = self._pick_worker()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        req = _PendingReq(messages=messages, params=params, future=fut, is_stream=is_stream)
        await worker.queue.put(req)
        return await fut


# -------------------------
# App initialization
# -------------------------
APP_TITLE = f"{MODEL_NAME} - Docomentsio"
app = FastAPI(
    title=APP_TITLE,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    default_response_class=ORJSONResponse,
)

# Inject Bearer auth into OpenAPI so Swagger shows the Authorize button
def custom_openapi():
    if getattr(app, "openapi_schema", None):
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version="1.0.0",
        routes=app.routes,
        description=None,
    )
    components = schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }
    # Set a global security requirement (UI only; enforcement happens via dependency)
    schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/")
async def root():
    # Unprotected welcome endpoint with deployment info
    return JSONResponse(
        {
            "message": "Omega LLM API is running",
            "model_name": MODEL_NAME,
            "app_title": APP_TITLE,
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json",
                "models": "/v1/models",
                "chat_completions": "/v1/chat/completions",
            },
            "auth": {
                "protected": API_KEY != "EMPTY",
                "scheme": "Authorization: Bearer",
            },
            "notes": "Docs and OpenAPI may require a Bearer token if protection is enabled.",
        }
    )

# Track startup status to avoid crashing the process on model load failures
STARTUP_OK: bool = False
STARTUP_ERROR: Optional[str] = None

@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    # Ensure the server never crashes due to uncaught errors
    if 'DEBUG_MODE' in globals() and DEBUG_MODE:
        logging.exception("Unhandled exception while processing request %s", getattr(request, "url", ""))
    return JSONResponse(status_code=500, content={"detail": "internal error", "error": str(exc)})

# Create workers: one per CUDA device when available, else a single CPU worker
if not CPU_FALLBACK and GPU_IDS:
    workers = [GPUWorker(device=f"cuda:{gid}", model_id=MODEL_PATH) for gid in GPU_IDS]
else:
    workers = [GPUWorker(device="cpu", model_id=MODEL_PATH)]
router = Router(workers=workers)


@app.on_event("startup")
async def _startup():
    global STARTUP_OK, STARTUP_ERROR
    async def _heartbeat():
        if not PROGRESS_LOGS:
            return
        t0 = time.time()
        dots = 0
        try:
            while not STARTUP_OK:
                print(".", end="", flush=True)
                dots += 1
                if dots % 10 == 0:
                    elapsed = int(time.time() - t0)
                    print(f"\n[startup] loading… {elapsed}s", flush=True)
                if dots % 20 == 0:
                    print("[startup] still loading model & weights…", flush=True)
                    if int(time.time() - t0) > 30:
                        print("[startup] tip: mount a local model path with --model-path to avoid first-run downloads", flush=True)
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    devices_plan = [f"cuda:{gid}" for gid in GPU_IDS] if (not CPU_FALLBACK and GPU_IDS) else ["cpu"]
    cache_dir = (
        os.getenv("HF_HUB_CACHE")
        or os.getenv("HF_HOME")
        or "/root/.cache/usinc/models"
    )
    logp("[startup] Initializing workers…")
    if REDACT_SOURCE:
        logp(f"[startup] loading model (attn={ATTN_IMPL} dtype={str(DTYPE).replace('torch.','')} gpus={devices_plan})…")
    else:
        logp(f"[startup] model={MODEL_PATH} rev={MODEL_REVISION} attn={ATTN_IMPL} dtype={str(DTYPE).replace('torch.','')} gpus={devices_plan} cache_dir={cache_dir}")
    logp("[startup] Downloading model weights (first run may take minutes)…")
    hb_task: Optional[asyncio.Task] = asyncio.create_task(_heartbeat()) if PROGRESS_LOGS else None
    try:
        await router.start()
        STARTUP_OK = True
        STARTUP_ERROR = None
        if hb_task:
            hb_task.cancel()
        logp(f"[server] Ready. Health: http://{HOST_BIND}:{PORT_BIND}/health")
    except Exception as e:
        # Keep the server up; report not ready via /health and 503s on generation endpoints
        STARTUP_OK = False
        STARTUP_ERROR = str(e)
        if hb_task:
            hb_task.cancel()
        if DEBUG_MODE:
            logging.exception("Startup failed")
        logp(f"[server] Not ready (degraded). Health: http://{HOST_BIND}:{PORT_BIND}/health Error: {e}")


@app.get("/health")
async def health():
    # Return per-worker queue depth and readiness
    device_names: List[str] = []
    for w in workers:
        d = str(w.device)
        if d.startswith("cuda:") and torch.cuda.is_available():
            try:
                idx = int(d.split(":")[1])
                device_names.append(torch.cuda.get_device_name(idx))
            except Exception:
                device_names.append("cuda")
        elif d == "cpu":
            device_names.append("cpu")
        else:
            device_names.append(d)
    visible_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return {
        "model_path": (None if REDACT_SOURCE else MODEL_PATH),
        "model_revision": (None if REDACT_SOURCE else MODEL_REVISION),
        "devices": [w.device for w in workers],
        "device_names": device_names,
        "visible_cuda_devices": visible_cuda,
        "queues": [w.queue.qsize() for w in workers],
        "ready": all(w.ready.is_set() for w in workers),
        "startup_ok": STARTUP_OK,
        "startup_error": STARTUP_ERROR,
        "overloaded": router.is_overloaded(),
        "max_queue_size": MAX_QUEUE_SIZE,
        "dtype": str(DTYPE),
        "attn": ATTN_IMPL,
        "batch": {"max_batch_size": MAX_BATCH_SIZE, "batch_timeout_ms": BATCH_TIMEOUT_MS},
        "vlm_config": {
            "max_model_length": MAX_MODEL_LENGTH,
            "max_image_size": MAX_IMAGE_SIZE,
            "max_images_per_request": MAX_IMAGES_PER_REQUEST,
            "allow_remote_images": ALLOW_REMOTE_IMAGES,
            "supported_image_formats": list(SUPPORTED_IMAGE_FORMATS),
        },
    }


@app.get("/openapi.json", dependencies=[Depends(require_api_key)])
async def openapi_json():
    # Use FastAPI's cached generator
    return ORJSONResponse(app.openapi())

@app.get("/docs", dependencies=[Depends(require_api_key)])
async def custom_swagger_ui_html():
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger UI")

@app.get("/redoc", dependencies=[Depends(require_api_key)], include_in_schema=False)
async def custom_redoc_html():
    return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")

@app.get("/v1/models", dependencies=[Depends(require_api_key)])
async def list_models():
    # Expose the configured display model name
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model"}],
    }


async def stream_response(
    streamer: TextIteratorStreamer,
    generation_thread: Thread,
    stop_strings: List[str],
    prompt_tokens: int,
    request_id: str,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """Generate SSE-formatted streaming responses in OpenAI format."""
    
    try:
        completion_tokens = 0
        full_text = ""
        
        # Send initial chunk
        chunk_data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Stream tokens
        for token_text in streamer:
            if not token_text:
                continue
                
            full_text += token_text
            completion_tokens += 1
            
            # Check for stop strings
            should_stop = False
            for stop_str in stop_strings:
                if stop_str and stop_str in full_text:
                    # Truncate at stop string
                    stop_idx = full_text.index(stop_str)
                    remaining = full_text[:stop_idx]
                    if remaining and remaining != full_text:
                        chunk_data = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": remaining[len(full_text) - len(token_text):]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    should_stop = True
                    break
            
            if should_stop:
                break
            
            # Send token chunk
            chunk_data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Wait for generation thread to complete
        generation_thread.join(timeout=1.0)
        
        # Send final chunk with finish reason and usage
        final_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }
            ],
            "error": str(e)
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    # Ensure workers are ready
    if not all(w.ready.is_set() for w in workers):
        raise HTTPException(status_code=503, detail="model is still loading")
    proc = workers[0].processor
    if proc is None:
        raise HTTPException(status_code=503, detail="processor not ready")

    # Validate message schema
    if not req.messages or not isinstance(req.messages, list):
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    for m in req.messages:
        role = getattr(m, "role", None)
        if role not in {"system", "user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"invalid role: {role}")
        if getattr(m, "content", None) is None:
            raise HTTPException(status_code=400, detail="message content must not be null")

    # Measure total request time
    req_start = time.time()
    rid = uuid.uuid4().hex[:8]
    logp(f"[req] chat start id={rid} model={MODEL_NAME}")

    # Only n=1 supported
    if req.n is not None and int(req.n) != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported")

    # Enforce model name contract
    req_model = req.model if req.model is not None else MODEL_NAME
    if req_model != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"invalid model: expected '{MODEL_NAME}'")

    # Prepare VLM messages (load images if present)
    try:
        vlm_messages = await prepare_vlm_messages(req.messages)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process messages: {e}")

    # Build generation parameters
    max_new_tokens = req.max_tokens if req.max_tokens is not None else MAX_NEW_TOKENS_DEFAULT
    max_new_tokens = min(int(max_new_tokens), MAX_NEW_TOKENS_DEFAULT)
    
    min_new_tokens = req.min_tokens if req.min_tokens is not None else 1
    
    temperature = req.temperature if req.temperature is not None else TEMPERATURE_DEFAULT
    top_p = req.top_p if req.top_p is not None else TOP_P_DEFAULT
    top_k = req.top_k if req.top_k is not None else TOP_K_DEFAULT
    repetition_penalty = req.repetition_penalty if req.repetition_penalty is not None else REPETITION_PENALTY_DEFAULT
    length_penalty = req.length_penalty if req.length_penalty is not None else 1.0
    
    # Determine sampling
    do_sample = DO_SAMPLE_DEFAULT
    if req.temperature is not None or req.top_p is not None:
        do_sample = (temperature > 0.0) or (top_p < 1.0)
    
    seed = req.seed if req.seed is not None else SEED_DEFAULT

    # Stop strings validation
    stop_list: List[str] = []
    if isinstance(req.stop, str):
        raise HTTPException(status_code=400, detail="stop must be an array of strings")
    elif isinstance(req.stop, list):
        if not all(isinstance(s, str) for s in req.stop):
            raise HTTPException(status_code=400, detail="all stop entries must be strings")
        stop_list = req.stop

    params = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "length_penalty": length_penalty,
        "seed": seed,
    }
    if stop_list:
        params["stop"] = stop_list

    # Generate with streaming or non-streaming
    try:
        gen_t0 = time.time()
        gen_result = await router.generate_vlm(vlm_messages, params, is_stream=req.stream)
        
        # Handle streaming response
        if req.stream:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            logp(f"[req] chat stream id={rid} request_id={request_id}")
            
            return StreamingResponse(
                stream_response(
                    streamer=gen_result["streamer"],
                    generation_thread=gen_result["generation_thread"],
                    stop_strings=gen_result.get("stop_strings", []),
                    prompt_tokens=gen_result.get("prompt_tokens", 0),
                    request_id=request_id,
                    model_name=MODEL_NAME,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                }
            )
        
        # Handle non-streaming response
        gen_seconds = time.time() - gen_t0
        
        if isinstance(gen_result, dict):
            completion = gen_result.get("text", "")
            prompt_tokens = int(gen_result.get("prompt_tokens", 0))
            completion_tokens = int(gen_result.get("completion_tokens", 0))
        else:
            completion = str(gen_result)
            prompt_tokens = 0
            completion_tokens = 0
    except HTTPException:
        raise
    except Exception as e:
        if DEBUG_MODE:
            logging.exception("Chat generation failed id=%s", rid)
        logp(f"[req] chat 500 id={rid} error=\"{e}\"")
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")

    # Finish reason heuristic
    finish_reason = "stop"
    if req.max_tokens is not None and completion_tokens >= int(req.max_tokens):
        finish_reason = "length"

    resp = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    
    # Add total latency
    total_latency = time.time() - req_start
    resp["latency_seconds"] = round(total_latency, 4)

    # Log metrics
    try:
        tps = (completion_tokens / max(gen_seconds, 1e-6)) if PROGRESS_LOGS else 0.0
        logp(
            f"[req] chat 200 id={rid} latency={total_latency:.3f}s "
            f"gen={gen_seconds:.3f}s model={MODEL_NAME} "
            f"tokens: prompt={prompt_tokens} completion={completion_tokens} total={prompt_tokens + completion_tokens} "
            f"tps={tps:.1f} do_sample={do_sample} temp={temperature} top_p={top_p} top_k={top_k} seed={seed}"
        )
    except Exception:
        pass

    return resp


# -------------
# Local launcher
# -------------
# IMPORTANT: Keep workers=1 so we don't duplicate model loads.
if __name__ == "__main__":
    import uvicorn

    # uvicorn[standard] provides uvloop and httptools for better throughput
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1)
