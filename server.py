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

# CUDA debugging environment variables (for better error messages)
# Set CUDA_LAUNCH_BLOCKING=1 to make CUDA operations synchronous for easier debugging
# Set TORCH_USE_CUDA_DSA=1 to enable device-side assertions for better CUDA error messages
# Note: These slow down performance significantly - only enable when debugging CUDA errors
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")  # Uncomment for debugging
# os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")    # Uncomment for debugging

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
MAX_IMAGES_PER_CONVERSATION = int(os.getenv("MAX_IMAGES_PER_CONVERSATION", "5"))  # Total images across entire conversation history
IMAGE_TOKEN_BUDGET_PER_IMAGE = int(os.getenv("IMAGE_TOKEN_BUDGET_PER_IMAGE", "2048"))  # Conservative token estimate per image
SAFETY_MARGIN_TOKENS = int(os.getenv("SAFETY_MARGIN_TOKENS", "512"))  # Buffer tokens for position embeddings
IMAGE_CACHE_SIZE = int(os.getenv("IMAGE_CACHE_SIZE", "256"))
ALLOW_REMOTE_IMAGES = os.getenv("ALLOW_REMOTE_IMAGES", "1").lower() in {"1", "true", "yes"}

# Supported image formats (common formats only)
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "GIF", "WEBP"}

# Micro-batching knobs (H200 + 35B MoE VLM optimized defaults)
# Configuration optimized for H200 GPU (141GB VRAM) with 35B MoE model:
# - Model: ~70GB (4B active params, memory-efficient MoE)
# - Per-request budget: ~17GB (141GB - 70GB model / 4 concurrent)
# - Total capacity per H200 worker: 16 (4 concurrent + 12 queue)
# - Conservative for production stability with images + long contexts
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "3"))  # H200 can handle larger batches with MoE
BATCH_TIMEOUT_MS = int(os.getenv("BATCH_TIMEOUT_MS", "10"))  # Allow time for image loading
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", os.getenv("MAX_CONTEXT_TOKENS", "8192")))  # hard clamp
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("MAX_NEW_TOKENS_DEFAULT", "8192"))  # Default to max model length
# Backpressure: cap queued requests per worker before 503
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "12"))  # Queue 3x concurrent capacity for burst traffic
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))  # VLM with images needs significant memory per request

# GPU utilization throttling (prevents OOM crashes)
_max_gpu_util_raw = float(os.getenv("MAX_GPU_UTILIZATION", "0.90"))
# Clamp to safe range: 0.85 (85%) to 0.95 (95%)
MAX_GPU_UTILIZATION = max(0.85, min(_max_gpu_util_raw, 0.95))

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
    
    # Note: Custom stop strings have been removed due to StopIteration asyncio conflicts
    # Use eos_token_id instead for stopping generation
    
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


def validate_image_integrity(image_data: bytes, source: str = "image") -> Image.Image:
    """
    Validate that image data is a genuine, uncorrupted image file.
    
    Performs multiple checks:
    1. Can be opened by PIL
    2. Has valid format
    3. Has valid dimensions
    4. Is not corrupted
    
    Returns the validated PIL Image object.
    """
    try:
        # First, try to open the image
        image = Image.open(BytesIO(image_data))
        
        # Verify the image is not corrupted
        # This loads the image header and validates it
        try:
            image.verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Corrupted or invalid image file from {source}: {str(e)}"
            )
        
        # After verify(), need to reopen the image as verify() closes it
        image = Image.open(BytesIO(image_data))
        
        # Validate format
        if not image.format:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to determine image format from {source}"
            )
        
        validate_image_format(image, source)
        
        # Validate dimensions
        width, height = image.size
        if width < 1 or height < 1:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image dimensions from {source}: {width}x{height}"
            )
        
        if width > 10000 or height > 10000:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions too large from {source}: {width}x{height} (max: 10000x10000)"
            )
        
        # Try to load the image data to ensure it's valid
        try:
            image.load()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load image data from {source}: {str(e)}"
            )
        
        return image
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file from {source}: {str(e)}"
        )


async def load_image_from_url(url: str) -> Image.Image:
    """Load image from URL asynchronously with comprehensive validation."""
    try:
        # Limit response size to prevent memory exhaustion (50MB max)
        max_size = 50 * 1024 * 1024  # 50MB
        
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                
                # Check content length if available
                content_length = response.headers.get('Content-Length')
                if content_length and int(content_length) > max_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image too large: {int(content_length) / (1024*1024):.1f}MB (max: 50MB)"
                    )
                
                # Read with size limit
                data = await response.read()
                if len(data) > max_size:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image too large: {len(data) / (1024*1024):.1f}MB (max: 50MB)"
                    )
                
                # Validate image integrity
                image = validate_image_integrity(data, url)
                
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
    except aiohttp.ClientError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image from URL: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {str(e)}")


def load_image_from_base64(data: str) -> Image.Image:
    """Load image from base64 string with comprehensive validation."""
    try:
        # Handle data URLs
        if data.startswith('data:image'):
            # Extract base64 part after comma
            if ',' in data:
                data = data.split(',', 1)[1]
        
        # Decode base64 with size limit (50MB max)
        max_size = 50 * 1024 * 1024  # 50MB
        
        try:
            image_data = base64.b64decode(data)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 encoding: {str(e)}"
            )
        
        # Check decoded size
        if len(image_data) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"Decoded image too large: {len(image_data) / (1024*1024):.1f}MB (max: 50MB)"
            )
        
        # Validate image integrity
        image = validate_image_integrity(image_data, "base64 image")
        
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
async def prepare_vlm_messages(messages: List[ChatMessage]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Convert OpenAI-style messages to VLM format with loaded images.
    
    IMPORTANT: ALL messages are converted to structured format [{"type": "text", ...}]
    even if the input is a simple string. This ensures consistency with VLM processor.
    
    Image counting is across ALL messages in the conversation (not per-message),
    matching OpenAI format behavior where MAX_IMAGES_PER_REQUEST is a total limit.
    
    Returns:
        Tuple of (vlm_messages, metadata) where metadata contains:
        - image_count: Total images in conversation
        - estimated_image_tokens: Conservative token estimate for all images
    """
    vlm_messages = []
    conversation_image_count = 0  # Track ALL images across entire conversation
    estimated_image_tokens = 0  # Estimate total tokens consumed by images
    
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
                            conversation_image_count += 1
                            estimated_image_tokens += IMAGE_TOKEN_BUDGET_PER_IMAGE
                            
                            # Check conversation-level image limit
                            if conversation_image_count > MAX_IMAGES_PER_CONVERSATION:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Maximum {MAX_IMAGES_PER_CONVERSATION} images allowed across entire conversation (found {conversation_image_count} images)"
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
                            conversation_image_count += 1
                            estimated_image_tokens += IMAGE_TOKEN_BUDGET_PER_IMAGE
                            
                            # Check conversation-level image limit
                            if conversation_image_count > MAX_IMAGES_PER_CONVERSATION:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Maximum {MAX_IMAGES_PER_CONVERSATION} images allowed across entire conversation (found {conversation_image_count} images)"
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
                                conversation_image_count += 1
                                estimated_image_tokens += IMAGE_TOKEN_BUDGET_PER_IMAGE
                                
                                # Check conversation-level image limit
                                if conversation_image_count > MAX_IMAGES_PER_CONVERSATION:
                                    raise HTTPException(
                                        status_code=400,
                                        detail=f"Maximum {MAX_IMAGES_PER_CONVERSATION} images allowed across entire conversation (found {conversation_image_count} images)"
                                    )
                                # Load and validate image
                                image = await load_image(image_source)
                                content_parts.append({"type": "image", "image": image})
                    elif part.get("type") == "image":
                        # Legacy format
                        image_source = part.get("image") or (part.get("image_url", {}).get("url") if isinstance(part.get("image_url"), dict) else None)
                        if image_source:
                            conversation_image_count += 1
                            estimated_image_tokens += IMAGE_TOKEN_BUDGET_PER_IMAGE
                            
                            # Check conversation-level image limit
                            if conversation_image_count > MAX_IMAGES_PER_CONVERSATION:
                                raise HTTPException(
                                    status_code=400,
                                    detail=f"Maximum {MAX_IMAGES_PER_CONVERSATION} images allowed across entire conversation (found {conversation_image_count} images)"
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
    
    # Return messages with metadata (actual token validation happens after tokenization)
    image_metadata = {
        "image_count": conversation_image_count,
        "estimated_image_tokens": estimated_image_tokens  # For display/logging only
    }
    
    return vlm_messages, image_metadata


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
    image_metadata: Dict[str, int] = None  # Image count and token estimates


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
        try:
            assert self.model is not None and self.processor is not None

            messages_list = [r.messages for r in batch]
            params_list = [r.params for r in batch]

            # Extract generation parameters (use max across batch for conservative approach)
            max_new_tokens = min(
                max([int(p.get("max_new_tokens", MAX_NEW_TOKENS_DEFAULT)) for p in params_list]),
                MAX_MODEL_LENGTH,  # Cap at model's maximum length
            )
            
            # Ensure we have a reasonable buffer for max_new_tokens
            # This prevents position_ids from exceeding max_position_embeddings
            max_new_tokens = min(max_new_tokens, MAX_MODEL_LENGTH // 2)
            
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
                if DEBUG_MODE:
                    logging.exception(f"[{self.device}] apply_chat_template failed")
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(
                            HTTPException(status_code=500, detail=f"Failed to process VLM inputs: {e}")
                        )
                return

            # Get ACTUAL input token count from tokenized inputs (includes text + images)
            lengths = inputs["attention_mask"].sum(dim=1) if "attention_mask" in inputs else torch.tensor([inputs["input_ids"].size(1)] * len(batch))
            actual_input_tokens = int(lengths.max().item()) if len(lengths) > 0 else 0
            
            # Get image count for error messages
            image_metadata = batch[0].image_metadata if batch else {}
            image_count = image_metadata.get("image_count", 0) if image_metadata else 0
            
            # STRICT VALIDATION: actual_tokens + max_new_tokens + 512 margin <= MAX_MODEL_LENGTH
            required_total = actual_input_tokens + max_new_tokens + SAFETY_MARGIN_TOKENS
            
            if required_total > MAX_MODEL_LENGTH:
                error_msg = (
                    f"Request exceeds model capacity: {actual_input_tokens} input tokens (with {image_count} images) "
                    f"+ {max_new_tokens} requested generation + {SAFETY_MARGIN_TOKENS} safety margin "
                    f"= {required_total} tokens > {MAX_MODEL_LENGTH} limit. "
                    f"Reduce conversation length, number of images, or max_tokens."
                )
                if DEBUG_MODE:
                    logging.error(f"[{self.device}] {error_msg}")
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(
                            HTTPException(status_code=400, detail=error_msg)
                        )
                return
            
            # Calculate safe max_new_tokens (should already fit, but clamp for safety)
            available_tokens = MAX_MODEL_LENGTH - actual_input_tokens - SAFETY_MARGIN_TOKENS
            safe_max_new_tokens = min(max_new_tokens, available_tokens)
            
            if safe_max_new_tokens < 1:
                error_msg = (
                    f"No room for generation: {actual_input_tokens} input tokens + {SAFETY_MARGIN_TOKENS} margin "
                    f"= {actual_input_tokens + SAFETY_MARGIN_TOKENS} / {MAX_MODEL_LENGTH}. "
                    f"Reduce conversation length or number of images."
                )
                if DEBUG_MODE:
                    logging.error(f"[{self.device}] {error_msg}")
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(
                            HTTPException(status_code=400, detail=error_msg)
                        )
                return
            
            # Use the safe value
            max_new_tokens = safe_max_new_tokens
            
            if DEBUG_MODE:
                logging.debug(
                    f"[{self.device}] Token validation: input={actual_input_tokens} (with {image_count} images), "
                    f"generation={max_new_tokens}, safety={SAFETY_MARGIN_TOKENS}, "
                    f"total={actual_input_tokens + max_new_tokens + SAFETY_MARGIN_TOKENS}, limit={MAX_MODEL_LENGTH}"
                )

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
                        "max_length": min(MAX_MODEL_LENGTH, max_input_length + max_new_tokens + 1),
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
                    if DEBUG_MODE:
                        logging.exception(f"[{self.device}] model.generate failed")
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
            # NOTE: Stop strings are already handled by StoppingCriteria during generation
            # No need for post-processing truncation
            for i, req in enumerate(batch):
                text = decoded_texts[i]

                # Token counts
                prompt_tok_count = int(lengths[i].item()) if i < len(lengths) else 0
                completion_tok_count = len(generated_ids_trimmed[i])

                if not req.future.done():
                    req.future.set_result({
                        "text": text,
                        "prompt_tokens": prompt_tok_count,
                        "completion_tokens": completion_tok_count,
                    })
        
        except Exception as e:
            # Catch-all error handler - ensures server never crashes
            if DEBUG_MODE:
                logging.exception(f"[{self.device}] _run_batch critical error")
            # Set exception on all pending futures
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(
                        HTTPException(status_code=500, detail=f"Batch processing failed: {e}")
                    )


    async def _run_streaming_batch(self, batch: List[_PendingReq]):
        """Process VLM batch with streaming support."""
        try:
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
            max_new_tokens = min(int(params.get("max_new_tokens", MAX_NEW_TOKENS_DEFAULT)), MAX_MODEL_LENGTH)  # Cap at model's maximum length
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
                if DEBUG_MODE:
                    logging.exception(f"[{self.device}] apply_chat_template failed (streaming)")
                if not req.future.done():
                    req.future.set_exception(
                        HTTPException(status_code=500, detail=f"Failed to process VLM inputs: {e}")
                    )
                return

            # Get ACTUAL input token count from tokenized inputs (includes text + images)
            actual_input_tokens = inputs["input_ids"].shape[1]
            
            # Get image count for error messages
            image_metadata = req.image_metadata if req.image_metadata else {}
            image_count = image_metadata.get("image_count", 0)
            
            # STRICT VALIDATION: actual_tokens + max_new_tokens + 512 margin <= MAX_MODEL_LENGTH
            required_total = actual_input_tokens + max_new_tokens + SAFETY_MARGIN_TOKENS
            
            if required_total > MAX_MODEL_LENGTH:
                error_msg = (
                    f"Request exceeds model capacity: {actual_input_tokens} input tokens (with {image_count} images) "
                    f"+ {max_new_tokens} requested generation + {SAFETY_MARGIN_TOKENS} safety margin "
                    f"= {required_total} tokens > {MAX_MODEL_LENGTH} limit. "
                    f"Reduce conversation length, number of images, or max_tokens."
                )
                if DEBUG_MODE:
                    logging.error(f"[{self.device}] {error_msg}")
                if not req.future.done():
                    req.future.set_exception(
                        HTTPException(status_code=400, detail=error_msg)
                    )
                return
            
            # Calculate safe max_new_tokens (should already fit, but clamp for safety)
            available_tokens = MAX_MODEL_LENGTH - actual_input_tokens - SAFETY_MARGIN_TOKENS
            safe_max_new_tokens = min(max_new_tokens, available_tokens)
            
            if safe_max_new_tokens < 1:
                error_msg = (
                    f"No room for generation: {actual_input_tokens} input tokens + {SAFETY_MARGIN_TOKENS} margin "
                    f"= {actual_input_tokens + SAFETY_MARGIN_TOKENS} / {MAX_MODEL_LENGTH}. "
                    f"Reduce conversation length or number of images."
                )
                if DEBUG_MODE:
                    logging.error(f"[{self.device}] {error_msg}")
                if not req.future.done():
                    req.future.set_exception(
                        HTTPException(status_code=400, detail=error_msg)
                    )
                return
            
            # Use the safe value
            max_new_tokens = safe_max_new_tokens
            
            if DEBUG_MODE:
                logging.debug(
                    f"[{self.device}] Streaming validation: input={actual_input_tokens} (with {image_count} images), "
                    f"generation={max_new_tokens}, safety={SAFETY_MARGIN_TOKENS}, "
                    f"total={actual_input_tokens + max_new_tokens + SAFETY_MARGIN_TOKENS}, limit={MAX_MODEL_LENGTH}"
                )
            
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
            try:
                streamer = TextIteratorStreamer(
                    self.processor.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            except Exception as e:
                if DEBUG_MODE:
                    logging.exception(f"[{self.device}] Failed to create streamer")
                if not req.future.done():
                    req.future.set_exception(
                        HTTPException(status_code=500, detail=f"Failed to create streamer: {e}")
                    )
                return

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
                "max_length": min(MAX_MODEL_LENGTH, prompt_length + max_new_tokens + 1),
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
            generation_error = None
            def generate_fn():
                nonlocal generation_error
                try:
                    with torch.inference_mode():
                        try:
                            self.model.generate(**gen_kwargs)
                        except Exception as e:
                            generation_error = str(e)
                            if DEBUG_MODE:
                                logging.exception(f"[{self.device}] Streaming generation failed")
                            # Signal end to streamer
                            try:
                                streamer.end()
                            except:
                                pass
                except Exception as e:
                    # Catch-all for any errors in the thread
                    generation_error = str(e)
                    if DEBUG_MODE:
                        logging.exception(f"[{self.device}] Critical error in generation thread")
                    try:
                        streamer.end()
                    except:
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
        
        except Exception as e:
            # Catch-all error handler - ensures server never crashes
            if DEBUG_MODE:
                logging.exception(f"[{self.device}] _run_streaming_batch critical error")
            # Set exception on all pending futures
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(
                        HTTPException(status_code=500, detail=f"Streaming batch processing failed: {e}")
                    )


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
        # Semaphore to limit concurrent requests across all workers
        self.concurrent_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.active_requests = 0  # Track active request count for monitoring
        # Cache GPU utilization to avoid querying every microsecond
        self._gpu_util_cache: Dict[int, Tuple[float, float]] = {}  # device_id -> (utilization, timestamp)
        self._gpu_util_cache_ttl = 0.1  # 100ms cache

    async def start(self):
        await asyncio.gather(*(w.start() for w in self.workers))

    def _pick_worker(self) -> GPUWorker:
        # Simple least-queue heuristic (effective for bursty loads)
        return min(self.workers, key=lambda w: w.queue.qsize())

    def get_gpu_utilization(self) -> Dict[int, float]:
        """
        Get current GPU memory utilization per device.
        Returns dict of device_id -> utilization (0.0 to 1.0)
        
        Uses caching to avoid excessive CUDA queries.
        Returns 0.0 for CPU fallback or if CUDA unavailable.
        """
        if CPU_FALLBACK or not torch.cuda.is_available():
            return {0: 0.0}
        
        current_time = time.time()
        utilizations = {}
        
        for worker in self.workers:
            device_str = str(worker.device)
            if not device_str.startswith("cuda:"):
                utilizations[0] = 0.0
                continue
            
            try:
                device_id = int(device_str.split(":")[1])
                
                # Check cache
                if device_id in self._gpu_util_cache:
                    cached_util, cached_time = self._gpu_util_cache[device_id]
                    if current_time - cached_time < self._gpu_util_cache_ttl:
                        utilizations[device_id] = cached_util
                        continue
                
                # Query GPU memory
                allocated = torch.cuda.memory_allocated(device_id)
                props = torch.cuda.get_device_properties(device_id)
                total = props.total_memory
                
                # Calculate utilization
                utilization = allocated / total if total > 0 else 0.0
                
                # Update cache
                self._gpu_util_cache[device_id] = (utilization, current_time)
                utilizations[device_id] = utilization
                
            except Exception:
                # On error, return 0.0 for safety
                utilizations.get(device_id, 0.0)
        
        return utilizations

    def is_overloaded(self) -> bool:
        """
        Check if server should reject new requests.
        Returns True if:
        1. All concurrent slots are taken AND all queues are full
        2. This prevents accepting requests that cannot be queued
        """
        try:
            # Check if we're at concurrent limit
            at_concurrent_limit = self.concurrent_semaphore.locked()
            
            # Check if all worker queues are full
            all_queues_full = all(w.queue.qsize() >= MAX_QUEUE_SIZE for w in self.workers)
            
            # Reject only if both concurrent limit reached AND queues are full
            return at_concurrent_limit and all_queues_full
        except Exception:
            return False

    async def generate_vlm(self, messages: List[Dict[str, Any]], params: Dict[str, Any], is_stream: bool = False, image_metadata: Dict[str, int] = None):
        """
        Generate response for VLM messages with concurrent request limiting and GPU throttling.
        
        Flow:
        1. Check GPU utilization (reject if over threshold)
        2. Check if should reject (concurrent limit + queue full)
        3. Acquire semaphore (blocks if at concurrent limit but queue has space)
        4. Process request
        5. Release semaphore in finally block (always happens)
        
        Args:
            messages: VLM messages with loaded images
            params: Generation parameters
            is_stream: Whether to stream the response
            image_metadata: Dict with 'image_count' and 'estimated_image_tokens'
        """
        # Check GPU utilization before accepting request
        gpu_utils = self.get_gpu_utilization()
        max_util = max(gpu_utils.values()) if gpu_utils else 0.0
        
        if max_util > MAX_GPU_UTILIZATION:
            raise HTTPException(
                status_code=503,
                detail=f"GPU utilization at {max_util:.1%}, max allowed {MAX_GPU_UTILIZATION:.1%}. Please retry later."
            )
        
        # Check if we should immediately reject (concurrent limit + all queues full)
        if self.is_overloaded():
            # Get current state for error message
            concurrent_active = MAX_CONCURRENT_REQUESTS - self.concurrent_semaphore._value
            total_queued = sum(w.queue.qsize() for w in self.workers)
            raise HTTPException(
                status_code=503,
                detail=f"server overloaded: {concurrent_active} processing, {total_queued} queued (max: {MAX_CONCURRENT_REQUESTS} concurrent, {MAX_QUEUE_SIZE} queue per worker)"
            )
        
        # Acquire semaphore to limit concurrent requests
        # This will block if at limit but queue has space (request will wait)
        await self.concurrent_semaphore.acquire()
        
        try:
            # Track active request count
            self.active_requests += 1
            
            # Pick worker and enqueue request
            worker = self._pick_worker()
            loop = asyncio.get_running_loop()
            fut: asyncio.Future = loop.create_future()
            req = _PendingReq(messages=messages, params=params, future=fut, is_stream=is_stream, image_metadata=image_metadata)
            await worker.queue.put(req)
            
            # Wait for result
            result = await fut
            return result
        finally:
            # CRITICAL: Always release semaphore, even on errors
            self.active_requests -= 1
            self.concurrent_semaphore.release()


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
    """
    Health check endpoint - called repeatedly by Kubernetes/load balancers/monitoring systems.
    
    This is NORMAL and EXPECTED behavior in production environments:
    - Kubernetes probes this endpoint every few seconds for liveness/readiness checks
    - Load balancers use it to determine if the instance should receive traffic
    - Monitoring systems poll it to track service health and availability
    
    This is NOT a bug - it's how production infrastructure monitoring works.
    """
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
    
    # Calculate load management metrics
    concurrent_active = MAX_CONCURRENT_REQUESTS - router.concurrent_semaphore._value
    concurrent_available = router.concurrent_semaphore._value
    total_queued = sum(w.queue.qsize() for w in workers)
    total_capacity = MAX_CONCURRENT_REQUESTS + (MAX_QUEUE_SIZE * len(workers))
    
    # Get GPU utilization metrics
    gpu_utils = router.get_gpu_utilization()
    gpu_memory_info = {}
    
    if not CPU_FALLBACK and torch.cuda.is_available():
        for device_id, utilization in gpu_utils.items():
            try:
                allocated = torch.cuda.memory_allocated(device_id)
                props = torch.cuda.get_device_properties(device_id)
                total_mem = props.total_memory
                
                gpu_memory_info[f"cuda:{device_id}"] = {
                    "utilization": round(utilization, 4),
                    "allocated_gb": round(allocated / (1024**3), 2),
                    "total_gb": round(total_mem / (1024**3), 2),
                    "allocated_bytes": allocated,
                    "total_bytes": total_mem,
                }
            except Exception:
                gpu_memory_info[f"cuda:{device_id}"] = {
                    "utilization": round(utilization, 4),
                    "error": "Failed to query memory"
                }
    
    # Check if throttling is active
    max_gpu_util = max(gpu_utils.values()) if gpu_utils else 0.0
    throttling_active = max_gpu_util > MAX_GPU_UTILIZATION
    
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
        "load_management": {
            "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
            "concurrent_active": concurrent_active,
            "concurrent_available": concurrent_available,
            "max_queue_size": MAX_QUEUE_SIZE,
            "total_queued": total_queued,
            "total_capacity": total_capacity,
            "current_load": concurrent_active + total_queued,
            "load_percentage": round(((concurrent_active + total_queued) / total_capacity) * 100, 1) if total_capacity > 0 else 0,
        },
        "gpu_utilization": {
            "max_threshold": MAX_GPU_UTILIZATION,
            "current_max": round(max_gpu_util, 4),
            "throttling_active": throttling_active,
            "per_device": gpu_memory_info,
        },
        "max_queue_size": MAX_QUEUE_SIZE,  # Keep for backwards compatibility
        "dtype": str(DTYPE),
        "attn": ATTN_IMPL,
        "batch": {"max_batch_size": MAX_BATCH_SIZE, "batch_timeout_ms": BATCH_TIMEOUT_MS},
        "vlm_config": {
            "max_model_length": MAX_MODEL_LENGTH,
            "max_image_size": MAX_IMAGE_SIZE,
            "max_images_per_request": MAX_IMAGES_PER_REQUEST,
            "max_images_per_conversation": MAX_IMAGES_PER_CONVERSATION,
            "image_token_budget_per_image": IMAGE_TOKEN_BUDGET_PER_IMAGE,
            "safety_margin_tokens": SAFETY_MARGIN_TOKENS,
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
    max_new_tokens: int,
) -> AsyncGenerator[str, None]:
    """
    Generate SSE-formatted streaming responses in OpenAI format.
    
    NOTE: Once streaming starts, HTTP status is already 200. Errors during
    generation are communicated via error chunks in the SSE stream.
    """
    
    completion_tokens = 0
    full_text = ""
    generation_error = None
    finish_reason = None  # Track finish reason during streaming
    
    try:
        # Send initial chunk with role - wrap in try/except for safety
        try:
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
        except Exception as e:
            if DEBUG_MODE:
                logging.exception("Error sending initial chunk")
            generation_error = f"Failed to send initial chunk: {e}"
        
        # Stream tokens asynchronously with timeout protection
        loop = asyncio.get_event_loop()
        streamer_iter = iter(streamer)
        
        # Total timeout for entire generation (5 minutes)
        generation_start = time.time()
        max_generation_time = 300.0  # 5 minutes total
        
        while True:
            # Check if generation thread is still alive
            if not generation_thread.is_alive():
                # Thread finished - check for any remaining tokens
                try:
                    # Try to get last token with very short timeout
                    # CRITICAL: Wrap next() to prevent StopIteration from escaping into asyncio
                    def safe_next_final():
                        try:
                            return next(streamer_iter)
                        except StopIteration:
                            return None
                    
                    token_text = await asyncio.wait_for(
                        loop.run_in_executor(None, safe_next_final),
                        timeout=0.1
                    )
                    # None means StopIteration was caught - stream is done
                    if token_text is None:
                        break
                    if token_text:
                        full_text += token_text
                        completion_tokens += 1
                        try:
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
                        except Exception as chunk_error:
                            if DEBUG_MODE:
                                logging.exception("Error sending final token chunk")
                            generation_error = f"Chunk send error: {chunk_error}"
                            break
                except asyncio.TimeoutError:
                    pass
                # Generation thread finished, exit loop
                break
            
            # Check total generation timeout
            if time.time() - generation_start > max_generation_time:
                generation_error = "Generation timeout - exceeded 5 minutes"
                break
            
            try:
                # Get next token with timeout (0.5 seconds per token for instant stop detection)
                # CRITICAL: Wrap next() call to prevent StopIteration from escaping into asyncio
                def safe_next_token():
                    try:
                        return next(streamer_iter)
                    except StopIteration:
                        # Convert StopIteration to None to signal end
                        return None
                
                token_text = await asyncio.wait_for(
                    loop.run_in_executor(None, safe_next_token),
                    timeout=0.5
                )
                
                # None signals end of iteration (StopIteration was caught)
                if token_text is None:
                    break
                    
                if not token_text:
                    continue
                    
                full_text += token_text
                completion_tokens += 1
                
                # Send token chunk immediately - wrap in try/except
                try:
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
                except Exception as chunk_error:
                    if DEBUG_MODE:
                        logging.exception("Error sending token chunk")
                    generation_error = f"Chunk send error: {chunk_error}"
                    break
                
                # Check if we've reached max_new_tokens limit
                if completion_tokens >= max_new_tokens:
                    finish_reason = "length"
                    if DEBUG_MODE:
                        logging.debug(
                            f"[stream] Hit max_new_tokens limit: completion_tokens={completion_tokens} "
                            f"max_new_tokens={max_new_tokens}"
                        )
                    break
                
                # Yield control to event loop to ensure immediate sending
                await asyncio.sleep(0)
                
            except asyncio.TimeoutError:
                # Token timeout - check if thread is still alive
                if not generation_thread.is_alive():
                    # Thread finished but no more tokens
                    break
                # Thread still alive but no token - this might indicate a hang
                # Wait a bit more but set error flag
                generation_error = "Token generation timeout"
                break
            except Exception as token_error:
                # Catch any other token processing errors
                if DEBUG_MODE:
                    logging.exception("Error processing token")
                generation_error = f"Token processing error: {token_error}"
                break
        
    except Exception as e:
        # Exception during streaming - log and set error
        if DEBUG_MODE:
            logging.exception("Streaming error")
        generation_error = str(e)
    
    finally:
        # ALWAYS send final chunk with finish reason and usage - this is critical!
        try:
            # Wait briefly for generation thread to complete
            if generation_thread.is_alive():
                generation_thread.join(timeout=1.0)
            
            # Determine finish reason based on completion status
            # Only set if not already set during streaming
            if finish_reason is None:
                finish_reason = "stop"  # Default: normal EOS token completion
                
                if generation_error:
                    finish_reason = "error"
                elif completion_tokens >= max_new_tokens:
                    # Hit the maximum token limit
                    finish_reason = "length"
            
            # Debug logging for finish_reason detection
            if DEBUG_MODE:
                logging.debug(
                    f"[stream] Final finish_reason={finish_reason} completion_tokens={completion_tokens} "
                    f"max_new_tokens={max_new_tokens} generation_error={generation_error}"
                )
            
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
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            }
            
            # Add error details if present
            if generation_error:
                final_chunk["error"] = {
                    "message": generation_error,
                    "type": "generation_error",
                    "code": "generation_failed"
                }
            
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as final_error:
            # Last resort - send minimal done signal
            if DEBUG_MODE:
                logging.exception("Error sending final chunk")
            try:
                yield "data: [DONE]\n\n"
            except:
                pass


@app.post("/v1/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(req: ChatCompletionRequest):
    # Measure total request time
    req_start = time.time()
    rid = uuid.uuid4().hex[:8]
    
    # CRITICAL: All validation must happen BEFORE starting streaming response
    # to ensure proper HTTP status codes are returned
    
    # Check 1: Ensure workers are ready (503 if not)
    if not all(w.ready.is_set() for w in workers):
        raise HTTPException(status_code=503, detail="model is still loading")
    
    # Check 2: Ensure startup was successful (503 if failed)
    if not STARTUP_OK:
        error_msg = f"model failed to load: {STARTUP_ERROR}" if STARTUP_ERROR else "model not ready"
        raise HTTPException(status_code=503, detail=error_msg)
    
    # Check 3: Verify processor is ready (503 if not)
    proc = workers[0].processor
    if proc is None:
        raise HTTPException(status_code=503, detail="processor not ready")

    # Check 4: Validate message schema (400 for bad requests)
    if not req.messages or not isinstance(req.messages, list):
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")
    for m in req.messages:
        role = getattr(m, "role", None)
        if role not in {"system", "user", "assistant"}:
            raise HTTPException(status_code=400, detail=f"invalid role: {role}")
        if getattr(m, "content", None) is None:
            raise HTTPException(status_code=400, detail="message content must not be null")

    # Check 5: Only n=1 supported (400 for bad request)
    if req.n is not None and int(req.n) != 1:
        raise HTTPException(status_code=400, detail="Only n=1 is supported")

    # Check 6: Enforce model name contract (400 for bad request)
    req_model = req.model if req.model is not None else MODEL_NAME
    if req_model != MODEL_NAME:
        raise HTTPException(status_code=400, detail=f"invalid model: expected '{MODEL_NAME}'")

    # Check 7: Check server overload before processing (503 if overloaded)
    if router.is_overloaded():
        raise HTTPException(status_code=503, detail="server overloaded: queue full")

    logp(f"[req] chat start id={rid} model={MODEL_NAME} stream={req.stream}")

    # Check 8: Prepare VLM messages - validate and load images (400 for bad data)
    try:
        vlm_messages, image_metadata = await prepare_vlm_messages(req.messages)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process messages: {e}")

    # Build generation parameters
    max_new_tokens = req.max_tokens if req.max_tokens is not None else MAX_NEW_TOKENS_DEFAULT
    max_new_tokens = min(int(max_new_tokens), MAX_MODEL_LENGTH)  # Cap at model's maximum length
    
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

    # Note: Custom stop strings feature has been removed due to StopIteration asyncio conflicts
    # Generation will stop naturally at eos_token_id
    if hasattr(req, 'stop') and req.stop is not None:
        raise HTTPException(status_code=400, detail="Custom stop strings are not supported. Use max_tokens to limit generation.")

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

    # Generate with streaming or non-streaming
    # Wrap in comprehensive error handling to prevent server crashes
    try:
        gen_t0 = time.time()
        try:
            gen_result = await router.generate_vlm(vlm_messages, params, is_stream=req.stream, image_metadata=image_metadata)
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as gen_error:
            # Catch ANY generation error and return proper error response
            if DEBUG_MODE:
                logging.exception(f"Generation error id={rid}")
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(gen_error)}"
            )
        
        # Handle streaming response
        if req.stream:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            logp(f"[req] chat stream id={rid} request_id={request_id}")
            
            return StreamingResponse(
                stream_response(
                    streamer=gen_result["streamer"],
                    generation_thread=gen_result["generation_thread"],
                    prompt_tokens=gen_result.get("prompt_tokens", 0),
                    request_id=request_id,
                    model_name=MODEL_NAME,
                    max_new_tokens=max_new_tokens,
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
    # Check against actual max_new_tokens used (not just req.max_tokens which may be None)
    if completion_tokens >= max_new_tokens:
        finish_reason = "length"
    
    # Debug logging for finish_reason detection
    if DEBUG_MODE:
        logging.debug(
            f"[req] finish_reason={finish_reason} completion_tokens={completion_tokens} "
            f"max_new_tokens={max_new_tokens} req.max_tokens={req.max_tokens}"
        )

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

    # Log metrics (internal only, not in response)
    total_latency = time.time() - req_start
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
