"""
Model loading.

Handles tokenizer and causal-LM loading with dtype/device selection and the
eager-attention flag required by head masking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class LoadConfig:
    """Parameters controlling how the model is loaded."""

    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    eager_attn: bool = False
    trust_remote_code: bool = False


@dataclass
class LoadedModel:
    """A loaded model together with its tokenizer and resolved metadata."""

    model: object           # PreTrainedModel
    tokenizer: object       # PreTrainedTokenizerBase
    model_id: str
    device: str             # resolved device string, e.g. "cuda:0"
    dtype: str              # resolved dtype name, e.g. "bfloat16"
    eager_attn: bool
    param_count: int


def _resolve_device(device: str) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16
        return torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": torch.bfloat16,
    }
    return mapping.get(dtype, torch.bfloat16)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def load_model(config: LoadConfig) -> LoadedModel:
    """
    Load a tokenizer and causal language model from *config.model_id*.

    model_id can be a HuggingFace Hub identifier or a local directory path.

    When config.eager_attn is True the model is loaded with
    attn_implementation="eager", which is required for head masking.  The
    default HuggingFace behaviour on recent Transformers is to use sdpa or
    flash_attention_2 when available, both of which prevent reliable
    head-level surgery.
    """
    # Import here so that non-GPU environments can at least import this module
    from transformers import AutoTokenizer, AutoModelForCausalLM

    resolved_device = _resolve_device(config.device)
    torch_dtype = _resolve_dtype(config.dtype, resolved_device)

    logger.info("Loading tokenizer: %s", config.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code,
    )
    # Ensure a pad token exists (required for batch generation)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": config.trust_remote_code,
    }

    if config.eager_attn:
        model_kwargs["attn_implementation"] = "eager"
        logger.info("Using eager attention (required for head masking)")

    # Use device_map="auto" for multi-GPU / CPU offload; fall back to explicit
    # device placement for single-device scenarios (avoids HF warnings on CPU).
    if resolved_device.startswith("cuda") or resolved_device == "mps":
        model_kwargs["device_map"] = "auto"
    else:
        # CPU: load normally then move
        pass

    logger.info(
        "Loading model: %s  device=%s  dtype=%s  eager=%s",
        config.model_id,
        resolved_device,
        _dtype_name(torch_dtype),
        config.eager_attn,
    )

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)

    # If not using device_map, move explicitly
    if "device_map" not in model_kwargs:
        model = model.to(resolved_device)

    model.eval()

    # Count parameters (millions)
    param_count = sum(p.numel() for p in model.parameters())

    # Determine actual device after placement
    try:
        actual_device = str(next(model.parameters()).device)
    except StopIteration:
        actual_device = resolved_device

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=config.model_id,
        device=actual_device,
        dtype=_dtype_name(torch_dtype),
        eager_attn=config.eager_attn,
        param_count=param_count,
    )
