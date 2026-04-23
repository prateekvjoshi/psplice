"""
Generation engine.

Handles prompt formatting (including HuggingFace chat templates), model.generate()
calls, token streaming via TextIteratorStreamer, and the base-vs-modified
comparison runs used by `psplice compare`.

All generation goes through this module so that decode settings, interventions,
and chat templates are applied consistently everywhere.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Iterator, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class DecodeSettings:
    """Generation hyperparameters stored in daemon state."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: int = 512

    def to_generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
            kwargs["do_sample"] = True
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
            kwargs["do_sample"] = True
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
            kwargs["do_sample"] = True
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        if "do_sample" not in kwargs:
            kwargs["do_sample"] = False
        return kwargs

    def merge_overrides(self, overrides: dict[str, Any]) -> "DecodeSettings":
        """Return a new DecodeSettings with overrides applied (non-destructive)."""
        import copy
        merged = copy.copy(self)
        for k, v in overrides.items():
            if v is not None and hasattr(merged, k):
                setattr(merged, k, v)
        return merged

    def serialize(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "DecodeSettings":
        return cls(
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            repetition_penalty=data.get("repetition_penalty"),
            max_new_tokens=data.get("max_new_tokens", 512),
        )


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    time_seconds: float
    label: str = "output"


def build_prompt(
    tokenizer,
    user_text: str,
    system_prompt: Optional[str] = None,
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> str:
    """
    Format a prompt using the tokenizer's chat template if available.

    Falls back to raw text if the tokenizer has no template (e.g. base models).

    conversation_history, if provided, should be a list of {"role": ..., "content": ...}
    dicts accumulated across chat turns, NOT including the current user_text.
    """
    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_text})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            logger.warning("Chat template failed (%s), falling back to raw text", exc)

    # Fallback: plain text (suitable for base/completion models)
    parts = []
    if system_prompt:
        parts.append(f"System: {system_prompt}\n\n")
    for msg in (conversation_history or []):
        parts.append(f"{msg['role'].capitalize()}: {msg['content']}\n")
    parts.append(f"User: {user_text}\nAssistant:")
    return "".join(parts)


def generate(
    model,
    tokenizer,
    prompt: str,
    decode_settings: DecodeSettings,
    device: Optional[str] = None,
) -> GenerationResult:
    """
    Run a single, non-streaming generation pass.

    Returns a GenerationResult with the generated text and timing info.
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to the model's device
    model_device = _model_device(model, device)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    gen_kwargs = decode_settings.to_generate_kwargs()

    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
    elapsed = time.perf_counter() - t0

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0][input_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    tokens_generated = len(new_ids)

    return GenerationResult(
        text=text,
        tokens_generated=tokens_generated,
        time_seconds=elapsed,
    )


def generate_streaming(
    model,
    tokenizer,
    prompt: str,
    decode_settings: DecodeSettings,
    device: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Streaming generation using TextIteratorStreamer.

    Yields text chunks as they are produced by the model.  The generator
    exhausts when generation is complete.
    """
    from transformers import TextIteratorStreamer

    inputs = tokenizer(prompt, return_tensors="pt")
    model_device = _model_device(model, device)
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "pad_token_id": tokenizer.eos_token_id,
        **decode_settings.to_generate_kwargs(),
    }

    # model.generate is blocking, so run it in a thread
    thread = threading.Thread(target=_run_generate, args=(model, gen_kwargs))
    thread.daemon = True
    thread.start()

    for text_chunk in streamer:
        yield text_chunk

    thread.join()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_device(model, device: Optional[str]) -> str:
    """Resolve the device to move inputs to."""
    if device:
        return device
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _run_generate(model, gen_kwargs: dict) -> None:
    """Target for threading.Thread — runs model.generate with no_grad."""
    with torch.no_grad():
        model.generate(**gen_kwargs)
