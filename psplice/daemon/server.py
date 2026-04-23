"""
psplice daemon server.

FastAPI application that runs in the background daemon process.  All model
state is owned by a single ModelSession object stored at _session.  Route
handlers are deliberately thin: validate HTTP input, call a ModelSession
method, return JSON.  No route handler touches hook_manager or
intervention_registry directly.

Entry point
-----------
Run as::

    python -m psplice.daemon.server \\
        --model-id Qwen/Qwen2.5-7B-Instruct \\
        --port 54321 \\
        --device cuda \\
        --dtype bfloat16 \\
        --eager-attn
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from psplice.state.model_session import ModelSession, ModelSessionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global session — set once by run_daemon(), read by every route handler
# ---------------------------------------------------------------------------

_session: Optional[ModelSession] = None


def _get_session() -> ModelSession:
    if _session is None:
        raise HTTPException(status_code=503, detail="No model loaded. Run `psplice load <model>`.")
    return _session


app = FastAPI(title="psplice daemon", version="0.1.0")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    conversation_history: Optional[list[dict[str, str]]] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    streaming: bool = False


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    time_seconds: float


class CompareRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    max_new_tokens: Optional[int] = None


class CompareResponse(BaseModel):
    base: GenerateResponse
    modified: GenerateResponse


class SteerAddRequest(BaseModel):
    name: str
    vector_path: str
    layer_indices: list[int]
    scale: float = 1.0


class HeadMaskRequest(BaseModel):
    layer_heads: dict[str, list[int]]


class LayerSkipRequest(BaseModel):
    skip_from: int


class LoraLoadRequest(BaseModel):
    adapter_path: str


class VectorExtractRequest(BaseModel):
    positive_prompts: list[str]
    negative_prompts: list[str]
    layer_indices: list[int]
    output_path: str
    token_aggregation: str = "mean"   # "mean" or "last"


class VectorExtractResponse(BaseModel):
    output_path: str
    layer_indices: list[int]
    hidden_size: int
    num_positive: int
    num_negative: int
    format: str   # "uniform" (1D tensor) or "per_layer" (dict)


class DecodeSetRequest(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    max_new_tokens: Optional[int] = None


class PresetSaveRequest(BaseModel):
    name: str


class PresetLoadRequest(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    sess = _get_session()
    return {"status": "ok", "model_id": sess.model_id}


@app.get("/status")
async def status():
    sess = _get_session()
    return sess.status_dict()


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

@app.post("/stop")
async def stop():
    _cleanup()
    asyncio.get_event_loop().call_soon(_do_shutdown)
    return {"ok": True}


def _do_shutdown():
    os.kill(os.getpid(), signal.SIGTERM)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    sess = _get_session()
    from psplice.runtime.generation import build_prompt, generate, generate_streaming

    prompt = build_prompt(
        sess.tokenizer,
        req.prompt,
        system_prompt=req.system_prompt,
        conversation_history=req.conversation_history,
    )

    overrides = {k: v for k, v in {
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
    }.items() if v is not None}

    settings = sess.decode_settings.merge_overrides(overrides)

    if req.streaming:
        async def token_stream() -> AsyncGenerator[str, None]:
            for chunk in generate_streaming(sess.model, sess.tokenizer, prompt, settings):
                yield f"data: {json.dumps({'token': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: generate(sess.model, sess.tokenizer, prompt, settings),
    )
    return GenerateResponse(
        text=result.text,
        tokens_generated=result.tokens_generated,
        time_seconds=result.time_seconds,
    )


# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------

@app.post("/compare")
async def compare_endpoint(req: CompareRequest):
    sess = _get_session()
    from psplice.runtime.generation import build_prompt

    prompt = build_prompt(
        sess.tokenizer,
        req.prompt,
        system_prompt=req.system_prompt,
    )

    overrides = {}
    if req.max_new_tokens is not None:
        overrides["max_new_tokens"] = req.max_new_tokens
    settings = sess.decode_settings.merge_overrides(overrides)

    base, modified = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: sess.compare(prompt, settings),
    )

    return CompareResponse(
        base=GenerateResponse(
            text=base.text,
            tokens_generated=base.tokens_generated,
            time_seconds=base.time_seconds,
        ),
        modified=GenerateResponse(
            text=modified.text,
            tokens_generated=modified.tokens_generated,
            time_seconds=modified.time_seconds,
        ),
    )


# ---------------------------------------------------------------------------
# Vector extraction
# ---------------------------------------------------------------------------

@app.post("/vectors/extract")
async def vectors_extract(req: VectorExtractRequest):
    sess = _get_session()

    if not sess.arch.has_standard_layers:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model family '{sess.arch.family}' does not expose model.model.layers. "
                f"Vector extraction is not supported for this architecture."
            ),
        )

    bad = [i for i in req.layer_indices if i < 0 or i >= sess.arch.num_layers]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"Layer indices out of range for {sess.arch.num_layers}-layer model: {bad}",
        )

    if req.token_aggregation not in ("mean", "last"):
        raise HTTPException(
            status_code=400,
            detail=f"token_aggregation must be 'mean' or 'last', got {req.token_aggregation!r}",
        )

    from psplice.runtime.vector_extraction import extract_contrastive_vector, save_vector

    try:
        vectors = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: extract_contrastive_vector(
                sess.model,
                sess.tokenizer,
                req.positive_prompts,
                req.negative_prompts,
                req.layer_indices,
                req.token_aggregation,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    try:
        save_vector(vectors, req.output_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save vector: {exc}")

    hidden_size = next(iter(vectors.values())).shape[0]
    fmt = "uniform" if len(req.layer_indices) == 1 else "per_layer"

    return VectorExtractResponse(
        output_path=req.output_path,
        layer_indices=sorted(vectors.keys()),
        hidden_size=hidden_size,
        num_positive=len(req.positive_prompts),
        num_negative=len(req.negative_prompts),
        format=fmt,
    )


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------

@app.post("/steer/add")
async def steer_add(req: SteerAddRequest):
    sess = _get_session()
    from psplice.interventions.steering import SteeringIntervention

    iv = SteeringIntervention(
        name=req.name,
        vector_path=req.vector_path,
        layer_indices=req.layer_indices,
        scale=req.scale,
    )
    try:
        sess.apply_intervention(iv)
    except ModelSessionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "name": req.name}


@app.delete("/steer/{name}")
async def steer_remove(name: str):
    sess = _get_session()
    if not sess.remove_intervention(name):
        raise HTTPException(status_code=404, detail=f"No intervention named '{name}'")
    return {"ok": True}


@app.get("/steer")
async def steer_list():
    sess = _get_session()
    return [iv.describe() for iv in sess.intervention_registry.by_type("steering")]


# ---------------------------------------------------------------------------
# Head masking
# ---------------------------------------------------------------------------

@app.post("/heads/mask")
async def heads_mask(req: HeadMaskRequest):
    sess = _get_session()
    from psplice.interventions.heads import HeadMaskIntervention

    layer_heads = {int(k): v for k, v in req.layer_heads.items()}

    # Remove existing mask before adding a new one (replace semantics)
    sess.remove_intervention("_head_mask")

    iv = HeadMaskIntervention(name="_head_mask", layer_heads=layer_heads)
    try:
        sess.apply_intervention(iv)
    except ModelSessionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True}


@app.delete("/heads")
async def heads_clear():
    sess = _get_session()
    removed = sess.remove_intervention("_head_mask")
    return {"ok": True, "removed": removed}


@app.get("/heads")
async def heads_list():
    sess = _get_session()
    return [iv.describe() for iv in sess.intervention_registry.by_type("head_mask")]


# ---------------------------------------------------------------------------
# Layer skip
# ---------------------------------------------------------------------------

@app.post("/layers/skip")
async def layers_skip(req: LayerSkipRequest):
    sess = _get_session()
    from psplice.interventions.layers import LayerSkipIntervention

    # Replace semantics
    sess.remove_intervention("_layer_skip")

    iv = LayerSkipIntervention(name="_layer_skip", skip_from=req.skip_from)
    try:
        sess.apply_intervention(iv)
    except ModelSessionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "skip_from": req.skip_from}


@app.delete("/layers")
async def layers_clear():
    sess = _get_session()
    removed = sess.remove_intervention("_layer_skip")
    return {"ok": True, "removed": removed}


@app.get("/layers")
async def layers_info():
    sess = _get_session()
    return [iv.describe() for iv in sess.intervention_registry.by_type("layer_skip")]


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

@app.post("/lora/load")
async def lora_load(req: LoraLoadRequest):
    sess = _get_session()
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: sess.load_lora(req.adapter_path),
        )
    except ModelSessionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "adapter_path": sess.active_lora_path}


@app.delete("/lora")
async def lora_unload():
    sess = _get_session()
    try:
        sess.unload_lora()
    except ModelSessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


@app.get("/lora")
async def lora_info():
    sess = _get_session()
    return {
        "active": sess.active_lora_path is not None,
        "adapter_path": sess.active_lora_path,
    }


# ---------------------------------------------------------------------------
# Decode settings
# ---------------------------------------------------------------------------

@app.post("/decode/set")
async def decode_set(req: DecodeSetRequest):
    sess = _get_session()
    updates = req.model_dump(exclude_none=True)
    for k, v in updates.items():
        if hasattr(sess.decode_settings, k):
            setattr(sess.decode_settings, k, v)
    return {"ok": True, "settings": sess.decode_settings.serialize()}


@app.get("/decode")
async def decode_show():
    sess = _get_session()
    return sess.decode_settings.serialize()


@app.delete("/decode")
async def decode_reset():
    sess = _get_session()
    from psplice.runtime.generation import DecodeSettings
    sess.decode_settings = DecodeSettings()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

@app.post("/preset/save")
async def preset_save(req: PresetSaveRequest):
    sess = _get_session()
    sess.save_preset(req.name)
    return {"ok": True, "name": req.name}


@app.post("/preset/load")
async def preset_load(req: PresetLoadRequest):
    sess = _get_session()
    try:
        errors = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: sess.load_preset(req.name),
        )
    except ModelSessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True, "name": req.name, "errors": errors}


@app.get("/preset/list")
async def preset_list_endpoint():
    from psplice.state.presets import list_presets
    return list_presets()


@app.post("/preset/clear")
async def preset_clear():
    sess = _get_session()
    sess.clear_interventions()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Startup / shutdown helpers
# ---------------------------------------------------------------------------

def _cleanup() -> None:
    from psplice.state.session import remove_session
    remove_session()
    logger.info("Session file removed.")


# ---------------------------------------------------------------------------
# Daemon entry point
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def run_daemon(
    model_id: str,
    port: Optional[int] = None,
    device: str = "auto",
    dtype: str = "auto",
    eager_attn: bool = False,
    trust_remote_code: bool = False,
) -> None:
    """
    Load model, build ModelSession, and start the uvicorn server.
    Called by daemon/manager.py in the spawned subprocess.
    """
    global _session

    from psplice.modeling.loader import LoadConfig, load_model
    from psplice.modeling.inspector import inspect_model
    from psplice.state.session import SessionMetadata, write_session

    if port is None:
        port = _find_free_port()

    pid = os.getpid()

    # Write session file before loading so CLI can poll the health endpoint
    session = SessionMetadata(
        pid=pid,
        port=port,
        model_id=model_id,
        device=device,
        dtype=dtype,
        eager_attn=eager_attn,
    )
    write_session(session)

    logger.info("Loading model: %s", model_id)
    cfg = LoadConfig(
        model_id=model_id,
        device=device,
        dtype=dtype,
        eager_attn=eager_attn,
        trust_remote_code=trust_remote_code,
    )

    try:
        loaded = load_model(cfg)
    except Exception as exc:
        from psplice.state.session import remove_session
        remove_session()
        logger.error("Model load failed: %s", exc)
        sys.exit(1)

    arch = inspect_model(loaded.model)

    _session = ModelSession(
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        arch=arch,
        model_id=loaded.model_id,
        device=loaded.device,
        dtype=loaded.dtype,
        eager_attn=loaded.eager_attn,
        param_count=loaded.param_count,
    )

    # Update session with resolved device/dtype
    session.device = loaded.device
    session.dtype = loaded.dtype
    write_session(session)

    logger.info(
        "Model loaded: %s  params=%.1fB  device=%s  dtype=%s",
        model_id,
        loaded.param_count / 1e9,
        loaded.device,
        loaded.dtype,
    )

    def _sigterm_handler(signum, frame):
        _cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    _cleanup()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="psplice daemon server")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--eager-attn", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    run_daemon(
        model_id=args.model_id,
        port=args.port,
        device=args.device,
        dtype=args.dtype,
        eager_attn=args.eager_attn,
        trust_remote_code=args.trust_remote_code,
    )
