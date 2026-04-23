"""
Contrastive activation vector extraction.

Computes a steering vector by running two sets of prompts through the model
and taking the mean activation difference at specified decoder layers:

    vector[layer] = mean(activations | positive_prompts)
                  - mean(activations | negative_prompts)

The result is a per-layer dict {layer_idx: 1D tensor of shape [hidden_size]}
that can be saved directly as a .pt file and used with `psplice steer add`.

Method
------
For each prompt, we collect the hidden states at the output of each target
decoder layer (i.e., the residual stream at that depth).  Hidden states are
pooled across the sequence dimension — by default with a mean pool, or
optionally taking only the last non-padding token.

This is the standard contrastive activation extraction approach from the
activation steering literature (Turner et al. 2023, Zou et al. 2023).
The resulting vector points in the direction of the positive concept in
activation space.

Practical notes
---------------
* The vectors are extracted with no_grad and with hooks removed afterwards.
* Token pooling defaults to mean across the full sequence.  For instruction-
  tuned models you may get cleaner vectors by using last-token pooling
  (the final token carries the most "summarised" representation).
* Using more prompts (10–30 per side) produces more stable vectors.
  Two or three prompts still produce something usable for quick experiments.
* Vectors are returned as float32 regardless of the model's working dtype.
  The steering hook casts them on the fly.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch

logger = logging.getLogger(__name__)

TokenAggregation = Literal["mean", "last"]


def extract_contrastive_vector(
    model,
    tokenizer,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layer_indices: list[int],
    token_aggregation: TokenAggregation = "mean",
) -> dict[int, torch.Tensor]:
    """
    Extract a contrastive steering vector from the resident model.

    Parameters
    ----------
    model:
        The loaded HuggingFace causal LM.
    tokenizer:
        The matching tokenizer.
    positive_prompts:
        Prompts representing the behaviour / concept to steer *toward*.
    negative_prompts:
        Prompts representing the opposite behaviour / concept.
    layer_indices:
        Decoder layer indices at which to collect activations.
    token_aggregation:
        How to pool the sequence dimension.
        "mean" — average over all token positions (default).
        "last" — use only the last token (good for instruction-tuned models).

    Returns
    -------
    dict[int, torch.Tensor]
        Maps each layer index to a 1D float32 tensor of shape [hidden_size].
        This format is directly compatible with SteeringIntervention's
        per-layer tensor format and can be saved with torch.save().
    """
    if not positive_prompts:
        raise ValueError("Need at least one positive prompt.")
    if not negative_prompts:
        raise ValueError("Need at least one negative prompt.")

    layer_indices = sorted(set(layer_indices))
    device = _resolve_device(model)

    # ---------------------------------------------------------------------------
    # Collect hidden states for one batch of prompts via forward hooks
    # ---------------------------------------------------------------------------

    def _collect(prompts: list[str]) -> dict[int, torch.Tensor]:
        """Run *prompts* through the model, return mean activation per layer."""
        # per_layer_vecs[layer_idx] = list of [hidden_size] tensors, one per prompt
        per_layer_vecs: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}
        captured: dict[int, torch.Tensor | None] = {i: None for i in layer_indices}

        def make_hook(layer_idx: int):
            def hook(module, inputs, output):
                # Decoder layer output is (hidden_states, ...) tuple
                h = output[0] if isinstance(output, tuple) else output
                h = h.detach().float()   # [batch=1, seq_len, hidden_size]
                if token_aggregation == "mean":
                    captured[layer_idx] = h.squeeze(0).mean(dim=0)  # [hidden]
                else:  # "last"
                    captured[layer_idx] = h.squeeze(0)[-1]           # [hidden]
            return hook

        handles = []
        try:
            for idx in layer_indices:
                layer = model.model.layers[idx]
                handles.append(layer.register_forward_hook(make_hook(idx)))

            for prompt in prompts:
                # Reset captures
                for idx in layer_indices:
                    captured[idx] = None

                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    model(**inputs)

                for idx in layer_indices:
                    if captured[idx] is not None:
                        per_layer_vecs[idx].append(captured[idx].cpu())
                    else:
                        logger.warning("No activation captured for layer %d on prompt: %r", idx, prompt[:40])

        finally:
            for h in handles:
                h.remove()

        # Average across prompts
        result: dict[int, torch.Tensor] = {}
        for idx in layer_indices:
            vecs = per_layer_vecs[idx]
            if not vecs:
                raise RuntimeError(
                    f"No activations captured for layer {idx}. "
                    f"Check that the model architecture exposes model.model.layers."
                )
            result[idx] = torch.stack(vecs).mean(dim=0)  # [hidden_size]
        return result

    logger.info(
        "Extracting activations: %d positive, %d negative prompts, layers %s",
        len(positive_prompts),
        len(negative_prompts),
        layer_indices,
    )

    pos_means = _collect(positive_prompts)
    neg_means = _collect(negative_prompts)

    # Contrastive difference
    vectors: dict[int, torch.Tensor] = {}
    for idx in layer_indices:
        vectors[idx] = pos_means[idx] - neg_means[idx]

    hidden_size = next(iter(vectors.values())).shape[0]
    logger.info("Extraction complete. hidden_size=%d layers=%s", hidden_size, layer_indices)

    return vectors


def save_vector(vectors: dict[int, torch.Tensor], output_path: str) -> None:
    """
    Save extraction result to a .pt file.

    If all layer vectors are identical (e.g. extraction was called with a
    single layer and the caller wants the uniform format), saves a 1D tensor.
    Otherwise saves the full per-layer dict.  Both formats are accepted by
    SteeringIntervention.
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if len(vectors) == 1:
        # Single layer — save as a plain 1D tensor for simplicity
        tensor = next(iter(vectors.values()))
        torch.save(tensor, output_path)
    else:
        torch.save(vectors, output_path)


def _resolve_device(model) -> str:
    """Return the device of the first model parameter."""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"
