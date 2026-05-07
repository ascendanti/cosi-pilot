"""
Activation extraction for the Cross-Operator Subspace Isomorphism (COSI) experiment.

Per `docs/research_notes/2026-05-06_cosi_design.md` §3.3, we extract residual
stream activations at specified layer fractions from two MLX models with
different architectures (Phi-4-Reasoning-Plus, Qwen3-Next-80B-A3B-Thinking) and
pool them to a single vector per (card, model, layer_fraction).

Design notes
------------
- We do NOT monkey-patch mlx_lm. We replicate the inner-model forward pass
  with the same logic as `Phi3Model.__call__` and `Qwen3NextModel.__call__`,
  capturing the residual stream at requested layer indices.
- Pooling is configurable. Default is last-token; alternatives (mean, max)
  are sweeps per §5 of the research note.
- Architecture dispatch is explicit, not auto-detected. Adding a new
  architecture means writing a new function. This is preferable to fragile
  introspection.
- We work with float32 numpy arrays after extraction; MLX arrays stay in MLX
  during forward to keep the device path clean, then cross to host once.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterable, Sequence

import mlx.core as mx
import numpy as np


PoolingMode = str  # "last" | "mean" | "max"


@dataclasses.dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for an activation extraction run.

    layer_fractions: which fractions of the layer stack to capture, e.g.
        (0.25, 0.5, 0.75) means capture the residual stream after the layer
        at index round(L * f) - 1 for f in fractions, where L is total layers.
        We use round-down (max(0, round(L*f)-1)) so 1.0 maps to the last block.
    pooling: how to reduce sequence dimension to a single vector.
    dtype: numpy dtype for stored activations (float16 saves disk; float32 is
        safer for numerical alignment).
    """

    layer_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    pooling: PoolingMode = "last"
    dtype: str = "float32"


def _layer_indices(num_layers: int, fractions: Sequence[float]) -> list[int]:
    """Map fractions to concrete layer indices (0-based, inclusive)."""
    out: list[int] = []
    for f in fractions:
        if not 0.0 < f <= 1.0:
            raise ValueError(f"layer fraction must be in (0, 1], got {f}")
        idx = max(0, int(round(num_layers * f)) - 1)
        out.append(idx)
    return out


def _pool(seq: mx.array, mode: PoolingMode) -> mx.array:
    """Pool over the sequence dimension. Input shape: (1, T, H). Output: (H,)."""
    assert seq.ndim == 3 and seq.shape[0] == 1, f"expected (1,T,H), got {seq.shape}"
    if mode == "last":
        return seq[0, -1, :]
    if mode == "mean":
        return seq[0].mean(axis=0)
    if mode == "max":
        return seq[0].max(axis=0)
    raise ValueError(f"unknown pooling mode: {mode}")


# ---------------------------------------------------------------------------
# Architecture-specific forward passes
# ---------------------------------------------------------------------------


def _forward_phi3_capture(
    inner_model: Any,
    input_ids: mx.array,
    capture_at: set[int],
) -> dict[int, mx.array]:
    """Replicate Phi3Model.__call__ and capture h after specified layer indices.

    See `mlx_lm.models.phi3.Phi3Model.__call__`. The residual stream is `h`
    after each `layer(h, mask, c)` call. We capture *after* the layer block,
    which means index 0 = output of layer 0.
    """
    from mlx_lm.models.base import create_attention_mask

    h = inner_model.embed_tokens(input_ids)
    cache = [None] * len(inner_model.layers)
    mask = create_attention_mask(h, cache[0])

    captured: dict[int, mx.array] = {}
    for i, (layer, c) in enumerate(zip(inner_model.layers, cache)):
        h = layer(h, mask, c)
        if i in capture_at:
            captured[i] = h
    return captured


def _forward_qwen3_next_capture(
    inner_model: Any,
    input_ids: mx.array,
    capture_at: set[int],
) -> dict[int, mx.array]:
    """Replicate Qwen3NextModel.__call__ with hybrid attention/SSM masking.

    See `mlx_lm.models.qwen3_next.Qwen3NextModel.__call__`. The architecture
    interleaves linear (SSM) and full-attention layers; each layer signals
    which mask to use via `layer.is_linear`.
    """
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.qwen3_next import create_ssm_mask

    hidden = inner_model.embed_tokens(input_ids)
    cache = [None] * len(inner_model.layers)
    fa_mask = create_attention_mask(hidden, cache[inner_model.fa_idx])
    ssm_mask = create_ssm_mask(hidden, cache[inner_model.ssm_idx])

    captured: dict[int, mx.array] = {}
    for i, (layer, c) in enumerate(zip(inner_model.layers, cache)):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden = layer(hidden, mask=mask, cache=c)
        if i in capture_at:
            captured[i] = hidden
    return captured


_DISPATCH: dict[str, Callable[[Any, mx.array, set[int]], dict[int, mx.array]]] = {
    "phi3": _forward_phi3_capture,
    "phi": _forward_phi3_capture,  # phi (Phi-4 family routes through phi3 in mlx_lm 0.31)
    # Qwen3 (non-next) shares the same forward shape as Phi3:
    #   for layer, c in zip(self.layers, cache): h = layer(h, mask, c)
    # The mask is also created via create_attention_mask. Reusing the Phi3
    # capture function is correct.
    "qwen3": _forward_phi3_capture,
    "qwen3_next": _forward_qwen3_next_capture,
}


def _detect_architecture(model: Any) -> str:
    """Detect architecture from the model's module path."""
    mod = type(model).__module__  # e.g. 'mlx_lm.models.phi3'
    arch = mod.rsplit(".", 1)[-1]
    if arch not in _DISPATCH:
        raise NotImplementedError(
            f"COSI extraction not implemented for architecture {arch!r}. "
            f"Add a forward-pass function in extract.py and register in _DISPATCH."
        )
    return arch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_residual(
    model: Any,
    tokenizer: Any,
    prompts: Iterable[str],
    config: ExtractionConfig = ExtractionConfig(),
) -> dict[float, np.ndarray]:
    """Extract pooled residual stream activations for a batch of prompts.

    Returns a dict mapping layer_fraction -> array of shape (N, H), where N is
    the number of prompts and H is the model's hidden size.
    """
    arch = _detect_architecture(model)
    forward = _DISPATCH[arch]
    inner = model.model  # mlx_lm wraps the inner model in `Model.model`
    num_layers = len(inner.layers)
    indices = _layer_indices(num_layers, config.layer_fractions)
    capture_set = set(indices)
    frac_to_idx = dict(zip(config.layer_fractions, indices))

    out: dict[float, list[np.ndarray]] = {f: [] for f in config.layer_fractions}

    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        input_ids = mx.array(ids).reshape(1, -1)

        captured = forward(inner, input_ids, capture_set)
        # Force evaluation before pooling (MLX is lazy)
        for v in captured.values():
            mx.eval(v)

        for frac, idx in frac_to_idx.items():
            pooled = _pool(captured[idx], config.pooling)
            # Cast to float32 before numpy crossing: MLX uses bfloat16 by
            # default for inference, and numpy has no native bfloat16, which
            # produces a PEP 3118 buffer-format error on direct conversion.
            pooled_f32 = pooled.astype(mx.float32)
            mx.eval(pooled_f32)
            arr = np.asarray(pooled_f32, dtype=config.dtype)
            out[frac].append(arr)

    return {f: np.stack(v, axis=0) for f, v in out.items()}


def model_fingerprint(model: Any) -> dict[str, Any]:
    """Return identifying metadata for manifest/reproducibility.

    Note: with quantized embeddings, ``embed_tokens.weight.shape`` reports the
    packed storage shape rather than the logical (vocab, hidden) shape. We
    read the logical shape from ``model.args`` (the dataclass populated from
    config.json) which is the source of truth.
    """
    inner = model.model
    args = getattr(model, "args", None) or getattr(inner, "args", None)
    if args is None:
        raise RuntimeError("could not locate ModelArgs on model")
    return {
        "architecture": _detect_architecture(model),
        "num_layers": int(getattr(args, "num_hidden_layers", len(inner.layers))),
        "hidden_size": int(getattr(args, "hidden_size")),
        "vocab_size": int(getattr(args, "vocab_size")),
    }
