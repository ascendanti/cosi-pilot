"""
Invariance-leakage measurement (Study A of the COSI program).

The COSI paper defines a $k$-dimensional subspace $V \subseteq \R^d$ as
approximately $\epsilon$-invariant under a (possibly nonlinear) map $f$ on a
sample $X$ if

    || (I - P_V) f(P_V X) ||_F  /  || f(P_V X) ||_F  <=  epsilon.

Here we compute exactly this statistic for a candidate subspace V at layer L
of a model, where:

    - V is the span of the top-k principal components of the pooled
      activations at layer L (matching the subspace used for Procrustes
      alignment in extract.py + align.py).
    - f is the next-layer forward map (i.e., layer L+1 of the inner model),
      including its mask construction and any state-space components.
    - The projection acts at the token level on the full sequence
      (1, T, d), not on pooled vectors, because f expects sequence input.
    - We aggregate the leakage statistic across the test sample.

Two summary statistics are reported:

    leakage_token_mean: mean over tokens (and prompts) of the
        per-token Frobenius ratio.
    leakage_sample:     ||(I-P_V) f(P_V X)||_F / ||f(P_V X)||_F computed
        on the entire (N*T, d) tensor at once.

Low leakage (close to 0) means V is approximately operator-invariant on this
sample. High leakage (close to 1) means the next-layer forward map scatters
projected activations substantially out of V.

The reference baseline is the leakage of a *random* k-dimensional subspace
of R^d. If the PCA-derived subspace's leakage is comparable to a random
subspace's leakage, then the PCA subspace is no more operator-invariant
than chance.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Iterable

import mlx.core as mx
import numpy as np

from .extract import (
    _DISPATCH,  # noqa
    _detect_architecture,
    _layer_indices,
    _pool,
)


@dataclasses.dataclass(frozen=True)
class LeakageResult:
    """Results of a leakage measurement for one (layer, k) cell."""

    layer_fraction: float
    layer_index: int
    k: int
    n_prompts: int
    leakage_pca_token_mean: float
    leakage_pca_sample: float
    leakage_random_token_mean_mean: float
    leakage_random_token_mean_std: float
    leakage_random_sample_mean: float
    leakage_random_sample_std: float
    n_random_subspaces: int


def _project_with_subspace(h: mx.array, U: mx.array) -> mx.array:
    """Project a (1, T, d) token-sequence through the subspace V = span(U).

    U has shape (d, k); P_V = U U^T; result has shape (1, T, d).
    """
    assert h.ndim == 3 and h.shape[0] == 1
    # h: (1, T, d). Reshape to (T, d) for matmul, then back.
    T, d = h.shape[1], h.shape[2]
    h2d = h.reshape(T, d)
    # U: (d, k). h2d @ U: (T, k). Then @ U^T: (T, d).
    coords = h2d @ U  # (T, k)
    proj = coords @ U.T  # (T, d)
    return proj.reshape(1, T, d)


def _orthogonal_random(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return a (d, k) matrix whose columns are orthonormal in R^d."""
    G = rng.standard_normal((d, k)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return Q  # (d, k)


def _next_layer_forward_phi3(inner_model: Any, h_proj: mx.array, layer_index: int) -> mx.array:
    """Run layer (layer_index + 1) of a phi3-style inner model on projected h."""
    from mlx_lm.models.base import create_attention_mask

    next_idx = layer_index + 1
    if next_idx >= len(inner_model.layers):
        # No "next layer"; use the final norm as the forward map for the last layer.
        return inner_model.norm(h_proj)
    mask = create_attention_mask(h_proj, None)
    return inner_model.layers[next_idx](h_proj, mask, None)


def _next_layer_forward_qwen3_next(inner_model: Any, h_proj: mx.array, layer_index: int) -> mx.array:
    """Run layer (layer_index + 1) of a qwen3_next inner model on projected h."""
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.qwen3_next import create_ssm_mask

    next_idx = layer_index + 1
    if next_idx >= len(inner_model.layers):
        return inner_model.norm(h_proj)
    layer = inner_model.layers[next_idx]
    fa_mask = create_attention_mask(h_proj, None)
    ssm_mask = create_ssm_mask(h_proj, None)
    mask = ssm_mask if layer.is_linear else fa_mask
    return layer(h_proj, mask=mask, cache=None)


_NEXT_LAYER_DISPATCH: dict[str, Callable[[Any, mx.array, int], mx.array]] = {
    "phi3": _next_layer_forward_phi3,
    "phi": _next_layer_forward_phi3,
    "qwen3": _next_layer_forward_phi3,  # same call shape
    "qwen3_next": _next_layer_forward_qwen3_next,
}


def _capture_layer_output(
    inner_model: Any, input_ids: mx.array, layer_index: int, arch: str
) -> mx.array:
    """Run forward pass up to and including layer L; return h^(L) (1, T, d)."""
    from .extract import _forward_phi3_capture, _forward_qwen3_next_capture

    if arch == "qwen3_next":
        captured = _forward_qwen3_next_capture(inner_model, input_ids, {layer_index})
    else:
        captured = _forward_phi3_capture(inner_model, input_ids, {layer_index})
    return captured[layer_index]


def measure_leakage(
    model: Any,
    tokenizer: Any,
    prompts: Iterable[str],
    layer_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    pooling_for_pca: str = "last",
    pca_variance_threshold: float = 0.90,
    n_random_subspaces: int = 30,
    rng_seed: int = 20260507,
) -> list[LeakageResult]:
    """Compute leakage for the PCA-derived subspace (and random baselines) at each layer fraction.

    For each layer fraction f and corresponding layer index L:

      1. Extract h^(L)(X) for all prompts (full sequences, then pool by
         pooling_for_pca to get one vector per prompt).
      2. Compute top-k PCA components of pooled activations such that
         cumulative variance >= pca_variance_threshold.
      3. For each prompt, take its full sequence h^(L)(X_i), project all
         tokens through V = span(U_pca), pass through next-layer forward.
      4. Compute the per-prompt Frobenius leakage ratio.
      5. Aggregate: per-token mean (averaged over prompts), and the global
         sample-level ratio computed on the concatenated tensor.
      6. Repeat with n_random_subspaces random orthonormal subspaces of
         dimension k as the chance baseline.
    """
    arch = _detect_architecture(model)
    next_forward = _NEXT_LAYER_DISPATCH[arch]
    inner = model.model
    num_layers = len(inner.layers)
    layer_indices = _layer_indices(num_layers, layer_fractions)
    rng = np.random.default_rng(rng_seed)

    prompts_list = list(prompts)
    encoded = [mx.array(tokenizer.encode(p)).reshape(1, -1) for p in prompts_list]

    results: list[LeakageResult] = []

    for frac, layer_idx in zip(layer_fractions, layer_indices):
        # Step 1+2: capture h^(L) for each prompt and PCA on pooled.
        h_full_list = [_capture_layer_output(inner, ids, layer_idx, arch) for ids in encoded]
        # Force eval and cast to float32 for numpy
        for h in h_full_list:
            mx.eval(h)
        pooled_arrays = []
        for h in h_full_list:
            p = _pool(h, pooling_for_pca).astype(mx.float32)
            mx.eval(p)
            pooled_arrays.append(np.asarray(p, dtype=np.float32))
        pooled = np.stack(pooled_arrays, axis=0)  # (N, d)
        N, d = pooled.shape

        # PCA via SVD on column-centered
        centered = pooled - pooled.mean(axis=0, keepdims=True)
        U_full, S_full, _ = np.linalg.svd(centered, full_matrices=False)
        var_total = (S_full ** 2).sum()
        cumvar = np.cumsum(S_full ** 2) / var_total if var_total > 0 else np.array([1.0])
        k = int(np.searchsorted(cumvar, pca_variance_threshold) + 1)
        k = max(1, min(k, len(S_full)))
        # Top-k principal directions in R^d come from V^T of SVD; equivalently,
        # the right singular vectors. Build them directly from PCA on (N x d):
        #   centered = U S V^T  where V is (d, d) when full_matrices=False is N>=d
        # Here N may be < d, so V_full has shape (rank, d). We need top-k
        # right singular directions = V_full[:k, :].T  (d, k).
        # Use the eigendecomposition of centered.T @ centered for stability.
        cov = centered.T @ centered  # (d, d) — large for d=5120 but tractable
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Top-k by eigenvalue
        order = np.argsort(eigvals)[::-1]
        U_pca = eigvecs[:, order[:k]].astype(np.float32)  # (d, k)

        # Step 3-5: compute leakage for PCA subspace
        leak_pca_token_means = []
        num_total = 0.0
        denom_total = 0.0
        for h in h_full_list:
            U_mx = mx.array(U_pca)  # (d, k)
            h_proj = _project_with_subspace(h.astype(mx.float32), U_mx)
            h_next = next_forward(inner, h_proj, layer_idx).astype(mx.float32)
            # Compute orthogonal residual (I - P_V) h_next
            # Easier: residual = h_next - (h_next @ U) @ U^T
            T = h_next.shape[1]
            h_next_2d = h_next.reshape(T, d)
            coords = h_next_2d @ U_mx  # (T, k)
            proj_back = coords @ U_mx.T  # (T, d)
            residual = h_next_2d - proj_back  # (T, d)
            num = float(mx.linalg.norm(residual, ord="fro").item())
            denom = float(mx.linalg.norm(h_next_2d, ord="fro").item())
            if denom > 0:
                leak_pca_token_means.append(num / denom)
            num_total += num ** 2
            denom_total += denom ** 2
        leakage_pca_token_mean = float(np.mean(leak_pca_token_means)) if leak_pca_token_means else 0.0
        leakage_pca_sample = float(np.sqrt(num_total / denom_total)) if denom_total > 0 else 0.0

        # Step 6: random-subspace baseline
        rand_token_means = []
        rand_sample_ratios = []
        for r in range(n_random_subspaces):
            U_rand_np = _orthogonal_random(d, k, rng)
            U_rand_mx = mx.array(U_rand_np)
            num_total_r = 0.0
            denom_total_r = 0.0
            this_token_means = []
            for h in h_full_list:
                h_proj = _project_with_subspace(h.astype(mx.float32), U_rand_mx)
                h_next = next_forward(inner, h_proj, layer_idx).astype(mx.float32)
                T = h_next.shape[1]
                h_next_2d = h_next.reshape(T, d)
                coords = h_next_2d @ U_rand_mx
                proj_back = coords @ U_rand_mx.T
                residual = h_next_2d - proj_back
                num = float(mx.linalg.norm(residual, ord="fro").item())
                denom = float(mx.linalg.norm(h_next_2d, ord="fro").item())
                if denom > 0:
                    this_token_means.append(num / denom)
                num_total_r += num ** 2
                denom_total_r += denom ** 2
            rand_token_means.append(float(np.mean(this_token_means)))
            rand_sample_ratios.append(
                float(np.sqrt(num_total_r / denom_total_r)) if denom_total_r > 0 else 0.0
            )

        results.append(
            LeakageResult(
                layer_fraction=frac,
                layer_index=layer_idx,
                k=k,
                n_prompts=N,
                leakage_pca_token_mean=leakage_pca_token_mean,
                leakage_pca_sample=leakage_pca_sample,
                leakage_random_token_mean_mean=float(np.mean(rand_token_means)),
                leakage_random_token_mean_std=float(np.std(rand_token_means, ddof=1) if len(rand_token_means) > 1 else 0.0),
                leakage_random_sample_mean=float(np.mean(rand_sample_ratios)),
                leakage_random_sample_std=float(np.std(rand_sample_ratios, ddof=1) if len(rand_sample_ratios) > 1 else 0.0),
                n_random_subspaces=n_random_subspaces,
            )
        )

    return results
