"""
Procrustes alignment and null baselines for COSI.

Per `docs/research_notes/2026-05-06_cosi_design.md` §3.5–§3.6.

Given two activation matrices X_A (N x d_A) and X_B (N x d_B) where row i in
both matrices corresponds to the same prompt, we ask: is there an orthogonal
transformation that aligns them, and does the residual after optimal
alignment fall significantly below chance?

Definitions
-----------
- residual(X_A, X_B, R) = ||X_A - X_B R||_F / ||X_A||_F
- Procrustes optimal R minimizes residual subject to R^T R = I.
- Permutation null: shuffle B's row correspondence; recompute Procrustes
  residual on permuted data. The distribution of these residuals defines
  "no semantic alignment" while preserving each model's internal geometry.
- Random-rotation null: sample R uniformly from O(d), compute residual.
  This is the weaker null (no optimization at all) and is included for
  completeness.

The two matrices need not have the same dimensionality. We project both onto
a shared k-dimensional subspace via PCA before alignment. k is chosen so
that cumulative variance reaches a threshold (default 0.90) in *both*
models' projections, and we use min(k_A, k_B) so neither side dominates.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
from scipy.linalg import orthogonal_procrustes, svd


@dataclasses.dataclass(frozen=True)
class AlignmentResult:
    """Result of one Procrustes alignment.

    Attributes:
        residual: ||X_A_proj - X_B_proj @ R||_F / ||X_A_proj||_F
        R: optimal orthogonal matrix in O(k)
        k: shared subspace dimension
        var_explained_a: variance retained in A's projection
        var_explained_b: variance retained in B's projection
    """

    residual: float
    R: np.ndarray
    k: int
    var_explained_a: float
    var_explained_b: float


@dataclasses.dataclass(frozen=True)
class NullStats:
    mean: float
    std: float
    p1: float  # 1st percentile
    p5: float
    n_samples: int
    samples: np.ndarray  # all sampled residuals, for visualization


@dataclasses.dataclass(frozen=True)
class CosiResult:
    observed: AlignmentResult
    permutation_null: NullStats
    rotation_null: NullStats

    @property
    def z_vs_permutation(self) -> float:
        """Standard score of observed residual against permutation null."""
        return (self.observed.residual - self.permutation_null.mean) / self.permutation_null.std

    @property
    def below_p1_permutation(self) -> bool:
        """True iff observed residual is below the 1st percentile of permutation null."""
        return self.observed.residual < self.permutation_null.p1


# ---------------------------------------------------------------------------
# Subspace projection
# ---------------------------------------------------------------------------


def _pca_project(X: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Project X (N x d) onto top-k principal components. Returns (projected, var_explained)."""
    X = X - X.mean(axis=0, keepdims=True)
    # SVD: X = U S V^T. Top-k components = V[:, :k]. Projected = X V[:, :k] = U[:, :k] S[:k].
    U, S, Vt = svd(X, full_matrices=False)
    var = (S ** 2).sum()
    if var == 0:
        return X[:, :k], 0.0
    var_top_k = (S[:k] ** 2).sum()
    proj = U[:, :k] * S[:k]
    return proj, float(var_top_k / var)


def choose_shared_k(
    X_A: np.ndarray, X_B: np.ndarray, var_threshold: float = 0.90
) -> int:
    """Choose k as the smaller of the two models' min components for variance threshold."""
    def needed_k(X: np.ndarray) -> int:
        Xc = X - X.mean(axis=0, keepdims=True)
        _, S, _ = svd(Xc, full_matrices=False)
        if S.sum() == 0:
            return 1
        cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
        return int(np.searchsorted(cumvar, var_threshold) + 1)

    return max(1, min(needed_k(X_A), needed_k(X_B)))


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def procrustes_residual(
    X_A: np.ndarray,
    X_B: np.ndarray,
    k: Optional[int] = None,
    var_threshold: float = 0.90,
) -> AlignmentResult:
    """Compute Procrustes alignment of X_B to X_A in shared k-d subspace."""
    if X_A.shape[0] != X_B.shape[0]:
        raise ValueError(
            f"row count mismatch: X_A has {X_A.shape[0]}, X_B has {X_B.shape[0]}"
        )
    if k is None:
        k = choose_shared_k(X_A, X_B, var_threshold=var_threshold)

    A_proj, var_a = _pca_project(X_A, k)
    B_proj, var_b = _pca_project(X_B, k)

    # scipy's orthogonal_procrustes returns R such that B @ R ≈ A
    R, _scale = orthogonal_procrustes(B_proj, A_proj)
    aligned = B_proj @ R
    num = np.linalg.norm(A_proj - aligned, ord="fro")
    den = np.linalg.norm(A_proj, ord="fro")
    residual = float(num / den) if den > 0 else 1.0
    return AlignmentResult(
        residual=residual,
        R=R,
        k=k,
        var_explained_a=var_a,
        var_explained_b=var_b,
    )


# ---------------------------------------------------------------------------
# Null baselines
# ---------------------------------------------------------------------------


def permutation_null(
    X_A: np.ndarray,
    X_B: np.ndarray,
    n_samples: int = 1000,
    k: Optional[int] = None,
    seed: int = 0,
) -> NullStats:
    """For many random permutations of B's rows, compute Procrustes residual."""
    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        perm = rng.permutation(X_B.shape[0])
        result = procrustes_residual(X_A, X_B[perm], k=k)
        samples[i] = result.residual
    return _summarize(samples)


def random_rotation_null(
    X_A: np.ndarray,
    X_B: np.ndarray,
    n_samples: int = 1000,
    k: Optional[int] = None,
    seed: int = 1,
) -> NullStats:
    """Apply a random orthogonal matrix to B (in shared subspace) instead of optimal."""
    if k is None:
        k = choose_shared_k(X_A, X_B)
    A_proj, _ = _pca_project(X_A, k)
    B_proj, _ = _pca_project(X_B, k)
    den = np.linalg.norm(A_proj, ord="fro")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        R = _random_orthogonal(k, rng)
        aligned = B_proj @ R
        samples[i] = float(np.linalg.norm(A_proj - aligned, ord="fro") / den)
    return _summarize(samples)


def _random_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample R uniformly from O(k) via QR decomposition of a Gaussian matrix."""
    G = rng.standard_normal((k, k))
    Q, R = np.linalg.qr(G)
    # Adjust for QR sign convention so Q is uniformly distributed over O(k)
    d = np.sign(np.diag(R))
    d[d == 0] = 1
    return Q * d


def _summarize(samples: np.ndarray) -> NullStats:
    return NullStats(
        mean=float(samples.mean()),
        std=float(samples.std(ddof=1)),
        p1=float(np.percentile(samples, 1)),
        p5=float(np.percentile(samples, 5)),
        n_samples=int(samples.size),
        samples=samples,
    )


# ---------------------------------------------------------------------------
# Top-level COSI run
# ---------------------------------------------------------------------------


def run_cosi(
    X_A: np.ndarray,
    X_B: np.ndarray,
    n_null_samples: int = 1000,
    k: Optional[int] = None,
    var_threshold: float = 0.90,
    seed: int = 0,
) -> CosiResult:
    """Compute observed Procrustes residual plus both null baselines."""
    if k is None:
        k = choose_shared_k(X_A, X_B, var_threshold=var_threshold)
    obs = procrustes_residual(X_A, X_B, k=k)
    perm = permutation_null(X_A, X_B, n_samples=n_null_samples, k=k, seed=seed)
    rot = random_rotation_null(X_A, X_B, n_samples=n_null_samples, k=k, seed=seed + 1)
    return CosiResult(observed=obs, permutation_null=perm, rotation_null=rot)
