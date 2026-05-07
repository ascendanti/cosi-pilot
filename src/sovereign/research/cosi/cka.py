"""
Centered Kernel Alignment (CKA) for Study F of the COSI program.

Per Kornblith et al. (2019): CKA is a similarity measure between two
representation matrices that is invariant to orthogonal transformations and
isotropic scaling. It is a complementary metric to Procrustes residual:
where Procrustes finds the optimal orthogonal alignment, CKA measures
similarity in a way that does not commit to any particular alignment.

The COSI methods section commits to reporting CKA in parallel with Procrustes
as a cross-validation metric. The pilot did not satisfy that commitment;
this module fixes that.

We implement both linear CKA (the simpler form) and a permutation null for
each CKA value.
"""

from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class CKAResult:
    cka_observed: float
    perm_null_mean: float
    perm_null_std: float
    perm_null_p99: float  # 99th percentile (CKA is "high = similar"; null upper tail is the chance level)
    z_vs_perm: float
    above_p99_permutation: bool
    n_perm: int


def _center(K: np.ndarray) -> np.ndarray:
    """Center a Gram matrix K = X X^T."""
    N = K.shape[0]
    H = np.eye(N) - np.ones((N, N)) / N
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between row-aligned representation matrices.

    X: (N x d_X), Y: (N x d_Y). Returns scalar in [0, 1].

    Uses the closed-form: CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    after column-centering X and Y. (Equivalent to centered HSIC ratio.)
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"row count mismatch: {X.shape[0]} vs {Y.shape[0]}")
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = np.linalg.norm(Yc.T @ Xc, ord="fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    if den == 0:
        return 0.0
    return float(num / den)


def cka_with_permutation_null(
    X: np.ndarray,
    Y: np.ndarray,
    n_perm: int = 1000,
    seed: int = 0,
) -> CKAResult:
    """Compute observed linear CKA and a permutation null distribution."""
    cka_obs = linear_cka(X, Y)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(Y.shape[0])
        samples[i] = linear_cka(X, Y[perm])
    mean = float(samples.mean())
    std = float(samples.std(ddof=1))
    p99 = float(np.percentile(samples, 99))
    z = (cka_obs - mean) / std if std > 0 else 0.0
    return CKAResult(
        cka_observed=cka_obs,
        perm_null_mean=mean,
        perm_null_std=std,
        perm_null_p99=p99,
        z_vs_perm=z,
        above_p99_permutation=cka_obs > p99,
        n_perm=n_perm,
    )
