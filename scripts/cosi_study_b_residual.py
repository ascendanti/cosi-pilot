#!/usr/bin/env python
"""Study B' (lexical-residual): the experiment Adam's review identifies as
the load-bearing test of the operator-theoretic interpretation.

For each LLM activation matrix X (N x d), regress out lexical/template
features L (N x V) via ridge regression:

    B = (L^T L + lambda I)^-1 L^T X     (V x d coefficient matrix)
    X_hat = L B                          (N x d predicted)
    X_resid = X - X_hat                  (N x d residual)

Then run the same Procrustes-against-permutation-null pipeline on the
residualized activations of the two LLMs. Compare the resulting residual
to the original (un-residualized) Procrustes residual.

Interpretation of the result:
  - If the residualized Procrustes residual is comparable to the
    original (~0.77 at layer 0.25), then there is a substantial
    cross-model alignment that survives lexical/template removal. This
    would be partial support for the operator-theoretic interpretation.
  - If the residualized residual is much higher (closer to 1.0), then
    the original alignment was largely captured by lexical/template
    structure. This would be the negative methodological result the
    paper should foreground.

Output: runs/cosi/study_b_residual_<timestamp>/results.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sovereign.research.cosi.align import run_cosi  # noqa: E402
from sovereign.research.cosi.cka import cka_with_permutation_null  # noqa: E402

PHASE1_RUN_DIR = REPO_ROOT / "runs" / "cosi"
PROBE_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v1.jsonl"
RUN_DIR = REPO_ROOT / "runs" / "cosi"

LAYER_FRACS = (0.25, 0.5, 0.75)
RIDGE_LAMBDA = 0.01
N_PERM_PROC = 500
N_PERM_CKA = 500

STOPWORDS = set("""a an and are as at be by for from has have he in is it its
of on or that the to was were will with this these those if then which what
do does did how when where why""".split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def template_only_tokens(prompt: str) -> list[str]:
    return [t if t in STOPWORDS else "<CONTENT>" for t in tokenize(prompt)]


def build_tfidf(corpus: list[list[str]]) -> np.ndarray:
    df: Counter = Counter()
    for tokens in corpus:
        for t in set(tokens):
            df[t] += 1
    vocab = sorted(df.keys())
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    N = len(corpus)
    V = len(vocab)
    M = np.zeros((N, V), dtype=np.float32)
    for i, tokens in enumerate(corpus):
        tf = Counter(tokens)
        for t, c in tf.items():
            j = vocab_idx[t]
            M[i, j] = c / max(1, len(tokens))
    idf = np.log((1 + N) / (1 + np.array([df[t] for t in vocab], dtype=np.float32))) + 1
    M *= idf[None, :]
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return M / norms


def latest_phase1() -> Path:
    candidates = sorted(p for p in PHASE1_RUN_DIR.glob("phase1_*") if p.is_dir())
    if not candidates:
        raise SystemExit("no phase1 run dir found")
    return candidates[-1]


def ridge_residualize(X: np.ndarray, L: np.ndarray, lam: float) -> tuple[np.ndarray, dict]:
    """Compute X_resid = X - L * (L^T L + lam I)^-1 L^T X, plus diagnostics.

    Returns (X_resid, info). info contains:
      - r2_per_dim: R^2 per output dimension (mean, min, max)
      - frac_variance_removed: fraction of total variance in X removed by L
    """
    Lc = L - L.mean(axis=0, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    V = Lc.shape[1]
    G = Lc.T @ Lc + lam * np.eye(V, dtype=Lc.dtype)
    B = np.linalg.solve(G, Lc.T @ Xc)  # (V, d)
    X_hat = Lc @ B  # (N, d)
    X_resid = Xc - X_hat
    # Diagnostics
    var_X = (Xc ** 2).sum()
    var_resid = (X_resid ** 2).sum()
    frac_removed = 1.0 - var_resid / var_X if var_X > 0 else 0.0
    # Per-dimension R^2
    var_per_dim = (Xc ** 2).sum(axis=0)
    var_resid_per_dim = (X_resid ** 2).sum(axis=0)
    valid = var_per_dim > 0
    r2_per_dim = np.zeros_like(var_per_dim)
    r2_per_dim[valid] = 1.0 - var_resid_per_dim[valid] / var_per_dim[valid]
    return X_resid.astype(np.float32), {
        "frac_variance_removed": float(frac_removed),
        "r2_mean": float(r2_per_dim.mean()),
        "r2_median": float(np.median(r2_per_dim)),
        "r2_min": float(r2_per_dim.min()),
        "r2_max": float(r2_per_dim.max()),
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_DIR / f"study_b_residual_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("=== COSI Study B' (lexical-residual) ===", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    probes = []
    with open(PROBE_PATH) as f:
        for line in f:
            probes.append(json.loads(line))
    prompts = [p["prompt"] for p in probes]
    N = len(prompts)
    print(f"probes: {N}", flush=True)

    word_tokens = [tokenize(p) for p in prompts]
    template_tokens = [template_only_tokens(p) for p in prompts]
    X_tfidf = build_tfidf(word_tokens)
    X_template = build_tfidf(template_tokens)
    L = np.hstack([X_tfidf, X_template]).astype(np.float32)
    print(f"lexical features L: {L.shape}", flush=True)

    phase1_dir = latest_phase1()
    label_a = "Phi-4-Reasoning-Plus"
    label_b = "Qwen3-Next-80B-Thinking-5bit"
    npz_a = np.load(phase1_dir / f"activations_{label_a}.npz")
    npz_b = np.load(phase1_dir / f"activations_{label_b}.npz")

    results: dict = {
        "timestamp": timestamp,
        "n_probes": N,
        "ridge_lambda": RIDGE_LAMBDA,
        "lexical_feature_dim": L.shape[1],
        "model_a": label_a,
        "model_b": label_b,
        "by_layer": {},
    }

    for frac in LAYER_FRACS:
        key = f"frac_{frac}"
        X_A = npz_a[key]
        X_B = npz_b[key]
        print(f"\n--- layer fraction {frac} ---", flush=True)
        print(f"  X_A: {X_A.shape}, X_B: {X_B.shape}", flush=True)

        # Original alignment
        t0 = time.time()
        cr_orig = run_cosi(X_A, X_B, n_null_samples=N_PERM_PROC, seed=int(frac * 100))
        cka_orig = cka_with_permutation_null(X_A, X_B, n_perm=N_PERM_CKA, seed=int(frac * 100))
        elapsed_orig = time.time() - t0
        print(
            f"  ORIGINAL: Procrustes residual={cr_orig.observed.residual:.4f} "
            f"(z={cr_orig.z_vs_permutation:.2f}), CKA={cka_orig.cka_observed:.4f} "
            f"(z={cka_orig.z_vs_perm:.2f}) ({elapsed_orig:.1f}s)",
            flush=True,
        )

        # Residualize
        t0 = time.time()
        X_A_resid, info_a = ridge_residualize(X_A, L, RIDGE_LAMBDA)
        X_B_resid, info_b = ridge_residualize(X_B, L, RIDGE_LAMBDA)
        elapsed_resid = time.time() - t0
        print(
            f"  ridge regression: X_A removed {info_a['frac_variance_removed']:.4f} of variance "
            f"(mean R^2={info_a['r2_mean']:.4f}); X_B removed "
            f"{info_b['frac_variance_removed']:.4f} (mean R^2={info_b['r2_mean']:.4f}) "
            f"({elapsed_resid:.1f}s)",
            flush=True,
        )

        # Residualized alignment
        t0 = time.time()
        cr_resid = run_cosi(X_A_resid, X_B_resid, n_null_samples=N_PERM_PROC, seed=int(frac * 100) + 1)
        cka_resid = cka_with_permutation_null(X_A_resid, X_B_resid, n_perm=N_PERM_CKA, seed=int(frac * 100) + 1)
        elapsed_align = time.time() - t0
        print(
            f"  RESIDUALIZED: Procrustes residual={cr_resid.observed.residual:.4f} "
            f"(z={cr_resid.z_vs_permutation:.2f}), CKA={cka_resid.cka_observed:.4f} "
            f"(z={cka_resid.z_vs_perm:.2f}) ({elapsed_align:.1f}s)",
            flush=True,
        )

        delta_proc = cr_resid.observed.residual - cr_orig.observed.residual
        delta_cka = cka_resid.cka_observed - cka_orig.cka_observed
        print(
            f"  DELTA: Procrustes Δ={delta_proc:+.4f} "
            f"(higher = alignment weakened by removing lexical), "
            f"CKA Δ={delta_cka:+.4f} (lower = similarity weakened)",
            flush=True,
        )

        results["by_layer"][str(frac)] = {
            "original": {
                "procrustes_residual": cr_orig.observed.residual,
                "procrustes_z": cr_orig.z_vs_permutation,
                "procrustes_below_p1": cr_orig.below_p1_permutation,
                "cka_observed": cka_orig.cka_observed,
                "cka_z": cka_orig.z_vs_perm,
                "cka_above_p99": cka_orig.above_p99_permutation,
            },
            "residualized": {
                "procrustes_residual": cr_resid.observed.residual,
                "procrustes_z": cr_resid.z_vs_permutation,
                "procrustes_below_p1": cr_resid.below_p1_permutation,
                "cka_observed": cka_resid.cka_observed,
                "cka_z": cka_resid.z_vs_perm,
                "cka_above_p99": cka_resid.above_p99_permutation,
            },
            "ridge_diagnostics": {
                "model_a": info_a,
                "model_b": info_b,
            },
            "deltas": {
                "procrustes": delta_proc,
                "cka": delta_cka,
            },
        }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}", flush=True)

    print("\n=== STUDY B' SUMMARY ===", flush=True)
    print("Question: does the LLM-LLM alignment survive removal of lexical/template features?", flush=True)
    for frac, r in results["by_layer"].items():
        d_p = r["deltas"]["procrustes"]
        d_c = r["deltas"]["cka"]
        if abs(d_p) < 0.05 and abs(d_c) < 0.10:
            verdict = "ALIGNMENT SURVIVES (mostly NOT lexical)"
        elif d_p > 0.15 or d_c < -0.30:
            verdict = "ALIGNMENT COLLAPSES (mostly lexical)"
        else:
            verdict = "PARTIAL: alignment partly explained by lexical structure"
        print(
            f"  layer_frac={frac}: original Proc={r['original']['procrustes_residual']:.4f} "
            f"-> residualized={r['residualized']['procrustes_residual']:.4f} (Δ={d_p:+.4f}), "
            f"original CKA={r['original']['cka_observed']:.4f} -> "
            f"residualized={r['residualized']['cka_observed']:.4f} (Δ={d_c:+.4f}) → {verdict}",
            flush=True,
        )


if __name__ == "__main__":
    main()
