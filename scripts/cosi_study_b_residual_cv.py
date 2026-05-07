#!/usr/bin/env python
"""Study B'' (cross-validated lexical-residual): the over-fitting control
for Study B'.

Study B' regressed LLM activations on 455 lexical/template features and
showed 94-97% of variance was removable. A reviewer concern: at N=600 with
V=455 the in-sample fit could be partly overfitting, exaggerating both the
variance-removed claim and the post-residualization CKA collapse.

This script does the cross-validated version:
  1. Random 70/30 split of prompts (50 splits with different seeds).
  2. For each split:
     - Fit ridge regression on the 70% training rows: B = ridge_fit(L_train, X_train)
     - Predict held-out activations: X_test_hat = L_test @ B
     - Compute held-out variance removed: 1 - var(X_test - X_test_hat) / var(X_test)
     - Residualize test rows: X_test_resid = X_test - X_test_hat
     - Compute CKA and Procrustes residual on (X_A_test_resid, X_B_test_resid)
  3. Report mean and 95% CI of held-out variance removed, residualized CKA,
     residualized Procrustes residual across the 50 splits.

If held-out variance removed >> 0 and CKA collapses on held-out residuals,
the negative methodological finding is robust to overfitting.

Output: runs/cosi/study_b_residual_cv_<timestamp>/results.json
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

from sovereign.research.cosi.cka import linear_cka  # noqa: E402
from sovereign.research.cosi.align import procrustes_residual, choose_shared_k  # noqa: E402

PHASE1_RUN_DIR = REPO_ROOT / "runs" / "cosi"
PROBE_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v1.jsonl"
RUN_DIR = REPO_ROOT / "runs" / "cosi"

LAYER_FRACS = (0.25, 0.5, 0.75)
RIDGE_LAMBDA = 0.01
N_SPLITS = 50
TRAIN_FRAC = 0.7
SEED = 20260507

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


def ridge_fit(L: np.ndarray, X: np.ndarray, lam: float, mean_X: np.ndarray, mean_L: np.ndarray):
    """Return ridge coefficient B such that (L - mean_L) @ B approximates (X - mean_X)."""
    Lc = L - mean_L
    Xc = X - mean_X
    V = Lc.shape[1]
    G = Lc.T @ Lc + lam * np.eye(V, dtype=Lc.dtype)
    return np.linalg.solve(G, Lc.T @ Xc)


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_DIR / f"study_b_residual_cv_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("=== COSI Study B'' (cross-validated lexical-residual) ===", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    probes = []
    with open(PROBE_PATH) as f:
        for line in f:
            probes.append(json.loads(line))
    prompts = [p["prompt"] for p in probes]
    N = len(prompts)
    print(f"probes: {N}, splits: {N_SPLITS}, train_frac: {TRAIN_FRAC}", flush=True)

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

    rng = np.random.default_rng(SEED)
    n_train = int(TRAIN_FRAC * N)

    results: dict = {
        "timestamp": timestamp,
        "n_probes": N,
        "n_splits": N_SPLITS,
        "train_frac": TRAIN_FRAC,
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
        d_A = X_A.shape[1]
        d_B = X_B.shape[1]
        print(f"\n--- layer fraction {frac} ---", flush=True)
        print(f"  X_A: {X_A.shape}, X_B: {X_B.shape}", flush=True)

        # Per-split metrics
        var_removed_a_test = []
        var_removed_b_test = []
        cka_test_resid = []
        procrustes_test_resid = []
        cka_test_orig = []  # held-out CKA without residualization for comparison
        procrustes_test_orig = []
        k_pca_per_split = []

        t0 = time.time()
        for s in range(N_SPLITS):
            perm = rng.permutation(N)
            tr_idx = perm[:n_train]
            te_idx = perm[n_train:]
            L_train, L_test = L[tr_idx], L[te_idx]
            X_A_train, X_A_test = X_A[tr_idx], X_A[te_idx]
            X_B_train, X_B_test = X_B[tr_idx], X_B[te_idx]

            # Fit ridge on training (centered using training means)
            mean_L = L_train.mean(axis=0, keepdims=True)
            mean_X_A = X_A_train.mean(axis=0, keepdims=True)
            mean_X_B = X_B_train.mean(axis=0, keepdims=True)
            B_A = ridge_fit(L_train, X_A_train, RIDGE_LAMBDA, mean_X_A, mean_L)
            B_B = ridge_fit(L_train, X_B_train, RIDGE_LAMBDA, mean_X_B, mean_L)

            # Predict test
            X_A_test_c = X_A_test - mean_X_A
            X_B_test_c = X_B_test - mean_X_B
            L_test_c = L_test - mean_L
            X_A_test_hat = L_test_c @ B_A
            X_B_test_hat = L_test_c @ B_B
            X_A_test_resid = X_A_test_c - X_A_test_hat
            X_B_test_resid = X_B_test_c - X_B_test_hat

            # Held-out variance removed (raw activation variance, summed across all dims)
            var_X_A_test = float((X_A_test_c ** 2).sum())
            var_X_B_test = float((X_B_test_c ** 2).sum())
            var_resid_A = float((X_A_test_resid ** 2).sum())
            var_resid_B = float((X_B_test_resid ** 2).sum())
            var_removed_a_test.append(1.0 - var_resid_A / var_X_A_test if var_X_A_test > 0 else 0.0)
            var_removed_b_test.append(1.0 - var_resid_B / var_X_B_test if var_X_B_test > 0 else 0.0)

            # CKA: held-out test, before and after residualization
            cka_test_orig.append(linear_cka(X_A_test, X_B_test))
            cka_test_resid.append(linear_cka(X_A_test_resid, X_B_test_resid))

            # Procrustes: held-out test, before and after residualization
            try:
                k = choose_shared_k(X_A_test, X_B_test, var_threshold=0.90)
                k_pca_per_split.append(k)
                proc_orig = procrustes_residual(X_A_test, X_B_test, k=k).residual
                k_resid = choose_shared_k(X_A_test_resid, X_B_test_resid, var_threshold=0.90)
                proc_resid = procrustes_residual(X_A_test_resid, X_B_test_resid, k=k_resid).residual
                procrustes_test_orig.append(proc_orig)
                procrustes_test_resid.append(proc_resid)
            except Exception as e:
                print(f"    split {s}: procrustes failed: {e}", flush=True)

        elapsed = time.time() - t0

        var_a_arr = np.array(var_removed_a_test)
        var_b_arr = np.array(var_removed_b_test)
        cka_orig_arr = np.array(cka_test_orig)
        cka_resid_arr = np.array(cka_test_resid)
        proc_orig_arr = np.array(procrustes_test_orig)
        proc_resid_arr = np.array(procrustes_test_resid)

        print(f"  {N_SPLITS} splits in {elapsed:.1f}s", flush=True)
        print(f"  Held-out variance removed:", flush=True)
        print(f"    X_A: mean={var_a_arr.mean():.4f}, 95%CI=[{np.percentile(var_a_arr,2.5):.4f}, {np.percentile(var_a_arr,97.5):.4f}]", flush=True)
        print(f"    X_B: mean={var_b_arr.mean():.4f}, 95%CI=[{np.percentile(var_b_arr,2.5):.4f}, {np.percentile(var_b_arr,97.5):.4f}]", flush=True)
        print(f"  Held-out CKA (no residualization): mean={cka_orig_arr.mean():.4f}, 95%CI=[{np.percentile(cka_orig_arr,2.5):.4f}, {np.percentile(cka_orig_arr,97.5):.4f}]", flush=True)
        print(f"  Held-out CKA (residualized):       mean={cka_resid_arr.mean():.4f}, 95%CI=[{np.percentile(cka_resid_arr,2.5):.4f}, {np.percentile(cka_resid_arr,97.5):.4f}]", flush=True)
        print(f"  Held-out Procrustes (no residualization): mean={proc_orig_arr.mean():.4f}", flush=True)
        print(f"  Held-out Procrustes (residualized):       mean={proc_resid_arr.mean():.4f}", flush=True)
        delta_cka = cka_resid_arr.mean() - cka_orig_arr.mean()
        delta_proc = proc_resid_arr.mean() - proc_orig_arr.mean()
        print(f"  DELTAS: CKA Δ={delta_cka:+.4f}, Procrustes Δ={delta_proc:+.4f}", flush=True)

        results["by_layer"][str(frac)] = {
            "variance_removed_test_a_mean": float(var_a_arr.mean()),
            "variance_removed_test_a_ci_low": float(np.percentile(var_a_arr, 2.5)),
            "variance_removed_test_a_ci_high": float(np.percentile(var_a_arr, 97.5)),
            "variance_removed_test_b_mean": float(var_b_arr.mean()),
            "variance_removed_test_b_ci_low": float(np.percentile(var_b_arr, 2.5)),
            "variance_removed_test_b_ci_high": float(np.percentile(var_b_arr, 97.5)),
            "cka_test_orig_mean": float(cka_orig_arr.mean()),
            "cka_test_orig_ci_low": float(np.percentile(cka_orig_arr, 2.5)),
            "cka_test_orig_ci_high": float(np.percentile(cka_orig_arr, 97.5)),
            "cka_test_resid_mean": float(cka_resid_arr.mean()),
            "cka_test_resid_ci_low": float(np.percentile(cka_resid_arr, 2.5)),
            "cka_test_resid_ci_high": float(np.percentile(cka_resid_arr, 97.5)),
            "procrustes_test_orig_mean": float(proc_orig_arr.mean()),
            "procrustes_test_resid_mean": float(proc_resid_arr.mean()),
            "delta_cka": float(delta_cka),
            "delta_procrustes": float(delta_proc),
        }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}", flush=True)

    print("\n=== STUDY B'' SUMMARY ===", flush=True)
    print("All metrics computed on held-out test rows (Procrustes/CKA fitted+evaluated on test only).", flush=True)
    for frac, r in results["by_layer"].items():
        print(
            f"  layer_frac={frac}: held-out variance removed: A={r['variance_removed_test_a_mean']:.4f}, "
            f"B={r['variance_removed_test_b_mean']:.4f}; "
            f"CKA orig={r['cka_test_orig_mean']:.4f} -> resid={r['cka_test_resid_mean']:.4f} "
            f"(Δ={r['delta_cka']:+.4f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
