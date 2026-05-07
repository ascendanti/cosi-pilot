#!/usr/bin/env python
"""Study C: train/test Procrustes split for Phase 1 activations.

The Phase 1 result fits Procrustes on the full probe set and evaluates on
the same set. Study C addresses this by:

  1. Random K-fold style: 70/30 random splits, fit R* on train rows,
     evaluate residual on held-out test rows. Repeat for many splits;
     report mean and 95% CI.
  2. Cross-domain: fit R* on one domain's rows (e.g., math), evaluate on
     another domain's rows (e.g., logic). All ordered domain pairs.

Output: runs/cosi/study_c_<timestamp>/results.json
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sovereign.research.cosi.align import _pca_project, choose_shared_k  # noqa: E402

PHASE1_RUN_DIR = REPO_ROOT / "runs" / "cosi"
PROBE_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v1.jsonl"
RUN_DIR = REPO_ROOT / "runs" / "cosi"

LAYER_FRACS = (0.25, 0.5, 0.75)
N_RANDOM_SPLITS = 50
TRAIN_FRAC = 0.7


def latest_phase1() -> Path:
    candidates = sorted(p for p in PHASE1_RUN_DIR.glob("phase1_*") if p.is_dir())
    if not candidates:
        raise SystemExit("no phase1 run dir found")
    return candidates[-1]


def fit_eval_split(
    X_A_train: np.ndarray,
    X_B_train: np.ndarray,
    X_A_test: np.ndarray,
    X_B_test: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Fit Procrustes on training projections, evaluate on test projections.

    PCA is fit on the *combined* train+test data so that the projection
    geometry is consistent (otherwise train PCs and test PCs would differ
    and the alignment would be on different bases). Returns
    (train_residual, test_residual).
    """
    X_A_full = np.vstack([X_A_train, X_A_test])
    X_B_full = np.vstack([X_B_train, X_B_test])
    A_proj_full, _ = _pca_project(X_A_full, k)
    B_proj_full, _ = _pca_project(X_B_full, k)
    n_train = X_A_train.shape[0]
    A_train = A_proj_full[:n_train]
    B_train = B_proj_full[:n_train]
    A_test = A_proj_full[n_train:]
    B_test = B_proj_full[n_train:]

    R, _ = orthogonal_procrustes(B_train, A_train)
    train_num = np.linalg.norm(A_train - B_train @ R, ord="fro")
    train_den = np.linalg.norm(A_train, ord="fro")
    test_num = np.linalg.norm(A_test - B_test @ R, ord="fro")
    test_den = np.linalg.norm(A_test, ord="fro")
    return float(train_num / train_den), float(test_num / test_den)


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_DIR / f"study_c_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== COSI Study C: Train/Test Procrustes Split ===", flush=True)

    phase1_dir = latest_phase1()
    label_a = "Phi-4-Reasoning-Plus"
    label_b = "Qwen3-Next-80B-Thinking-5bit"
    npz_a = np.load(phase1_dir / f"activations_{label_a}.npz")
    npz_b = np.load(phase1_dir / f"activations_{label_b}.npz")
    print(f"loaded activations from {phase1_dir}", flush=True)

    probes = []
    with open(PROBE_PATH) as f:
        for line in f:
            probes.append(json.loads(line))
    by_domain_idx: dict[str, list[int]] = {}
    for i, p in enumerate(probes):
        by_domain_idx.setdefault(p["domain"], []).append(i)
    print(f"probes: {len(probes)}, domains: {sorted(by_domain_idx)}", flush=True)

    rng = np.random.default_rng(20260507)

    results: dict = {
        "timestamp": timestamp,
        "n_random_splits": N_RANDOM_SPLITS,
        "train_frac": TRAIN_FRAC,
        "by_layer": {},
    }

    print("\n--- random 70/30 split ---", flush=True)
    for frac in LAYER_FRACS:
        key = f"frac_{frac}"
        X_A = npz_a[key]
        X_B = npz_b[key]
        N = X_A.shape[0]
        n_train = int(TRAIN_FRAC * N)
        k = choose_shared_k(X_A, X_B, var_threshold=0.90)
        train_residuals = []
        test_residuals = []
        t0 = time.time()
        for s in range(N_RANDOM_SPLITS):
            perm = rng.permutation(N)
            tr_idx = perm[:n_train]
            te_idx = perm[n_train:]
            train_r, test_r = fit_eval_split(X_A[tr_idx], X_B[tr_idx], X_A[te_idx], X_B[te_idx], k)
            train_residuals.append(train_r)
            test_residuals.append(test_r)
        elapsed = time.time() - t0
        train_arr = np.array(train_residuals)
        test_arr = np.array(test_residuals)
        # Compare to original (fit and eval on same data)
        original_residual = None
        try:
            from sovereign.research.cosi.align import procrustes_residual
            original = procrustes_residual(X_A, X_B, k=k)
            original_residual = original.residual
        except Exception:
            pass
        orig_str = f"{original_residual:.4f}" if original_residual is not None else "N/A"
        print(
            f"  layer_frac={frac}: k={k}, "
            f"train_mean={train_arr.mean():.4f} (±{train_arr.std(ddof=1):.4f}), "
            f"test_mean={test_arr.mean():.4f} (±{test_arr.std(ddof=1):.4f}), "
            f"original={orig_str} ({elapsed:.1f}s)",
            flush=True,
        )
        results["by_layer"][str(frac)] = {
            "k": k,
            "train_mean": float(train_arr.mean()),
            "train_std": float(train_arr.std(ddof=1)),
            "train_ci_low": float(np.percentile(train_arr, 2.5)),
            "train_ci_high": float(np.percentile(train_arr, 97.5)),
            "test_mean": float(test_arr.mean()),
            "test_std": float(test_arr.std(ddof=1)),
            "test_ci_low": float(np.percentile(test_arr, 2.5)),
            "test_ci_high": float(np.percentile(test_arr, 97.5)),
            "original_full_set_residual": original_residual,
            "test_train_gap": float(test_arr.mean() - train_arr.mean()),
        }

    print("\n--- cross-domain (fit on one, eval on other) ---", flush=True)
    domains = sorted(by_domain_idx.keys())
    results["cross_domain"] = {}
    for frac in LAYER_FRACS:
        key = f"frac_{frac}"
        X_A_full = npz_a[key]
        X_B_full = npz_b[key]
        results["cross_domain"][str(frac)] = {}
        print(f"\n  layer_frac={frac}:", flush=True)
        for d_train in domains:
            for d_test in domains:
                if d_train == d_test:
                    continue
                tr_idx = np.array(by_domain_idx[d_train])
                te_idx = np.array(by_domain_idx[d_test])
                # Use a k that fits both subsets
                k = min(
                    choose_shared_k(X_A_full[tr_idx], X_B_full[tr_idx]),
                    choose_shared_k(X_A_full[te_idx], X_B_full[te_idx]),
                )
                train_r, test_r = fit_eval_split(
                    X_A_full[tr_idx], X_B_full[tr_idx], X_A_full[te_idx], X_B_full[te_idx], k
                )
                pair_label = f"{d_train}->{d_test}"
                print(
                    f"    {pair_label}: k={k}, train_resid={train_r:.4f}, "
                    f"test_resid={test_r:.4f}, gap={test_r - train_r:+.4f}",
                    flush=True,
                )
                results["cross_domain"][str(frac)][pair_label] = {
                    "k": k,
                    "train_residual": train_r,
                    "test_residual": test_r,
                    "gap": test_r - train_r,
                }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}", flush=True)

    print("\n=== STUDY C SUMMARY ===", flush=True)
    print("Train/test gap > 0.05 indicates substantial overfitting.", flush=True)
    for frac, r in results["by_layer"].items():
        verdict = (
            "minimal overfitting"
            if abs(r["test_train_gap"]) < 0.02
            else "modest overfitting"
            if abs(r["test_train_gap"]) < 0.05
            else "substantial overfitting"
        )
        print(
            f"  layer_frac={frac}: train={r['train_mean']:.4f}, test={r['test_mean']:.4f}, "
            f"gap={r['test_train_gap']:+.4f} → {verdict}",
            flush=True,
        )


if __name__ == "__main__":
    main()
