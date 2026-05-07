#!/usr/bin/env python
"""Study B4 (grouped lexical residualization): the strict version of B3.

Random 70/30 splits are honest at the row level but leak template family
structure, because held-out rows share templates with training rows. The
review proposes grouped cross-validation as the strict version: train on
all templates except one, evaluate on the held-out template; train on all
domains except one, evaluate on the held-out domain.

Three grouped splits implemented:
  (a) Leave-one-template-family-out (LOTFO): 28 splits, one per template
      family. For each, fit ridge on the 27 other families, residualize
      the held-out family's prompts, compute CKA/Procrustes on those
      residuals.
  (b) Leave-one-domain-out (LODO): 3 splits, one per domain (math, logic,
      polecon).
  (c) Single 2-vs-1 split: train on math+logic, test on polecon.

Output: runs/cosi/study_b4_grouped_<timestamp>/results.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sovereign.research.cosi.cka import linear_cka  # noqa: E402
from sovereign.research.cosi.align import procrustes_residual, choose_shared_k  # noqa: E402

PHASE1_RUN_DIR = REPO_ROOT / "runs" / "cosi"
PROBE_PATH_V2 = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v2.jsonl"
RUN_DIR = REPO_ROOT / "runs" / "cosi"

LAYER_FRACS = (0.25, 0.5, 0.75)
RIDGE_LAMBDA = 0.01
SEED = 20260507

STOPWORDS = set("""a an and are as at be by for from has have he in is it its
of on or that the to was were will with this these those if then which what
do does did how when where why""".split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def template_only_tokens(prompt: str) -> list[str]:
    return [t if t in STOPWORDS else "<CONTENT>" for t in tokenize(prompt)]


def build_tfidf(corpus: list[list[str]]) -> tuple[np.ndarray, list[str], dict]:
    """Fit TF-IDF and return (matrix, vocab, idf_vector_dict)."""
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
    return M / norms, vocab, {"vocab_idx": vocab_idx, "idf": idf}


def build_tfidf_apply(test_corpus: list[list[str]], vocab_info: dict) -> np.ndarray:
    """Apply previously-fit TF-IDF (training-time vocab + IDF) to test corpus.

    Uses TRAINING vocab and TRAINING IDF — out-of-vocab test tokens are
    dropped. This is the proper way to extend lexical featurization to
    held-out rows in a leave-out-template setting.
    """
    vocab_idx = vocab_info["vocab_idx"]
    idf = vocab_info["idf"]
    V = len(vocab_idx)
    N = len(test_corpus)
    M = np.zeros((N, V), dtype=np.float32)
    for i, tokens in enumerate(test_corpus):
        tf = Counter(tokens)
        for t, c in tf.items():
            j = vocab_idx.get(t)
            if j is not None:
                M[i, j] = c / max(1, len(tokens))
    M *= idf[None, :]
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return M / norms


def ridge_fit(L: np.ndarray, X: np.ndarray, lam: float):
    Lc = L - L.mean(axis=0, keepdims=True)
    Xc = X - X.mean(axis=0, keepdims=True)
    V = Lc.shape[1]
    G = Lc.T @ Lc + lam * np.eye(V, dtype=Lc.dtype)
    B = np.linalg.solve(G, Lc.T @ Xc)
    return B, L.mean(axis=0, keepdims=True), X.mean(axis=0, keepdims=True)


def latest_phase1() -> Path:
    candidates = sorted(p for p in PHASE1_RUN_DIR.glob("phase1_*") if p.is_dir())
    if not candidates:
        raise SystemExit("no phase1 run dir found")
    return candidates[-1]


def evaluate_split(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    word_tokens: list[list[str]],
    template_tokens: list[list[str]],
    X_A: np.ndarray,
    X_B: np.ndarray,
    lam: float,
) -> dict:
    """Fit lexical-to-activation ridge on train rows, residualize test rows,
    compute held-out CKA + Procrustes on raw and residualized.
    """
    # Fit lexical features on train tokens only (proper grouped CV)
    X_tfidf_train, _, tfidf_info = build_tfidf([word_tokens[i] for i in train_idx])
    X_template_train, _, template_info = build_tfidf([template_tokens[i] for i in train_idx])
    L_train = np.hstack([X_tfidf_train, X_template_train]).astype(np.float32)
    # Apply to test
    X_tfidf_test = build_tfidf_apply([word_tokens[i] for i in test_idx], tfidf_info)
    X_template_test = build_tfidf_apply([template_tokens[i] for i in test_idx], template_info)
    L_test = np.hstack([X_tfidf_test, X_template_test]).astype(np.float32)

    X_A_train, X_A_test = X_A[train_idx], X_A[test_idx]
    X_B_train, X_B_test = X_B[train_idx], X_B[test_idx]

    B_A, mean_L_A, mean_X_A = ridge_fit(L_train, X_A_train, lam)
    B_B, mean_L_B, mean_X_B = ridge_fit(L_train, X_B_train, lam)

    L_test_centered_A = L_test - mean_L_A
    L_test_centered_B = L_test - mean_L_B
    X_A_test_centered = X_A_test - mean_X_A
    X_B_test_centered = X_B_test - mean_X_B
    X_A_test_resid = X_A_test_centered - L_test_centered_A @ B_A
    X_B_test_resid = X_B_test_centered - L_test_centered_B @ B_B

    var_A = float((X_A_test_centered ** 2).sum())
    var_B = float((X_B_test_centered ** 2).sum())
    var_resid_A = float((X_A_test_resid ** 2).sum())
    var_resid_B = float((X_B_test_resid ** 2).sum())
    var_removed_A = 1.0 - var_resid_A / var_A if var_A > 0 else 0.0
    var_removed_B = 1.0 - var_resid_B / var_B if var_B > 0 else 0.0

    cka_orig = linear_cka(X_A_test, X_B_test)
    cka_resid = linear_cka(X_A_test_resid, X_B_test_resid)

    proc_orig = None
    proc_resid = None
    try:
        if X_A_test.shape[0] >= 8:
            k_orig = max(1, min(choose_shared_k(X_A_test, X_B_test, var_threshold=0.90), X_A_test.shape[0] - 1))
            proc_orig = procrustes_residual(X_A_test, X_B_test, k=k_orig).residual
            k_resid = max(1, min(choose_shared_k(X_A_test_resid, X_B_test_resid, var_threshold=0.90), X_A_test_resid.shape[0] - 1))
            proc_resid = procrustes_residual(X_A_test_resid, X_B_test_resid, k=k_resid).residual
    except Exception:
        pass

    return {
        "n_test": int(len(test_idx)),
        "var_removed_a": var_removed_A,
        "var_removed_b": var_removed_B,
        "cka_orig": cka_orig,
        "cka_resid": cka_resid,
        "procrustes_orig": proc_orig,
        "procrustes_resid": proc_resid,
    }


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_DIR / f"study_b4_grouped_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print("=== COSI Study B4 (grouped lexical residualization) ===", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    probes = []
    with open(PROBE_PATH_V2) as f:
        for line in f:
            probes.append(json.loads(line))
    prompts = [p["prompt"] for p in probes]
    template_ids = [p["template_id"] for p in probes]
    domains = [p["domain"] for p in probes]
    N = len(prompts)
    print(f"probes: {N}", flush=True)
    print(f"unique template families: {len(set(template_ids))}", flush=True)
    print(f"unique domains: {sorted(set(domains))}", flush=True)

    word_tokens = [tokenize(p) for p in prompts]
    template_tokens = [template_only_tokens(p) for p in prompts]

    phase1_dir = latest_phase1()
    label_a = "Phi-4-Reasoning-Plus"
    label_b = "Qwen3-Next-80B-Thinking-5bit"
    npz_a = np.load(phase1_dir / f"activations_{label_a}.npz")
    npz_b = np.load(phase1_dir / f"activations_{label_b}.npz")
    print(f"loaded activations from {phase1_dir}", flush=True)

    # Build index lookups
    by_template: dict[str, list[int]] = defaultdict(list)
    by_domain: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(probes):
        by_template[p["template_id"]].append(i)
        by_domain[p["domain"]].append(i)

    results: dict = {
        "timestamp": timestamp,
        "n_probes": N,
        "n_template_families": len(by_template),
        "ridge_lambda": RIDGE_LAMBDA,
        "model_a": label_a,
        "model_b": label_b,
        "by_layer": {},
    }

    for frac in LAYER_FRACS:
        key = f"frac_{frac}"
        X_A = npz_a[key]
        X_B = npz_b[key]
        print(f"\n--- layer fraction {frac} ---", flush=True)
        layer_results: dict = {}

        # (a) Leave-one-template-family-out
        print(f"  (a) Leave-one-template-family-out ({len(by_template)} splits)", flush=True)
        all_idx = np.arange(N)
        lotfo_per_template: dict[str, dict] = {}
        var_a_list, var_b_list, cka_orig_list, cka_resid_list, proc_orig_list, proc_resid_list = [], [], [], [], [], []
        t0 = time.time()
        for tid, te_idx_list in sorted(by_template.items()):
            te_idx = np.array(te_idx_list)
            tr_idx = np.setdiff1d(all_idx, te_idx)
            r = evaluate_split(tr_idx, te_idx, word_tokens, template_tokens, X_A, X_B, RIDGE_LAMBDA)
            lotfo_per_template[tid] = r
            var_a_list.append(r["var_removed_a"])
            var_b_list.append(r["var_removed_b"])
            cka_orig_list.append(r["cka_orig"])
            cka_resid_list.append(r["cka_resid"])
            if r["procrustes_orig"] is not None:
                proc_orig_list.append(r["procrustes_orig"])
                proc_resid_list.append(r["procrustes_resid"])
        elapsed = time.time() - t0
        var_a_arr = np.array(var_a_list)
        var_b_arr = np.array(var_b_list)
        cka_orig_arr = np.array(cka_orig_list)
        cka_resid_arr = np.array(cka_resid_list)
        proc_orig_arr = np.array(proc_orig_list) if proc_orig_list else np.array([np.nan])
        proc_resid_arr = np.array(proc_resid_list) if proc_resid_list else np.array([np.nan])
        print(f"    LOTFO completed in {elapsed:.1f}s", flush=True)
        print(f"    var_removed: A mean={var_a_arr.mean():.4f} (CI [{np.percentile(var_a_arr,2.5):.4f},{np.percentile(var_a_arr,97.5):.4f}]); "
              f"B mean={var_b_arr.mean():.4f} (CI [{np.percentile(var_b_arr,2.5):.4f},{np.percentile(var_b_arr,97.5):.4f}])", flush=True)
        print(f"    CKA orig mean={cka_orig_arr.mean():.4f}; CKA resid mean={cka_resid_arr.mean():.4f}; "
              f"Δ={cka_resid_arr.mean() - cka_orig_arr.mean():+.4f}", flush=True)
        print(f"    Procrustes orig mean={proc_orig_arr.mean():.4f}; Procrustes resid mean={proc_resid_arr.mean():.4f}; "
              f"Δ={proc_resid_arr.mean() - proc_orig_arr.mean():+.4f}", flush=True)
        layer_results["lotfo"] = {
            "n_splits": int(len(by_template)),
            "var_removed_a_mean": float(var_a_arr.mean()),
            "var_removed_a_ci": [float(np.percentile(var_a_arr, 2.5)), float(np.percentile(var_a_arr, 97.5))],
            "var_removed_b_mean": float(var_b_arr.mean()),
            "var_removed_b_ci": [float(np.percentile(var_b_arr, 2.5)), float(np.percentile(var_b_arr, 97.5))],
            "cka_orig_mean": float(cka_orig_arr.mean()),
            "cka_resid_mean": float(cka_resid_arr.mean()),
            "delta_cka": float(cka_resid_arr.mean() - cka_orig_arr.mean()),
            "procrustes_orig_mean": float(proc_orig_arr.mean()),
            "procrustes_resid_mean": float(proc_resid_arr.mean()),
            "delta_procrustes": float(proc_resid_arr.mean() - proc_orig_arr.mean()),
        }

        # (b) Leave-one-domain-out
        print(f"  (b) Leave-one-domain-out (3 splits)", flush=True)
        lodo_per_domain: dict[str, dict] = {}
        for d, te_idx_list in sorted(by_domain.items()):
            te_idx = np.array(te_idx_list)
            tr_idx = np.setdiff1d(all_idx, te_idx)
            r = evaluate_split(tr_idx, te_idx, word_tokens, template_tokens, X_A, X_B, RIDGE_LAMBDA)
            lodo_per_domain[d] = r
            print(f"    held-out {d}: var_removed A={r['var_removed_a']:.4f}, B={r['var_removed_b']:.4f}; "
                  f"CKA orig={r['cka_orig']:.4f} → resid={r['cka_resid']:.4f} (Δ={r['cka_resid']-r['cka_orig']:+.4f}); "
                  f"Proc orig={r['procrustes_orig']:.4f} → resid={r['procrustes_resid']:.4f}",
                  flush=True)
        layer_results["lodo"] = lodo_per_domain

        # (c) math+logic → polecon
        print(f"  (c) train math+logic, test polecon", flush=True)
        te_idx = np.array(by_domain["polecon"])
        tr_idx = np.array(by_domain["math"] + by_domain["logic"])
        r = evaluate_split(tr_idx, te_idx, word_tokens, template_tokens, X_A, X_B, RIDGE_LAMBDA)
        print(f"    var_removed A={r['var_removed_a']:.4f}, B={r['var_removed_b']:.4f}; "
              f"CKA orig={r['cka_orig']:.4f} → resid={r['cka_resid']:.4f} (Δ={r['cka_resid']-r['cka_orig']:+.4f}); "
              f"Proc orig={r['procrustes_orig']:.4f} → resid={r['procrustes_resid']:.4f}",
              flush=True)
        layer_results["math_logic_to_polecon"] = r

        results["by_layer"][str(frac)] = layer_results

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}", flush=True)

    print("\n=== STUDY B4 SUMMARY ===", flush=True)
    for frac, lr in results["by_layer"].items():
        l = lr["lotfo"]
        print(f"  layer_frac={frac} LOTFO: CKA orig={l['cka_orig_mean']:.4f} → resid={l['cka_resid_mean']:.4f} "
              f"(Δ={l['delta_cka']:+.4f}); var_removed A={l['var_removed_a_mean']:.4f}, B={l['var_removed_b_mean']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
