#!/usr/bin/env python
"""Study B: lexical and template baselines for cross-model alignment.

The Phase 1 result reports a Procrustes residual of 0.77 between Phi-4 and
Qwen3-Next-80B at layer 0.25, against a permutation null of ~1.0. A natural
question raised by the methodological critique: how much of that alignment
is attributable to shared lexical/template structure that ANY two
representations of the same prompts would exhibit?

This script computes alignment residuals for non-LLM representations of the
same Phase 1 probe set:

  1. TF-IDF (term-frequency × inverse-document-frequency) of the prompts.
  2. Bag-of-character-n-grams (character 3-grams) of the prompts.
  3. Template-only TF-IDF: prompts with content words masked out, leaving
     only template skeleton.

It then compares Procrustes residuals across the following pairings:

  (a) LLM_A vs LLM_B            — the original Phase 1 result.
  (b) LLM_A vs TF-IDF           — does LLM-A geometry track TF-IDF?
  (c) LLM_B vs TF-IDF           — does LLM-B geometry track TF-IDF?
  (d) TF-IDF vs char-n-grams    — pure lexical baseline.
  (e) TF-IDF vs template-only   — template-structure baseline.

If (a) is substantially lower than (b), (c), and (e), the LLM-vs-LLM
alignment carries information beyond lexical/template structure. If (a)
is comparable to or higher than the baselines, the operator-theoretic
interpretation is undermined.

Outputs:
- runs/cosi/study_b_<timestamp>/results.json
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

PROBE_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v1.jsonl"
PHASE1_RUN_DIR = REPO_ROOT / "runs" / "cosi"
RUN_DIR = REPO_ROOT / "runs" / "cosi"


# ---------------------------------------------------------------------------
# Lexical representation builders (no external dependencies)
# ---------------------------------------------------------------------------

STOPWORDS = set("""a an and are as at be by for from has have he in is it its
of on or that the to was were will with this these those if then which what
do does did how when where why""".split())

CONTENT_PLACEHOLDER = "<CONTENT>"


def tokenize(text: str) -> list[str]:
    """Lowercase and split into word-level tokens (alphanumeric)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Return all character n-grams of a string (lowercased, padded)."""
    s = " " + text.lower() + " "
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def build_tfidf(corpus: list[list[str]]) -> tuple[np.ndarray, list[str]]:
    """Compute TF-IDF matrix from a list of token-lists.

    Returns (matrix N x V, vocabulary list).
    """
    df: Counter = Counter()
    for tokens in corpus:
        for t in set(tokens):
            df[t] += 1
    vocab = sorted(df.keys())
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    N = len(corpus)
    V = len(vocab)
    matrix = np.zeros((N, V), dtype=np.float32)
    for i, tokens in enumerate(corpus):
        tf = Counter(tokens)
        for t, c in tf.items():
            j = vocab_idx[t]
            matrix[i, j] = c / max(1, len(tokens))
    # IDF
    idf = np.log((1 + N) / (1 + np.array([df[t] for t in vocab], dtype=np.float32))) + 1
    matrix = matrix * idf[None, :]
    # L2 normalize rows
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms
    return matrix, vocab


def template_only_tokens(prompt: str) -> list[str]:
    """Replace content tokens (non-stopword alphanumeric) with placeholder."""
    out = []
    for t in tokenize(prompt):
        out.append(t if t in STOPWORDS else CONTENT_PLACEHOLDER)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def latest_phase1_run() -> Path:
    candidates = sorted(p for p in PHASE1_RUN_DIR.glob("phase1_*") if p.is_dir())
    if not candidates:
        raise SystemExit("no phase1 run dir found")
    return candidates[-1]


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_DIR / f"study_b_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== COSI Study B: Lexical and Template Baselines ===", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    # Load probes
    probes = []
    with open(PROBE_PATH) as f:
        for line in f:
            probes.append(json.loads(line))
    prompts = [p["prompt"] for p in probes]
    N = len(prompts)
    print(f"probes: {N}", flush=True)

    # Build lexical representations
    print("\n--- building lexical representations ---", flush=True)
    word_tokens = [tokenize(p) for p in prompts]
    char_tokens = [char_ngrams(p, n=3) for p in prompts]
    template_tokens = [template_only_tokens(p) for p in prompts]

    X_tfidf, vocab_tfidf = build_tfidf(word_tokens)
    X_chargram, vocab_char = build_tfidf(char_tokens)
    X_template, vocab_template = build_tfidf(template_tokens)
    print(f"  TF-IDF (word): N={N}, V={len(vocab_tfidf)}", flush=True)
    print(f"  char-3gram:    N={N}, V={len(vocab_char)}", flush=True)
    print(f"  template-only: N={N}, V={len(vocab_template)}", flush=True)

    # Load LLM activations from Phase 1 run
    phase1_dir = latest_phase1_run()
    print(f"\nUsing Phase 1 activations from: {phase1_dir}", flush=True)
    label_a = "Phi-4-Reasoning-Plus"
    label_b = "Qwen3-Next-80B-Thinking-5bit"
    npz_a = np.load(phase1_dir / f"activations_{label_a}.npz")
    npz_b = np.load(phase1_dir / f"activations_{label_b}.npz")

    # Use layer 0.25 — strongest LLM-LLM signal
    X_llm_a = npz_a["frac_0.25"]
    X_llm_b = npz_b["frac_0.25"]
    print(f"  LLM_A (layer 0.25): {X_llm_a.shape}", flush=True)
    print(f"  LLM_B (layer 0.25): {X_llm_b.shape}", flush=True)

    # Run all the comparisons
    pairings = [
        ("(a) LLM_A vs LLM_B", X_llm_a, X_llm_b),
        ("(b) LLM_A vs TF-IDF", X_llm_a, X_tfidf),
        ("(c) LLM_B vs TF-IDF", X_llm_b, X_tfidf),
        ("(d) TF-IDF vs char-3gram", X_tfidf, X_chargram),
        ("(e) TF-IDF vs template-only", X_tfidf, X_template),
        ("(f) LLM_A vs char-3gram", X_llm_a, X_chargram),
        ("(g) LLM_A vs template-only", X_llm_a, X_template),
    ]

    results: dict = {
        "timestamp": timestamp,
        "n_probes": N,
        "n_null_samples": 500,
        "layer": 0.25,
        "vocab_sizes": {
            "tfidf_word": len(vocab_tfidf),
            "char_3gram": len(vocab_char),
            "template_only": len(vocab_template),
        },
        "shapes": {
            "LLM_A": list(X_llm_a.shape),
            "LLM_B": list(X_llm_b.shape),
            "TF-IDF": list(X_tfidf.shape),
            "char_3gram": list(X_chargram.shape),
            "template_only": list(X_template.shape),
        },
        "comparisons": [],
    }

    print("\n--- alignment + nulls (S=500 per cell) ---", flush=True)
    for label, X, Y in pairings:
        t0 = time.time()
        cr = run_cosi(X, Y, n_null_samples=500, seed=20260507)
        elapsed = time.time() - t0
        verdict = "below p1" if cr.below_p1_permutation else "above p1"
        print(
            f"  {label}: k={cr.observed.k}, residual={cr.observed.residual:.4f}, "
            f"perm_null mean={cr.permutation_null.mean:.4f} p1={cr.permutation_null.p1:.4f}, "
            f"z={cr.z_vs_permutation:.2f} [{verdict}] ({elapsed:.1f}s)",
            flush=True,
        )
        results["comparisons"].append({
            "pairing": label,
            "k": cr.observed.k,
            "observed_residual": cr.observed.residual,
            "permutation_null_mean": cr.permutation_null.mean,
            "permutation_null_p1": cr.permutation_null.p1,
            "z_vs_permutation": cr.z_vs_permutation,
            "below_p1_permutation": cr.below_p1_permutation,
        })

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}", flush=True)

    print("\n=== STUDY B SUMMARY ===", flush=True)
    print("Question: does LLM-vs-LLM alignment exceed what lexical baselines achieve?", flush=True)
    llm_llm = results["comparisons"][0]["observed_residual"]
    print(f"\nLLM-LLM residual (a): {llm_llm:.4f}\n", flush=True)
    for c in results["comparisons"][1:]:
        delta = c["observed_residual"] - llm_llm
        verdict = (
            "LLM-LLM is meaningfully better"
            if delta > 0.05
            else "LLM-LLM not distinguishable from baseline"
            if abs(delta) <= 0.05
            else "BASELINE IS BETTER (LLM-LLM is suspect)"
        )
        print(f"  {c['pairing']}: residual={c['observed_residual']:.4f} (Δ={delta:+.4f}) → {verdict}",
              flush=True)


if __name__ == "__main__":
    main()
