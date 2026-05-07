#!/usr/bin/env python
"""Study F (CKA piece): centered kernel alignment between Phase 1 model activations.

The COSI methods section commits to reporting CKA as a cross-validation
metric alongside Procrustes. The pilot did not satisfy that commitment.
This script computes linear CKA between Phi-4-Reasoning-Plus and
Qwen3-Next-80B-A3B-Thinking activations from the Phase 1 run, with
permutation null, at all three pre-registered layer fractions and within
each domain.

Output: runs/cosi/study_f_cka_<timestamp>/results.json
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sovereign.research.cosi.cka import cka_with_permutation_null  # noqa: E402

PHASE1_RUN_DIR = REPO_ROOT / "runs" / "cosi"
PROBE_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v1.jsonl"
RUN_DIR = REPO_ROOT / "runs" / "cosi"


def latest_phase1() -> Path:
    candidates = sorted(p for p in PHASE1_RUN_DIR.glob("phase1_*") if p.is_dir())
    if not candidates:
        raise SystemExit("no phase1 run dir found")
    return candidates[-1]


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_DIR / f"study_f_cka_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== COSI Study F (CKA): Centered Kernel Alignment ===", flush=True)

    phase1_dir = latest_phase1()
    print(f"Using Phase 1 activations from: {phase1_dir}", flush=True)
    label_a = "Phi-4-Reasoning-Plus"
    label_b = "Qwen3-Next-80B-Thinking-5bit"
    npz_a = np.load(phase1_dir / f"activations_{label_a}.npz")
    npz_b = np.load(phase1_dir / f"activations_{label_b}.npz")

    # Load probes for per-domain stratification
    probes = []
    with open(PROBE_PATH) as f:
        for line in f:
            probes.append(json.loads(line))

    by_domain_idx: dict[str, list[int]] = {}
    for i, p in enumerate(probes):
        by_domain_idx.setdefault(p["domain"], []).append(i)

    layer_keys = sorted(npz_a.files)  # frac_0.25, frac_0.5, frac_0.75
    results: dict = {
        "timestamp": timestamp,
        "model_a": label_a,
        "model_b": label_b,
        "n_probes_full": npz_a[layer_keys[0]].shape[0],
        "by_layer_full": {},
        "by_layer_domain": {},
    }

    print("\n--- full set CKA (S=1000 perm null) ---", flush=True)
    for k in layer_keys:
        frac = float(k.split("_", 1)[1])
        X = npz_a[k]
        Y = npz_b[k]
        t0 = time.time()
        cr = cka_with_permutation_null(X, Y, n_perm=1000, seed=int(frac * 100))
        elapsed = time.time() - t0
        verdict = "ABOVE p99" if cr.above_p99_permutation else "above p99 NOT confirmed"
        print(
            f"  layer_frac={frac}: CKA={cr.cka_observed:.4f}, "
            f"perm_null mean={cr.perm_null_mean:.4f}±{cr.perm_null_std:.4f}, "
            f"p99={cr.perm_null_p99:.4f}, z={cr.z_vs_perm:.2f} [{verdict}] ({elapsed:.1f}s)",
            flush=True,
        )
        results["by_layer_full"][str(frac)] = dataclasses.asdict(cr)

    print("\n--- per-domain CKA (S=500 perm null) ---", flush=True)
    for domain, idx in by_domain_idx.items():
        idx_arr = np.array(idx)
        results["by_layer_domain"][domain] = {}
        print(f"\n  domain={domain} (n={len(idx_arr)}):", flush=True)
        for k in layer_keys:
            frac = float(k.split("_", 1)[1])
            X = npz_a[k][idx_arr]
            Y = npz_b[k][idx_arr]
            t0 = time.time()
            cr = cka_with_permutation_null(X, Y, n_perm=500, seed=int(frac * 100) + hash(domain) % 1000)
            elapsed = time.time() - t0
            verdict = "above p99" if cr.above_p99_permutation else "below p99"
            print(
                f"    layer_frac={frac}: CKA={cr.cka_observed:.4f}, "
                f"perm_null mean={cr.perm_null_mean:.4f}, z={cr.z_vs_perm:.2f} [{verdict}] ({elapsed:.1f}s)",
                flush=True,
            )
            results["by_layer_domain"][domain][str(frac)] = dataclasses.asdict(cr)

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}", flush=True)

    print("\n=== STUDY F SUMMARY: CKA vs Procrustes ===", flush=True)
    print("CKA range [0,1]; higher = more similar.", flush=True)
    for frac, r in results["by_layer_full"].items():
        verdict = "consistent with Procrustes" if r["above_p99_permutation"] else "DISCREPANT WITH PROCRUSTES"
        print(
            f"  layer_frac={frac}: CKA={r['cka_observed']:.4f} (perm null mean={r['perm_null_mean']:.4f}) "
            f"→ {verdict}",
            flush=True,
        )


if __name__ == "__main__":
    main()
