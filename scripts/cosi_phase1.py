#!/usr/bin/env python
"""COSI Phase 1: Phi-4-Reasoning-Plus vs Qwen3-Next-80B-A3B-Thinking on the
600-probe cross-domain Phase 1 probe set.

Per the COSI design note §3 and the Phase 0 result note §3:
- Maximum-distance pair within local zoo (dense pure-attention vs hybrid
  attention/SSM)
- Probe set extended to N=600 across three domains (math, logic, polecon)
- Pre-registered layer fractions (0.25, 0.5, 0.75)
- Pre-registered pooling (last-token; mean and max as Phase 1b sweeps)
- Pre-registered significance criterion (residual < perm null p1 AND z < -2)
- 1000 permutation null samples per layer fraction

Outputs:
- runs/cosi/phase1_<timestamp>/activations_<model>.npz
- runs/cosi/phase1_<timestamp>/results.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sovereign.research.cosi.extract import (  # noqa: E402
    ExtractionConfig,
    extract_residual,
    model_fingerprint,
)
from sovereign.research.cosi.align import run_cosi  # noqa: E402


PROBE_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v1.jsonl"
LAYER_FRACTIONS = (0.25, 0.5, 0.75)
N_NULL_SAMPLES = 1000
RUN_DIR = REPO_ROOT / "runs" / "cosi"


def load_probes() -> list[dict]:
    probes = []
    with open(PROBE_PATH) as f:
        for line in f:
            probes.append(json.loads(line))
    return probes


def extract_for_model(model_path: str, label: str, prompts: list[str], lazy: bool) -> tuple[dict, dict]:
    from mlx_lm import load

    print(f"  loading {label} (lazy={lazy})", flush=True)
    t0 = time.time()
    model, tokenizer = load(model_path, lazy=lazy)
    print(f"    load: {time.time() - t0:.1f}s", flush=True)

    fp = model_fingerprint(model)
    print(f"    fingerprint: {fp}", flush=True)

    cfg = ExtractionConfig(layer_fractions=LAYER_FRACTIONS, pooling="last")
    print(f"    extracting {len(prompts)} prompts", flush=True)
    t0 = time.time()
    out = extract_residual(model, tokenizer, prompts, config=cfg)
    elapsed = time.time() - t0
    print(f"    extract: {elapsed:.1f}s ({elapsed / len(prompts):.2f}s/prompt)", flush=True)

    del model, tokenizer
    gc.collect()
    return out, fp


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_DIR / f"phase1_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== COSI Phase 1 ===", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    probes = load_probes()
    prompts = [p["prompt"] for p in probes]
    print(f"probes: {len(prompts)}", flush=True)
    by_domain: dict[str, int] = {}
    for p in probes:
        by_domain[p["domain"]] = by_domain.get(p["domain"], 0) + 1
    print(f"by domain: {by_domain}", flush=True)

    models = [
        ("Phi-4-Reasoning-Plus", str(REPO_ROOT / "models" / "mlx" / "Phi-4-Reasoning-Plus"), False),
        ("Qwen3-Next-80B-Thinking-5bit", str(REPO_ROOT / "models" / "mlx" / "Qwen3-Next-80B-Thinking-5bit"), True),
    ]

    activations: dict[str, dict] = {}
    fingerprints: dict[str, dict] = {}

    print("\n--- extracting activations ---", flush=True)
    for label, path, lazy in models:
        print(f"\n[{label}]", flush=True)
        out, fp = extract_for_model(path, label, prompts, lazy=lazy)
        activations[label] = out
        fingerprints[label] = fp
        npz_path = run_dir / f"activations_{label}.npz"
        np.savez(
            npz_path,
            **{f"frac_{frac}": arr for frac, arr in out.items()},
        )
        print(f"  saved {npz_path}", flush=True)

    label_a, label_b = models[0][0], models[1][0]

    print("\n--- alignment + nulls ---", flush=True)
    results: dict = {
        "timestamp": timestamp,
        "phase": 1,
        "probes": len(prompts),
        "by_domain": by_domain,
        "model_a": label_a,
        "model_b": label_b,
        "fingerprints": fingerprints,
        "n_null_samples": N_NULL_SAMPLES,
        "by_layer_fraction": {},
    }

    # Run alignment on full set, then per-domain stratification
    print("\n  full set:", flush=True)
    for frac in LAYER_FRACTIONS:
        X_A = activations[label_a][frac]
        X_B = activations[label_b][frac]
        print(f"    layer_frac={frac}: X_A {X_A.shape}, X_B {X_B.shape}", flush=True)
        t0 = time.time()
        cr = run_cosi(X_A, X_B, n_null_samples=N_NULL_SAMPLES, seed=int(frac * 100))
        elapsed = time.time() - t0
        verdict = "BELOW p1" if cr.below_p1_permutation else "above p1"
        print(
            f"      k={cr.observed.k}, residual={cr.observed.residual:.4f}, "
            f"perm_null mean={cr.permutation_null.mean:.4f} p1={cr.permutation_null.p1:.4f}, "
            f"z={cr.z_vs_permutation:.2f} [{verdict}], rot_null mean={cr.rotation_null.mean:.4f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        results["by_layer_fraction"][str(frac)] = {
            "k": cr.observed.k,
            "var_explained_a": cr.observed.var_explained_a,
            "var_explained_b": cr.observed.var_explained_b,
            "observed_residual": cr.observed.residual,
            "permutation_null_mean": cr.permutation_null.mean,
            "permutation_null_std": cr.permutation_null.std,
            "permutation_null_p1": cr.permutation_null.p1,
            "permutation_null_p5": cr.permutation_null.p5,
            "rotation_null_mean": cr.rotation_null.mean,
            "rotation_null_std": cr.rotation_null.std,
            "z_vs_permutation": cr.z_vs_permutation,
            "below_p1_permutation": cr.below_p1_permutation,
        }

    # Per-domain stratification
    print("\n  per-domain:", flush=True)
    domain_idx: dict[str, list[int]] = {}
    for i, p in enumerate(probes):
        domain_idx.setdefault(p["domain"], []).append(i)
    results["by_domain_alignment"] = {}
    for domain, indices in domain_idx.items():
        idx = np.array(indices)
        print(f"\n    domain={domain} (n={len(idx)}):", flush=True)
        results["by_domain_alignment"][domain] = {}
        for frac in LAYER_FRACTIONS:
            X_A = activations[label_a][frac][idx]
            X_B = activations[label_b][frac][idx]
            t0 = time.time()
            # Smaller n_null to keep total runtime bounded
            cr = run_cosi(X_A, X_B, n_null_samples=500, seed=int(frac * 100) + hash(domain) % 1000)
            elapsed = time.time() - t0
            verdict = "BELOW p1" if cr.below_p1_permutation else "above p1"
            print(
                f"      layer_frac={frac}: k={cr.observed.k}, residual={cr.observed.residual:.4f}, "
                f"perm_null mean={cr.permutation_null.mean:.4f}, z={cr.z_vs_permutation:.2f} [{verdict}] "
                f"({elapsed:.1f}s)",
                flush=True,
            )
            results["by_domain_alignment"][domain][str(frac)] = {
                "k": cr.observed.k,
                "observed_residual": cr.observed.residual,
                "permutation_null_mean": cr.permutation_null.mean,
                "z_vs_permutation": cr.z_vs_permutation,
                "below_p1_permutation": cr.below_p1_permutation,
            }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    print("Full set:", flush=True)
    for frac, r in results["by_layer_fraction"].items():
        verdict = "ISOMORPHIC" if r["below_p1_permutation"] else "no signal"
        print(
            f"  layer_frac={frac}: residual={r['observed_residual']:.4f} "
            f"vs perm null mean={r['permutation_null_mean']:.4f}, z={r['z_vs_permutation']:+.2f} → {verdict}",
            flush=True,
        )
    print("Per domain (layer 0.25):", flush=True)
    for domain, by_frac in results["by_domain_alignment"].items():
        r = by_frac["0.25"]
        verdict = "ISOMORPHIC" if r["below_p1_permutation"] else "no signal"
        print(
            f"  {domain}: residual={r['observed_residual']:.4f}, z={r['z_vs_permutation']:+.2f} → {verdict}",
            flush=True,
        )


if __name__ == "__main__":
    main()
