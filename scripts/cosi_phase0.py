#!/usr/bin/env python
"""COSI Phase 0: Phi-4-Reasoning-Plus vs Qwen3-4B on the frontier-titles probe set.

This is the calibration run. It validates the full pipeline (extract → align →
null) on a small architecture-distance pair before committing to Phase 1
(Phi-4 vs Qwen3-Next-80B, 51GB load).

The Phase 0 result is informative either way:
- If the residual is significantly below the permutation null, the pipeline is
  sensitive enough to detect cross-architecture isomorphism, and Phase 1 is
  the depth amplification.
- If the residual is at chance, either there is no isomorphism between this
  pair (interesting in itself) or the probe set is too small (96 propositions
  is at the lower bound; the design note suggested 500+ as the comfortable
  range).

Outputs:
- runs/cosi/phase0_<timestamp>/activations_<model>.npz (per layer fraction)
- runs/cosi/phase0_<timestamp>/results.json (residuals + null statistics)
- runs/cosi/phase0_<timestamp>/manifest.json (model fingerprints + config)
"""

from __future__ import annotations

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


PROBE_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_frontier_v1.jsonl"
LAYER_FRACTIONS = (0.25, 0.5, 0.75)
N_NULL_SAMPLES = 1000
RUN_DIR = REPO_ROOT / "runs" / "cosi"


def load_probes() -> list[dict]:
    probes = []
    with open(PROBE_PATH) as f:
        for line in f:
            probes.append(json.loads(line))
    return probes


def extract_for_model(model_path: str, label: str, prompts: list[str]) -> tuple[dict, dict]:
    """Load model, extract activations, return (activations_per_frac, fingerprint)."""
    from mlx_lm import load

    print(f"  loading {label} from {model_path}")
    t0 = time.time()
    model, tokenizer = load(model_path, lazy=True)
    print(f"    load: {time.time() - t0:.1f}s")

    fp = model_fingerprint(model)
    print(f"    fingerprint: {fp}")

    cfg = ExtractionConfig(layer_fractions=LAYER_FRACTIONS, pooling="last")
    print(f"    extracting {len(prompts)} prompts at {LAYER_FRACTIONS}")
    t0 = time.time()
    out = extract_residual(model, tokenizer, prompts, config=cfg)
    elapsed = time.time() - t0
    print(f"    extract: {elapsed:.1f}s ({elapsed / len(prompts):.2f}s/prompt)")

    # Free model before next load
    del model, tokenizer
    import gc

    gc.collect()
    return out, fp


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_DIR / f"phase0_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== COSI Phase 0 ===")
    print(f"run dir: {run_dir}")

    probes = load_probes()
    prompts = [p["prompt"] for p in probes]
    print(f"probes: {len(prompts)}")

    models = [
        ("Phi-4-Reasoning-Plus", str(REPO_ROOT / "models" / "mlx" / "Phi-4-Reasoning-Plus")),
        ("Qwen3-4B", str(REPO_ROOT / "models" / "mlx" / "Qwen3-4B")),
    ]

    activations: dict[str, dict] = {}
    fingerprints: dict[str, dict] = {}

    print("\n--- extracting activations ---")
    for label, path in models:
        out, fp = extract_for_model(path, label, prompts)
        activations[label] = out
        fingerprints[label] = fp
        # Save activations
        npz_path = run_dir / f"activations_{label}.npz"
        np.savez(
            npz_path,
            **{f"frac_{frac}": arr for frac, arr in out.items()},
        )
        print(f"    saved {npz_path}")

    label_a, label_b = models[0][0], models[1][0]

    print("\n--- alignment + nulls ---")
    results: dict = {
        "timestamp": timestamp,
        "probes": len(prompts),
        "model_a": label_a,
        "model_b": label_b,
        "fingerprints": fingerprints,
        "n_null_samples": N_NULL_SAMPLES,
        "by_layer_fraction": {},
    }

    for frac in LAYER_FRACTIONS:
        X_A = activations[label_a][frac]
        X_B = activations[label_b][frac]
        print(f"  layer_frac={frac}: X_A {X_A.shape}, X_B {X_B.shape}")
        t0 = time.time()
        cr = run_cosi(X_A, X_B, n_null_samples=N_NULL_SAMPLES, seed=int(frac * 100))
        elapsed = time.time() - t0
        verdict = "BELOW p1" if cr.below_p1_permutation else "above p1"
        print(
            f"    k={cr.observed.k}, residual={cr.observed.residual:.4f}, "
            f"perm_null mean={cr.permutation_null.mean:.4f} p1={cr.permutation_null.p1:.4f}, "
            f"z={cr.z_vs_permutation:.2f} [{verdict}], rot_null mean={cr.rotation_null.mean:.4f} "
            f"({elapsed:.1f}s)"
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

    # Save manifest + results
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}")

    # Print summary
    print("\n=== SUMMARY ===")
    for frac, r in results["by_layer_fraction"].items():
        verdict = "ISOMORPHIC" if r["below_p1_permutation"] else "no signal"
        print(
            f"  layer_frac={frac}: residual={r['observed_residual']:.4f} "
            f"vs perm null mean={r['permutation_null_mean']:.4f}, z={r['z_vs_permutation']:+.2f} → {verdict}"
        )


if __name__ == "__main__":
    main()
