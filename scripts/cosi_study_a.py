#!/usr/bin/env python
"""Study A: invariance-leakage measurement on the Phase 1 model pair.

For each model in the Phase 1 pair (Phi-4-Reasoning-Plus,
Qwen3-Next-80B-A3B-Thinking) and each pre-registered layer fraction (0.25,
0.5, 0.75), measure the invariance-leakage statistic
||(I-P_V) f(P_V X)||_F / ||f(P_V X)||_F for the PCA-derived subspace and
compare it to a random-orthogonal-subspace baseline.

Conservative defaults: subset of the 600-probe set (first 100, sampled
across domains) to keep runtime tractable for the 80B model. n_random
subspaces = 20 (compute budget).

Outputs:
- runs/cosi/study_a_<timestamp>/results.json
"""

from __future__ import annotations

import json
import sys
import time
import dataclasses
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sovereign.research.cosi.leakage import measure_leakage  # noqa: E402
from sovereign.research.cosi.extract import model_fingerprint  # noqa: E402


PROBE_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v1.jsonl"
N_PROBES = 60   # 20 from each domain to keep 80B runtime reasonable
N_RANDOM = 20
LAYER_FRACTIONS = (0.25, 0.5, 0.75)
RUN_DIR = REPO_ROOT / "runs" / "cosi"


def load_balanced_probes(n_per_domain: int) -> list[dict]:
    by_domain: dict[str, list[dict]] = {}
    with open(PROBE_PATH) as f:
        for line in f:
            p = json.loads(line)
            by_domain.setdefault(p["domain"], []).append(p)
    selected = []
    for domain, probes in sorted(by_domain.items()):
        selected.extend(probes[:n_per_domain])
    return selected


def main() -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUN_DIR / f"study_a_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== COSI Study A: Invariance-Leakage Measurement ===", flush=True)
    print(f"run dir: {run_dir}", flush=True)

    n_per = N_PROBES // 3
    probes = load_balanced_probes(n_per)
    prompts = [p["prompt"] for p in probes]
    print(f"probes: {len(prompts)} ({n_per} per domain)", flush=True)

    models = [
        ("Phi-4-Reasoning-Plus", str(REPO_ROOT / "models" / "mlx" / "Phi-4-Reasoning-Plus"), False),
        ("Qwen3-Next-80B-Thinking-5bit", str(REPO_ROOT / "models" / "mlx" / "Qwen3-Next-80B-Thinking-5bit"), True),
    ]

    all_results: dict = {
        "timestamp": timestamp,
        "n_probes": len(prompts),
        "n_random_subspaces": N_RANDOM,
        "layer_fractions": list(LAYER_FRACTIONS),
        "by_model": {},
    }

    from mlx_lm import load
    import gc

    for label, path, lazy in models:
        print(f"\n[{label}]", flush=True)
        t0 = time.time()
        model, tokenizer = load(path, lazy=lazy)
        print(f"  load: {time.time() - t0:.1f}s", flush=True)
        fp = model_fingerprint(model)
        print(f"  fingerprint: {fp}", flush=True)

        t0 = time.time()
        results = measure_leakage(
            model,
            tokenizer,
            prompts,
            layer_fractions=LAYER_FRACTIONS,
            pooling_for_pca="last",
            n_random_subspaces=N_RANDOM,
        )
        print(f"  measure_leakage: {time.time() - t0:.1f}s", flush=True)

        all_results["by_model"][label] = {
            "fingerprint": fp,
            "by_layer": [dataclasses.asdict(r) for r in results],
        }

        print(f"\n  --- results for {label} ---", flush=True)
        for r in results:
            print(
                f"    layer_frac={r.layer_fraction} (idx {r.layer_index}, k={r.k}): "
                f"PCA leakage_token_mean={r.leakage_pca_token_mean:.4f}, "
                f"PCA leakage_sample={r.leakage_pca_sample:.4f}, "
                f"random leakage_token_mean={r.leakage_random_token_mean_mean:.4f}±{r.leakage_random_token_mean_std:.4f}",
                flush=True,
            )

        del model, tokenizer
        gc.collect()

    with open(run_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {run_dir / 'results.json'}", flush=True)

    print("\n=== STUDY A SUMMARY ===", flush=True)
    print("Lower leakage = more invariant. Random subspace = chance baseline.", flush=True)
    for label, data in all_results["by_model"].items():
        print(f"\n  {label}:", flush=True)
        for r in data["by_layer"]:
            ratio = r["leakage_pca_token_mean"] / r["leakage_random_token_mean_mean"] if r["leakage_random_token_mean_mean"] > 0 else float("nan")
            verdict = (
                "PCA more invariant than random"
                if r["leakage_pca_token_mean"] < r["leakage_random_token_mean_mean"] - 2 * r["leakage_random_token_mean_std"]
                else "PCA NOT distinguishably more invariant than random"
            )
            print(
                f"    layer_frac={r['layer_fraction']}: PCA leakage={r['leakage_pca_token_mean']:.4f}, "
                f"random={r['leakage_random_token_mean_mean']:.4f}±{r['leakage_random_token_mean_std']:.4f}, "
                f"ratio={ratio:.3f}, → {verdict}",
                flush=True,
            )


if __name__ == "__main__":
    main()
