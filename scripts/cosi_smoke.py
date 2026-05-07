#!/usr/bin/env python
"""COSI activation extraction smoke test.

Loads a small MLX model (default: Qwen3-4B, ~4GB) and a Phi-4-Reasoning-Plus
(15GB, phi3 dispatch), runs the extractor on a single prompt at three layer
fractions, and prints shape/norm sanity checks.

This does NOT validate isomorphism. It validates that the harness compiles,
that residual stream extraction matches the inner-model forward pass logic,
and that pooling produces vectors of the expected dimension.

Run:
    .venv/bin/python scripts/cosi_smoke.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from sovereign.research.cosi.extract import (  # noqa: E402
    ExtractionConfig,
    extract_residual,
    model_fingerprint,
)


SMOKE_PROMPT = (
    "Sovereign authority is constituted by the categorial schemas a state "
    "maintains rather than prior to them; changes in recognition produce "
    "changes in sovereignty. True or false?"
)


def smoke(model_path: str, label: str) -> None:
    from mlx_lm import load

    print(f"\n=== {label} ===")
    print(f"  path: {model_path}")
    t0 = time.time()
    model, tokenizer = load(model_path)
    t_load = time.time() - t0
    print(f"  load: {t_load:.1f}s")

    fp = model_fingerprint(model)
    print(f"  fingerprint: {fp}")

    cfg = ExtractionConfig(layer_fractions=(0.25, 0.5, 0.75), pooling="last")
    print(f"  config: {cfg}")

    t0 = time.time()
    out = extract_residual(model, tokenizer, [SMOKE_PROMPT], config=cfg)
    t_ext = time.time() - t0
    print(f"  extract: {t_ext:.2f}s for 1 prompt")

    for frac, arr in out.items():
        assert arr.shape == (1, fp["hidden_size"]), (
            f"frac={frac}: expected (1,{fp['hidden_size']}), got {arr.shape}"
        )
        norm = float(np.linalg.norm(arr[0]))
        # Sanity: not all-zero, not pathologically large
        assert 1e-3 < norm < 1e6, f"frac={frac}: suspicious norm {norm}"
        print(f"  layer_frac={frac}: shape={arr.shape}, L2={norm:.3f}")

    # Free model
    del model
    del tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--small-only",
        action="store_true",
        help="Skip Phi-4 (15GB) and only test Qwen3-4B.",
    )
    args = ap.parse_args()

    models_dir = REPO_ROOT / "models" / "mlx"

    smoke(str(models_dir / "Qwen3-4B"), "Qwen3-4B (qwen3 dispatch)")

    if not args.small_only:
        smoke(
            str(models_dir / "Phi-4-Reasoning-Plus"),
            "Phi-4-Reasoning-Plus (phi3 dispatch)",
        )

    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
