#!/usr/bin/env python
"""Standalone smoke test for the qwen3_next dispatch on Qwen3-Next-80B-Thinking.

Kept in a separate file from cosi_smoke.py because the 80B load is heavy
(~51GB of unified memory pressure on a 64GB M1 Ultra). Run only when other
memory-hungry processes are quiet.
"""

from __future__ import annotations

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
    "maintains rather than prior to them. True or false?"
)


def main() -> None:
    from mlx_lm import load

    model_path = str(REPO_ROOT / "models" / "mlx" / "Qwen3-Next-80B-Thinking-5bit")
    print(f"Loading {model_path}")
    print("(this is ~51GB; expect 30s–2min)")
    t0 = time.time()
    try:
        model, tokenizer = load(model_path, lazy=True)
    except Exception as e:
        print(f"\nLOAD FAILED: {type(e).__name__}: {e}")
        sys.exit(1)
    print(f"  load: {time.time() - t0:.1f}s")

    fp = model_fingerprint(model)
    print(f"  fingerprint: {fp}")
    assert fp["architecture"] == "qwen3_next", f"expected qwen3_next, got {fp['architecture']}"

    cfg = ExtractionConfig(layer_fractions=(0.5,), pooling="last")
    print(f"  config: {cfg}")

    t0 = time.time()
    out = extract_residual(model, tokenizer, [SMOKE_PROMPT], config=cfg)
    print(f"  extract: {time.time() - t0:.1f}s for 1 prompt")

    arr = out[0.5]
    assert arr.shape == (1, fp["hidden_size"]), f"shape mismatch: {arr.shape}"
    norm = float(np.linalg.norm(arr[0]))
    assert 1e-3 < norm < 1e6, f"suspicious norm: {norm}"
    print(f"  layer_frac=0.5: shape={arr.shape}, L2={norm:.3f}")
    print("\n80B smoke test passed: qwen3_next dispatch handles hybrid attn/SSM masking.")


if __name__ == "__main__":
    main()
