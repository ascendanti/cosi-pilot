#!/usr/bin/env python
"""Generate publication figures for COSI Phase 0 / Phase 1 results.

Produces, for each phase:
  fig_residual_vs_null.pdf   — observed Procrustes residual vs permutation
                                null distribution per layer fraction.
  fig_residual_by_domain.pdf — (Phase 1 only) per-domain alignment quality.

Figures are saved to papers/cosi/figures/.

Usage:
    .venv/bin/python scripts/cosi_figures.py phase0
    .venv/bin/python scripts/cosi_figures.py phase1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "runs" / "cosi"
FIG_DIR = REPO_ROOT / "papers" / "cosi" / "figures"


def latest_run(phase: str) -> Path:
    candidates = sorted(p for p in RUN_DIR.glob(f"{phase}_*") if p.is_dir())
    if not candidates:
        raise SystemExit(f"no runs found for {phase}")
    return candidates[-1]


def regen_perm_null_samples(activations: dict, frac: float, n: int = 1000, seed: int = 0) -> np.ndarray:
    """Regenerate permutation null samples for a layer fraction."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from sovereign.research.cosi.align import procrustes_residual, choose_shared_k

    X_A = activations["A"][frac]
    X_B = activations["B"][frac]
    k = choose_shared_k(X_A, X_B)
    rng = np.random.default_rng(seed)
    samples = np.empty(n, dtype=np.float64)
    for i in range(n):
        perm = rng.permutation(X_B.shape[0])
        samples[i] = procrustes_residual(X_A, X_B[perm], k=k).residual
    return samples


def fig_residual_vs_null(phase: str, run_dir: Path) -> None:
    """Plot observed residual against permutation null distribution per layer."""
    import matplotlib.pyplot as plt

    with open(run_dir / "results.json") as f:
        results = json.load(f)

    # Load activation matrices to regenerate null samples for plotting
    label_a = results["model_a"]
    label_b = results["model_b"]
    npz_a = np.load(run_dir / f"activations_{label_a}.npz")
    npz_b = np.load(run_dir / f"activations_{label_b}.npz")
    activations = {
        "A": {float(k.split("_", 1)[1]): npz_a[k] for k in npz_a.files},
        "B": {float(k.split("_", 1)[1]): npz_b[k] for k in npz_b.files},
    }

    layer_fractions = sorted(activations["A"].keys())

    fig, axes = plt.subplots(1, len(layer_fractions), figsize=(4.5 * len(layer_fractions), 3.5), sharey=False)
    if len(layer_fractions) == 1:
        axes = [axes]

    for ax, frac in zip(axes, layer_fractions):
        rec = results["by_layer_fraction"][str(frac)]
        observed = rec["observed_residual"]
        # Regenerate samples for histogram visualization. We use 200 (sufficient
        # for histogram shape) rather than the 1000 used in the main alignment
        # run, because regen at N=600 is expensive. The summary statistics
        # plotted (perm_null mean, p1, observed residual, z-score) come from
        # the full 1000-sample run as recorded in results.json; only the
        # histogram density is approximated.
        samples = regen_perm_null_samples(activations, frac, n=200, seed=int(frac * 100))

        ax.hist(samples, bins=40, density=True, alpha=0.55, color="#888", edgecolor="white", linewidth=0.4)
        ax.axvline(observed, color="#c0392b", linewidth=2.0, label=f"observed = {observed:.3f}")
        ax.axvline(rec["permutation_null_p1"], color="#2c3e50", linestyle="--", linewidth=1.0,
                   label=f"null $p_1$ = {rec['permutation_null_p1']:.3f}")
        ax.set_xlabel("Procrustes residual")
        ax.set_ylabel("density" if frac == layer_fractions[0] else "")
        ax.set_title(f"layer fraction {frac}\n$z = {rec['z_vs_permutation']:+.1f}\\sigma$")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        # Annotate
        if rec["below_p1_permutation"]:
            ax.text(0.97, 0.03, "below $p_1$", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2c3e50"))

    fig.suptitle(
        f"COSI {phase.upper()}: observed Procrustes residual vs.\\ permutation null\n"
        f"{label_a} vs.\\ {label_b}, $N={results['probes']}$ probes, "
        f"$S={results['n_null_samples']}$ null samples",
        fontsize=10,
    )
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{phase}_residual_vs_null.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def fig_residual_by_domain(phase: str, run_dir: Path) -> None:
    """Plot per-domain residual at each layer fraction (Phase 1 only)."""
    import matplotlib.pyplot as plt

    with open(run_dir / "results.json") as f:
        results = json.load(f)
    if "by_domain_alignment" not in results:
        print(f"  (no per-domain data in {phase}; skipping)")
        return

    domains = sorted(results["by_domain_alignment"].keys())
    layer_fractions = sorted(results["by_layer_fraction"].keys(), key=float)
    full = [results["by_layer_fraction"][f]["observed_residual"] for f in layer_fractions]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(layer_fractions))
    width = 0.18
    palette = {"math": "#1f77b4", "logic": "#ff7f0e", "polecon": "#2ca02c"}
    ax.bar(x - 1.5 * width, full, width, label="full set", color="#444", edgecolor="white")
    for i, dom in enumerate(domains):
        vals = [results["by_domain_alignment"][dom][f]["observed_residual"] for f in layer_fractions]
        ax.bar(x + (i - 0.5) * width, vals, width, label=dom, color=palette.get(dom, "gray"), edgecolor="white")
    # Reference line at chance level
    null_means = [results["by_layer_fraction"][f]["permutation_null_mean"] for f in layer_fractions]
    ax.plot(x, null_means, "k--", linewidth=1.0, label="permutation null mean")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(f):.2f}" for f in layer_fractions])
    ax.set_xlabel("layer fraction")
    ax.set_ylabel("Procrustes residual")
    ax.set_title(
        f"COSI {phase.upper()}: alignment quality by domain\n"
        f"{results['model_a']} vs.\\ {results['model_b']}",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_ylim(0, max(1.1, max(null_means) * 1.1))

    fig.tight_layout()
    out = FIG_DIR / f"{phase}_residual_by_domain.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["phase0", "phase1"])
    ap.add_argument("--run-dir", help="Specific run dir; defaults to latest matching phase")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else latest_run(args.phase)
    print(f"Using run dir: {run_dir}")

    fig_residual_vs_null(args.phase, run_dir)
    fig_residual_by_domain(args.phase, run_dir)


if __name__ == "__main__":
    main()
