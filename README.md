# Cross-Operator Subspace Isomorphism (COSI) — Reproducibility Repository

This is the public reproducibility repository for the paper:

> **Cross-Architectural Activation Alignment in Frontier Language Models:
> A Representational-Geometry Pilot, with an Operator-Theoretic Program It Motivates.**
> Adam A. Bensaid (Atlas Consulting & Technology Services) and Claude (Anthropic). 2026.

---

## Status

This is a **representational-alignment pilot**, not an operator-theoretic
confirmation. The companion paper specifies seven studies (A–G) that
would, jointly, support the strong operator-theoretic claim. This
repository contains the pilot artifacts (Phase 0 and Phase 1) plus
in-progress results from Studies A, B, C, F-CKA. Studies D, E, and
external pre-registration (G) are work in progress; see the paper's
§Required Additional Work for the full program.

## Layout

```
papers/cosi/
  main.tex                       — paper source
  main.pdf                       — built paper
  references.bib                 — 50+ references
  sections/                      — paper sections
  figures/                       — generated figures
  preregistration_packet/        — pre-registration document + checksums + env
  synthesis_outline.md           — outline for the larger integrative paper

src/sovereign/research/cosi/
  extract.py                     — activation-extraction harness (phi3, qwen3, qwen3_next dispatches)
  align.py                       — Procrustes + permutation null + random-rotation null
  leakage.py                     — Study A: invariance-leakage measurement
  cka.py                         — Study F-CKA: centered kernel alignment

scripts/
  cosi_smoke.py / cosi_smoke_80b.py    — architecture-dispatch smoke tests
  cosi_phase0.py                       — Phase 0 runner
  cosi_phase1.py                       — Phase 1 runner
  build_phase1_probes.py               — Phase 1 probe-set generator (seed 20260506)
  cosi_figures.py                      — figure generation
  cosi_study_a.py                      — Study A: leakage
  cosi_study_b.py                      — Study B: lexical baselines
  cosi_study_c.py                      — Study C: train/test Procrustes
  cosi_study_f_cka.py                  — Study F-CKA

data/cosi/
  probe_set_frontier_v1.jsonl    — Phase 0 (96 probes)
  probe_set_phase1_v1.jsonl      — Phase 1 (600 probes)

runs/cosi/
  phase0_<timestamp>/            — Phase 0 results + activation matrices
  phase1_<timestamp>/            — Phase 1 results + activation matrices
  study_a_<timestamp>/           — Study A results
  study_b_<timestamp>/           — Study B results
  study_c_<timestamp>/           — Study C results
  study_f_cka_<timestamp>/       — Study F-CKA results

docs/research_notes/
  2026-05-06_cosi_design.md      — pre-registration document
  2026-05-06_cosi_phase0_result.md
  2026-05-06_cosi_phase1_result.md
```

## Reproducing the Pilot

Requirements: Apple Silicon Mac with ≥48 GB unified memory (for the 80 B
model in Phase 1; the smaller Phase 0 pair fits in ≤16 GB). Python 3.11.
MLX 0.31+ and `mlx_lm` 0.31+. SciPy and NumPy.

```bash
# Set up environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Models (one-time download from HuggingFace; ~70 GB total for full reproduction)
huggingface-cli download mlx-community/Phi-4-reasoning-plus-8bit
huggingface-cli download mlx-community/Qwen3-4B-8bit
huggingface-cli download mlx-community/Qwen3-Next-80B-A3B-Thinking-5bit

# Verify infrastructure
python scripts/cosi_smoke.py             # Phi-4 + Qwen3-4B dispatch
python scripts/cosi_smoke_80b.py         # Qwen3-Next 80B dispatch

# Phase 0 (~3 minutes)
python scripts/cosi_phase0.py

# Phase 1 (~16 minutes; loads 80B)
python scripts/cosi_phase1.py

# Studies A, B, C, F-CKA (~10 minutes each)
python scripts/cosi_study_a.py           # invariance leakage
python scripts/cosi_study_b.py           # lexical baselines
python scripts/cosi_study_c.py           # train/test Procrustes
python scripts/cosi_study_f_cka.py       # CKA cross-validation

# Figures
python scripts/cosi_figures.py phase0
python scripts/cosi_figures.py phase1

# Build paper
cd papers/cosi
tectonic main.tex
```

## Pre-Registration

The pilot's pre-registration document is at
`papers/cosi/preregistration_packet/PREREGISTRATION.md` with content
checksums in `CHECKSUMS.txt` and pip environment in `environment.txt`.
See the document for a full discussion of pre-registration commitments
and a candid note on the retrospective nature of the pilot's
pre-registration.

Future runs (Studies A through G) will be pre-registered under their
own externally timestamped deposits before execution.

## License

Code: MIT.
Probe sets and analysis artifacts: CC-BY 4.0.
Paper source: CC-BY 4.0.

## Citation

```bibtex
@article{bensaid2026cosi,
  author  = {Bensaid, Adam A. and {Claude} (Anthropic)},
  title   = {Cross-Architectural Activation Alignment in Frontier Language Models:
             A Representational-Geometry Pilot, with an Operator-Theoretic Program It Motivates},
  year    = {2026},
  journal = {arXiv preprint},
  note    = {arXiv:[TO BE FILLED]}
}
```

## Contact

Adam A. Bensaid, Atlas Consulting & Technology Services, Toronto, ON.
`adam.a.bensaid@gmail.com`.

Issues and contributions welcome via the repository's issue tracker.
