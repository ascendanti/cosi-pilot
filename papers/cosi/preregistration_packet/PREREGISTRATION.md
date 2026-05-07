# COSI Cross-Architectural Activation Alignment Pilot — Pre-Registration

**Project:** Cross-Operator Subspace Isomorphism (COSI) — Representational-Alignment Pilot
**Authors:** Adam A. Bensaid (Atlas Consulting & Technology Services); Claude (Anthropic)
**Date of pre-registration:** [TO BE FILLED ON UPLOAD — should be the moment this packet is deposited to OSF/Zenodo, BEFORE any new data is collected]
**Pre-registration platform:** OSF (Open Science Framework) — recommended URL pattern `https://osf.io/...`
**Companion paper:** `papers/cosi/main.tex`

---

## 1. Background and Motivation

The mechanistic interpretability literature has documented that sparse
autoencoders (SAEs) recover monosemantic feature dictionaries from trained
transformer language models, with features broadly conserved across
architecturally different models. The Platonic Representation Hypothesis
(Huh et al. 2024) generalizes this convergence claim. We hypothesize an
operator-theoretic underpinning: that the recovered features are concrete
realizations of approximately invariant subspaces of the nonlinear forward
operator, and that two architecturally different trained transformers
should therefore exhibit alignable invariant-subspace structure on shared
compositional probes.

The present pilot tests one operationalization of that hypothesis:
Procrustes alignment of PCA-projected last-token activation matrices,
between architecturally different frontier transformers, against a
row-permutation null. The pilot does not test the strong operator-theoretic
claim directly; it tests a weaker representational-alignment claim that
the strong claim would predict.

---

## 2. Hypothesis

**H1 (representational-alignment):** On matched compositional-evaluation
prompts, the PCA-projected last-token activation matrices of two
architecturally different frontier transformers admit an orthogonal
alignment with Procrustes residual significantly below the chance level
defined by row-permutation null.

**H1 status:** Pilot hypothesis. A positive H1 result is consistent with
both (a) the strong operator-theoretic claim and (b) several weaker
alternatives (shared training-distribution residue, shared lexical/template
encoding, last-token pooling artifact). The pilot does not discriminate
between these. Studies A through G (companion paper §6) specify the
discriminating controls.

---

## 3. Experimental Protocol

### 3.1 Models

**Phase 0:**
- $M_A$: `mlx-community/Phi-4-reasoning-plus-8bit` (40 layers, hidden 5120)
- $M_B$: `mlx-community/Qwen3-4B-8bit` (36 layers, hidden 2560)

**Phase 1:**
- $M_A$: `mlx-community/Phi-4-reasoning-plus-8bit` (as above)
- $M_B$: `mlx-community/Qwen3-Next-80B-A3B-Thinking-5bit` (48 layers, hidden 2048, hybrid attention/SSM)

Model checkpoints are immutable HuggingFace artifacts; revision pins are
to be recorded in the run manifest at execution time.

### 3.2 Probe Set

**Phase 0:** 96 probes drawn from frontier-NNN paper draft titles in the
Sovereign research corpus, formatted as coherence-evaluation prompts. The
probe set is fixed in `data/cosi/probe_set_frontier_v1.jsonl` with content
hash `[TO BE COMPUTED ON UPLOAD]`.

**Phase 1:** 600 probes generated programmatically via
`scripts/build_phase1_probes.py` with seed 20260506. 200 probes per domain
(math, logic, polecon). Probe set fixed in
`data/cosi/probe_set_phase1_v1.jsonl` with content hash
`[TO BE COMPUTED ON UPLOAD]`.

### 3.3 Activation Extraction

Per-architecture replication of the inner-model forward pass (no
modifications to `mlx_lm` source). Residual stream captured after layer
blocks at pre-registered layer fractions $f \in \{0.25, 0.5, 0.75\}$.
Last-token pooling. bfloat16 → float32 casting before host crossing.

Implementation: `src/sovereign/research/cosi/extract.py`, with
architecture dispatches for `phi3`, `qwen3`, `qwen3_next`.

### 3.4 Subspace Projection

Each model's activation matrix is projected onto its top-$k$ principal
components, where $k$ is the smaller of the two models' minimum dimension
required to retain $0.90$ cumulative variance.

### 3.5 Procrustes Alignment

Orthogonal Procrustes minimization in the shared $k$-dimensional subspace,
via SVD. Procrustes residual is $\rho = \|X_A - X_B R^*\|_F / \|X_A\|_F$.

Implementation: `src/sovereign/research/cosi/align.py`.

### 3.6 Null Distributions

**Permutation null:** $S = 1000$ random permutations of $X_B$'s rows
(Phase 0 and Phase 1 full set); $S = 500$ for per-domain stratification.
Recompute Procrustes residual on each permuted pair.

**Random-rotation null:** $S = 1000$ random orthogonal $R \in O(k)$
sampled via QR decomposition with sign correction. Reported for
completeness; not the binding test.

### 3.7 Significance Criterion

Pre-registered:

1. The observed Procrustes residual is below the first percentile of the
   permutation null distribution.
2. The standardized $z$-score of the observed residual against the
   permutation null mean is at most $-2$.

A run that fails either condition is reported as a null result and
published.

### 3.8 Sensitivity Sweeps (deferred but pre-registered)

The following sensitivity sweeps are part of the protocol but reserved
for follow-up runs (Study F of the companion paper):

- Pooling mode: last-token (primary), mean, max.
- Layer fraction: full layer-by-layer sweep.
- PCA variance threshold: 0.85, 0.90 (primary), 0.95.
- Whitening: with and without.
- CKA cross-validation alongside Procrustes (Kornblith et al. 2019).

---

## 4. Pre-Registration Commitments

1. The experimental protocol specified in §3 is fixed before any new data
   collection. Any deviation from the protocol will be reported as such.
2. All results, positive or null, will be reported. The publication of the
   companion paper is not contingent on the result direction.
3. Code, data, and analysis artifacts are released under the public
   repository at `[TO BE FILLED]` with content hashes recorded in this
   pre-registration document.
4. The strong operator-theoretic claim (cross-architectural isomorphism
   of compositional reasoning subspaces) is NOT tested by this protocol;
   the pilot tests only the weaker representational-alignment claim. The
   companion paper specifies Studies A–G that would, jointly, support the
   strong claim. None of Studies A–G is performed under this
   pre-registration; each will require its own pre-registration.

---

## 5. Artifact Manifest

The following artifacts constitute the pre-registered protocol and its
implementation. All file content hashes are to be computed and recorded at
the moment of OSF/Zenodo upload, before any data is collected under this
pre-registration.

### Code (`src/sovereign/research/cosi/`)
- `extract.py` — activation extraction harness
- `align.py` — Procrustes alignment, permutation null, random-rotation null

### Runners (`scripts/`)
- `cosi_phase0.py` — Phase 0 run script
- `cosi_phase1.py` — Phase 1 run script
- `build_phase1_probes.py` — Phase 1 probe generator (seed 20260506)
- `cosi_smoke.py`, `cosi_smoke_80b.py` — architecture-dispatch smoke tests
- `cosi_figures.py` — figure generation

### Probe Sets (`data/cosi/`)
- `probe_set_frontier_v1.jsonl` — Phase 0 (96 probes)
- `probe_set_phase1_v1.jsonl` — Phase 1 (600 probes)

### Pre-Registration Document
- `papers/cosi/preregistration_packet/PREREGISTRATION.md` (this file)
- `papers/cosi/preregistration_packet/CHECKSUMS.txt` — computed at upload

### Environment
- `papers/cosi/preregistration_packet/environment.txt` — pip freeze, MLX version, Python version

---

## 6. Authorial Note

This pre-registration was drafted by the language model co-author on
2026-05-07 and reviewed by the human author. The human author is
responsible for upload, content-hash computation, and final verification.
The pre-registration is binding once deposited; subsequent runs of the
specified protocol will reference the deposit DOI in their reports.

The pilot was already executed before this pre-registration was
finalized, owing to the compressed timeline of the original session in
which the work was done. The pre-registration is therefore *not* a true
pre-registration of the pilot itself; it is a retrospective formalization
of the protocol that the pilot followed, intended to lock that protocol
in advance of any future replication or extension. Future runs (Study A
through Study G of the companion paper) will be pre-registered under
their own deposits before execution.

This honest accounting is itself a methodological commitment: subsequent
runs will be pre-registered before data collection, with deposit
timestamps that precede any analysis.
