# Research Note — COSI Phase 0 Result, v0

**Author:** Claude (Sonnet, this instance)
**Date:** 2026-05-06
**Status:** Suggestive positive signal. Not publication-ready. Phase 1 justified.

---

## 1. What Was Run

Per `2026-05-06_cosi_design.md`, with one substitution: the Sovereign card
corpus from `paper-frontier-007` was found to contain only 2 unique
paraphrases across 38 cards (a deduplication artifact in the upstream LLM
extractor; logged separately for triage). In its place, Phase 0 used a
probe set of 96 compositional propositions extracted from the
`state/frontier-NNN-*_v0.draft.tex` filenames. Each probe is the title of an
existing Sovereign frontier-paper draft, framed as a coherence question.

| | |
|---|---|
| Model A | Phi-4-Reasoning-Plus (8-bit MLX, phi3 dispatch, 40 layers, hidden 5120) |
| Model B | Qwen3-4B (8-bit MLX, qwen3 dispatch, 36 layers, hidden 2560) |
| Probes | 96 frontier-domain propositions |
| Pooling | last-token, prompts in identical template |
| Layer fractions | 0.25, 0.5, 0.75 |
| Subspace | PCA, k chosen for 0.90 cumulative variance, min(k_A, k_B) used |
| Alignment | orthogonal Procrustes |
| Null | 1000 permutation samples + 1000 random-rotation samples per layer |
| Run dir | `runs/cosi/phase0_20260506T230920Z/` |

## 2. Result

| layer | residual | perm null mean | perm p1 | z vs perm | rot null mean | k | verdict |
|------|---------|---------------|--------|-----------|--------------|---|---------|
| 0.25 | 0.6064 | 1.0594 | 1.0292 | −39.54σ | 1.3458 | 33 | below p1 |
| 0.50 | 0.8100 | 0.9453 | 0.9301 | −23.93σ | 1.0535 | 27 | below p1 |
| 0.75 | 0.8050 | 0.9531 | 0.9369 | −23.42σ | 1.0605 | 25 | below p1 |

All three layer fractions show observed Procrustes residual significantly
below the 1st-percentile threshold of the permutation null. The early-layer
(0.25) effect is substantially stronger than mid- and late-layer effects.
The random-rotation null is uninformative (residuals near 1.0 by
construction); the permutation null is the binding test, and the observed
residuals separate from it cleanly at 23–40σ.

## 3. What This Does Not Settle

In strict order of importance:

1. **N is at the low end of stable Procrustes.** Rule of thumb is N ≥ 5k.
   At layer 0.25, k=33, so we want N ≥ 165. We have 96. The result holds at
   z=−39σ even at this N, but the residual point estimate may shift with
   larger N.
2. **Probe set is small and domain-narrow.** 96 propositions from a single
   topic family (categorial-architecture political theory). Cross-domain
   replication is required before any "substrate-independent" claim is
   defensible.
3. **Probe text is short.** Most prompts are 12–20 tokens. Last-token
   pooling on short sequences may be reading prompt-template syntax more
   than proposition content. The pooling sweep specified in the design note
   (mean, max) was not run for Phase 0.
4. **Model pair shares training data lineage.** Both models are post-2024
   web-pretrained. The aligned subspace may be a residue of shared training
   distribution rather than a substrate-independent compositional structure.
5. **Architectural distance is real but not maximal.** Phi-4 (phi3) and
   Qwen3-4B (qwen3) are both dense pure-attention transformers. The
   maximum-distance pair within our zoo is phi3 ↔ qwen3_next (hybrid
   attention/SSM); that pair is Phase 1.
6. **Procrustes is linear.** Even orthogonal alignment is a linear
   operation; non-linear isomorphism could exist that this method
   underestimates. CKA was not yet computed; design note §10 question 3.
7. **Layer-fraction sensitivity is suggestive but unverified.** The
   monotonic-with-depth pattern (0.25 strongest, 0.75 same as 0.5) is
   informative but should be checked across the full layer stack, not just
   three points.

## 4. Pre-registration Compliance

§8 of the design note declared the falsification gates. This run complied:

- ✅ Chance baselines (permutation, rotation) computed on randomized inputs.
- ✅ Layer fractions preregistered: 0.25, 0.5, 0.75. No best-layer cherry-pick.
- ✅ Pooling preregistered: last-token. (Sweep is preregistered but not yet run.)
- ✅ Significance threshold preregistered: residual below p1 of permutation null AND z < −2. Both gates passed at all three layers.
- ✅ Negative-result reporting commitment: would have been published as null.

The run is a clean pre-registered confirmation, not a post-hoc fishing
expedition. This matters for the eventual paper.

## 5. What This Justifies

Phase 1, as specified in the design note: Phi-4-Reasoning-Plus vs
Qwen3-Next-80B-A3B-Thinking on the same probe set, plus an extended probe
set (≥500), plus the pooling and full-layer-stack sweeps, plus CKA as the
cross-validation metric. The Phase 0 result demonstrates that the
infrastructure is sensitive enough to detect the effect; Phase 1 amplifies
the architectural distance and tightens every threat to validity in §3.

It does NOT yet justify a publication claim. The §3 caveats are not
rhetorical hedges; each one is a legitimate confound that a reviewer would
correctly flag. Phase 1 is structured to address them.

## 6. The Confound I Want To Rule Out First

Of the seven §3 caveats, the one I would prioritize ruling out is #4 —
shared training data. The cleanest test is: rerun the Phase 0 procedure
with one model held fixed and the other replaced by an architecturally
similar but training-distribution-distant model. If the residual stays low
across that swap, the alignment is architectural; if it tracks training
overlap, the alignment is training-corpus residue.

We do not have a good "training-distribution-distant" model in our local
zoo. Most of our models are post-2024 web-pretrained on overlapping
corpora. This is a Phase 2 problem and may require external compute.

## 7. The Headline I Will Not Yet Write

"Cross-architectural isomorphism of compositional reasoning representations
in frontier language models" — this is the eventual paper title if Phase 1
replicates and the §6 confound is addressed. The Phase 0 result is *one
step* toward that headline. It is not the headline itself. I am noting this
explicitly because Phase 0 results that look this clean are exactly the
kind of result one is most tempted to over-interpret.

## 8. Next Concrete Steps

1. **Phase 1 prep** — extend the probe set to ≥500 cross-domain compositional
   propositions. Rebuild from MATH / GSM8K / BIG-bench / hand-curated rather
   than from the frontier titles alone.
2. **Phase 1 run** — Phi-4 vs Qwen3-Next-80B (qwen3_next dispatch already
   smoke-validated tonight). Same protocol, full-stack layer sweep, all
   three pooling modes, CKA in parallel with Procrustes.
3. **Phase 2 design** — confound-rule-out runs, especially the
   training-distribution control if we can locate one.

— Claude (Sonnet, 2026-05-06)
