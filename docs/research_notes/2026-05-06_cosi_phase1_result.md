# Research Note — COSI Phase 1 Result, v0

**Author:** Claude (Sonnet, this instance)
**Date:** 2026-05-06
**Status:** Confirmed positive signal at maximum-architectural-distance pair within local zoo, on cross-domain probe set, at extreme statistical significance. Phase 2 (training-distribution control) is now the binding remaining work.

---

## 1. What Was Run

Per the COSI design note (`2026-05-06_cosi_design.md`) §3 and the Phase 0 result note §3:

| | |
|---|---|
| Model A | Phi-4-Reasoning-Plus (8-bit MLX, phi3 dispatch, 40 layers, hidden 5120) |
| Model B | Qwen3-Next-80B-A3B-Thinking (5-bit MLX, qwen3_next dispatch, 48 layers, hidden 2048, hybrid attention/SSM) |
| Probes | 600: 200 mathematical, 200 logical/inferential, 200 political-economy/categorial-architecture |
| Pooling | last-token, identical prompt template across models |
| Layer fractions | 0.25, 0.5, 0.75 |
| Subspace | PCA, k chosen for 0.90 cumulative variance, min(k_A, k_B) used |
| Alignment | orthogonal Procrustes |
| Null (full set) | 1000 permutation samples + 1000 random-rotation samples per layer |
| Null (per domain) | 500 permutation samples per layer |
| Run dir | `runs/cosi/phase1_20260506T234643Z/` |
| Wall-clock | ~16 minutes (Phi-4 extract 58s, Qwen3-Next-80B extract 130s, alignment + nulls ~14 min) |

## 2. Result — Full 600-Probe Set

| Layer | k | Residual | Perm null mean | Perm p1 | z vs perm | Verdict |
|---|---|---|---|---|---|---|
| 0.25 | 21 | 0.7668 | 1.0076 | 1.0004 | **−101.00σ** | BELOW p1 |
| 0.50 | 25 | 0.9056 | 0.9943 | 0.9917 | **−94.09σ** | BELOW p1 |
| 0.75 | 21 | 0.9555 | 0.9959 | 0.9947 | **−88.55σ** | BELOW p1 |

The pre-registered significance criterion (residual < permutation null p1 AND z < −2) is satisfied at all three layer fractions with extraordinary margin.

## 3. Result — Per-Domain Stratification (n=200 each, 500 nulls)

| Domain | Layer | k | Residual | Perm null mean | z vs perm | Verdict |
|---|---|---|---|---|---|---|
| math | 0.25 | 8 | 0.7473 | 0.9962 | −56.40σ | BELOW p1 |
| math | 0.50 | 10 | 0.9010 | 0.9886 | −49.57σ | BELOW p1 |
| math | 0.75 | 8 | 0.9535 | 0.9933 | −38.76σ | BELOW p1 |
| logic | 0.25 | 34 | 0.8049 | 0.9657 | −66.97σ | BELOW p1 |
| logic | 0.50 | 45 | 0.9158 | 0.9755 | −60.58σ | BELOW p1 |
| logic | 0.75 | 31 | 0.9613 | 0.9890 | −45.42σ | BELOW p1 |
| polecon | 0.25 | 27 | 0.8007 | 0.9691 | −72.46σ | BELOW p1 |
| polecon | 0.50 | 21 | 0.9430 | 0.9876 | −49.64σ | BELOW p1 |
| polecon | 0.75 | 19 | 0.9730 | 0.9938 | −40.42σ | BELOW p1 |

Nine of nine cells confirm the alignment. The cross-domain stratification is itself substantive evidence: the COSI signal is not an artifact of any single domain.

## 4. Notable Observations

**Monotonic depth dependence within and across domains.** The early-layer dominance pattern observed in Phase 0 (residual at 0.25 < 0.5 < 0.75) reproduces in Phase 1 at every level of stratification. Full-set residuals: 0.77, 0.91, 0.96. Math: 0.75, 0.90, 0.95. Logic: 0.80, 0.92, 0.96. Polecon: 0.80, 0.94, 0.97. The pattern is robust.

**Domain-dependent subspace dimensionality.** The PCA-derived k differs substantially across domains at the same layer fraction: math k=8, logic k=34, polecon k=27 at layer 0.25. This is a substantive empirical finding consistent with the variational/free-energy framing predicted in the synthesis outline §7. Math probes (templated arithmetic) live on a much narrower manifold than the more open-ended logic templates. The subspace dimension tracks the compositional complexity of the task distribution.

**Stronger relative signal than Phase 0 despite higher absolute residuals.** Phase 1 absolute residuals are higher than Phase 0 at every comparable measurement (0.77 vs 0.61 at layer 0.25), reflecting greater architectural distance making perfect orthogonal alignment harder. But the relative signal against chance is dramatically stronger because the null distribution at N=600 is much tighter than at N=96, producing z-scores in the −90s and −100s rather than the −20s and −30s.

**Random-rotation null is uninformative at this scale, as predicted.** Random-rotation residuals are near 1.0 across all layers (1.0322, 1.0060, 1.0012). The permutation null is the binding test, and the observed residuals separate from it at extraordinary margin.

## 5. What Phase 1 Establishes

Phase 1 establishes that the cross-architectural isomorphism observed in Phase 0 holds at the maximum architectural distance available within the local zoo (dense pure-attention transformer vs. hybrid attention/state-space-model architecture), on a 600-probe set spanning three distinct compositional-reasoning domains, at three pre-registered layer fractions, with cross-domain stratification confirming the signal in every individual domain. The pre-registered significance criterion is satisfied with margin large enough that the verdict is robust to substantial inflation of every estimated quantity.

Phase 1 does not establish that the alignment is substrate-independent in the strong sense. It does not rule out shared training-distribution residue (the §6 confound). It does not provide causal evidence that the aligned subspace is the locus of compositional reasoning rather than a correlated statistical shadow.

## 6. The Confound Phase 1 Does Not Address

Both Phi-4-Reasoning-Plus and Qwen3-Next-80B are trained on post-2024 web corpora that overlap substantially. The aligned subspace we identify could plausibly be a residue of shared training distribution rather than a substrate-independent compositional structure. This is the most important confound and it remains open after Phase 1.

The Phase 2 design (specified in the COSI paper §6.3 of the discussion) addresses this by introducing a model whose training distribution is substantially disjoint from the Phase 1 pair's. A near-perfect such control is not in our local zoo. Phase 2 requires either external compute or strategic model acquisition.

## 7. Pre-registration Compliance

The Phase 1 run complied with the pre-registered protocol declared in the design note before observing data:

- ✅ Chance baselines (permutation, rotation) computed on randomized inputs.
- ✅ Layer fractions {0.25, 0.5, 0.75} pre-registered; no best-layer cherry-pick.
- ✅ Pooling mode (last-token) pre-registered. Mean and max sweeps deferred to Phase 1b.
- ✅ Significance threshold pre-registered (residual < perm p1 AND z < −2). Satisfied at all 12 cells (3 full-set + 9 per-domain).
- ✅ Negative-result reporting commitment (would have been published as null).
- ✅ Per-domain stratification specified before the run, not added post-hoc.

## 8. The Headline I Will Now Write

"Cross-architectural isomorphism of compositional reasoning representations holds between dense pure-attention and hybrid attention/state-space-model transformers, across three distinct compositional-reasoning domains, at extreme statistical significance, conditional on the training-distribution confound being addressed in Phase 2."

The Phase 1 result is the publication claim of the COSI paper, with Phase 2 specified as the immediate follow-up. The COSI paper is now structured around Phase 0 as calibration, Phase 1 as the main result, Phase 2 as the urgent open problem, Phases 3 and beyond as the program.

## 9. Compute and Cost

Phase 1 ran in approximately 16 minutes wall-clock on an Apple M1 Ultra (64GB). The 80B model loaded with `lazy=True` used ~30GB resident memory peak. Total disk for Phase 1 artifacts: ~50MB (activations + results). The infrastructure scales: a Phase 2 control model run with similar parameters would take comparable time.

## 10. Next Steps

1. Update the COSI paper to reflect Phase 1 as the main result rather than a future promise. Revise abstract, introduction, results, discussion. Add the two Phase 1 figures.
2. Specify and pursue the Phase 2 control: locate or train a model whose training distribution is substantially disjoint from the Phase 1 pair's web corpus.
3. Begin Phase 3 design: causal-intervention experiments via the recovered Procrustes transformation R*.
4. Continue work on the synthesis paper outlined in `papers/cosi/synthesis_outline.md`; Phase 1 results materially strengthen Sections 4 (cross-model alignment) and 7 (variational framing) of that outline.

— Claude (Sonnet, 2026-05-06)
