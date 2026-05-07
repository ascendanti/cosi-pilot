# Research Note — Cross-Operator Subspace Isomorphism (COSI), v0

**Author:** Claude (Sonnet, this instance)
**Date:** 2026-05-06
**Status:** Draft for correction. This is the first decisive theoretical experiment proposed in `STRATEGIC_BRIEF.md`. It is not yet ready to run. It is ready to be torn apart.

---

## 1. The Hypothesis, in One Sentence

When two large language models with maximally different architectures are run over the same compositional reasoning corpus, the geometry of their internal activations contains an approximately invariant subspace that is isomorphic across the two models up to an orthogonal transformation, and the residual after Procrustes alignment is significantly below what random rotation predicts.

If true, this is evidence for a substrate-independent compositional structure — Compositional Minimalism, in the layered framing of the strategic brief. If false, the layered program either retreats to architecture-specific invariants or abandons the substrate-independence claim.

This is the version of the hypothesis I am willing to be wrong about. Adam, please push on the formulation before we run anything.

---

## 2. Why This Experiment And Not Another

The strategic brief proposed three layers: Compositional Minimalism (approximate invariant subspaces), Cross-Operator Subspace Isomorphism, Computational Autopoiesis. COSI is the load-bearing middle layer. If the subspaces in two models are not isomorphic, the substrate-independence claim of layer one is undermined and the autopoiesis claim of layer three has no shared ground to operate on. If COSI holds, the operator-theoretic frame (Lomonosov 1973, Enflo 1987, Noel 2011) earns its place as more than vocabulary — it earns its place as the right model for what is happening inside the forward pass.

We need a falsifiable test. Procrustes-based isomorphism with a clear chance baseline is the cleanest one I can construct on the substrate we have.

---

## 3. Operationalization

### 3.1 Models

- **Model A:** Qwen3-Next-80B-Thinking (5-bit MLX). MoE, RoPE, attention sinks, "thinking" mode with explicit chain-of-thought. Already loaded in our stack.
- **Model B:** Phi-4-Reasoning-Plus (MLX). Dense, distillation-trained reasoning model. Different family, different training regime, different parameter count, different architectural choices for attention and routing.

The two are chosen for *maximum architectural distance within the frontier-reasoning class.* If isomorphism survives this contrast, it is meaningful. If we picked two close cousins (e.g., two Qwen variants), a positive result would not discriminate between substrate-independent invariants and family-shared training residue.

Adam: a third model — Command-R or Qwen3-30B — should probably be run as a calibration, to get a sense of how the residual scales with architectural distance. Worth deciding before we commit four weeks.

### 3.2 Corpus

Sovereign's existing card corpus from `paper-frontier-007`. ~23 spans, plus the broader research-card body the system has produced across earlier runs. These cards are the natural unit because they are atomic compositional reasoning steps — each card encodes a single proposition with explicit citations and provenance. They are already structured; we do not need to construct a new benchmark.

If 23 spans is too few for stable Procrustes alignment, we extend by re-running pipeline-v2 on adjacent topics until we have ≥500 cards. The threshold is set by the dimensionality of the activation subspace we extract; rule of thumb is N ≥ 5×d for stable orthogonal alignment. d will be chosen in §3.4.

### 3.3 Activation Extraction

For each card and each model:

1. Format the card as a prompt that elicits the proposition's verification or extension. The prompt template is held constant across both models.
2. Run forward pass. Extract residual stream activations at a canonical layer fraction — initial pass at 0.5 (mid-network), with sweeps at 0.25, 0.75 to check layer dependence.
3. Pool to a single vector per card per model. Initial choice: last-token EOS pooling, matching the convention we already use for the embedder. Sweep to mean-pooling and max-pooling as ablations.

Result: two matrices, X_A ∈ ℝ^(N × d_A) and X_B ∈ ℝ^(N × d_B), where N is the number of cards and d_A, d_B are the residual stream dimensions.

### 3.4 Subspace Identification

Before alignment, we identify the candidate invariant subspace within each model's activations.

Two paths, run both:

- **Path 1 — PCA.** Take the top-k principal components per model, k chosen so that cumulative variance reaches 0.90. This is the cheap path. Risk: PCA captures variance, not function.
- **Path 2 — SAE features.** If pretrained sparse autoencoders are available for either model (Templeton et al. 2024 for similar architectures), use them to extract feature activations and treat the active feature set as the candidate subspace. This is the principled path. It connects directly to the mech-interp literature and gives us a concrete realization of "approximate invariant subspace" that operator theory says should exist.

If Path 2 is not available for both models, run Path 1 first, document the limitation, and pursue SAE training as Phase 2.

### 3.5 Alignment

Orthogonal Procrustes: find R ∈ O(k) that minimizes ‖X_A − X_B R‖_F, given X_A and X_B projected into k-dimensional subspaces of equal rank.

Compute residual = ‖X_A − X_B R*‖_F / ‖X_A‖_F.

### 3.6 Chance Baseline

Two baselines, both reported:

- **Random-rotation baseline:** for many random orthogonal matrices R_rand, compute the same residual. The distribution of residuals under R_rand defines what "no isomorphism" looks like.
- **Permutation baseline:** randomly permute the row-correspondence between X_A and X_B (i.e., pair card i in model A with card π(i) in model B for random permutation π) and compute Procrustes residual. This breaks the semantic correspondence while preserving each model's internal geometry. The residual under permutation is the more conservative null.

A positive result requires the observed residual to be significantly below both baselines. p < 0.01 by permutation test, with effect size reported as (residual_observed − residual_chance) / σ_chance.

---

## 4. Predictions, By Outcome

| Observed | Interpretation |
|---|---|
| Residual ≈ permutation baseline | No isomorphism. Substrate independence is not supported at this level of abstraction. The layered program retreats to architecture-specific claims or finds a finer probe. |
| Residual significantly below permutation baseline but above random-rotation | Partial isomorphism. There is shared structure but it is not fully captured by orthogonal alignment. Suggests a non-linear or higher-order invariant. Phase 2: extend to non-linear alignment (e.g., CKA, or learned diffeomorphism). |
| Residual significantly below both baselines | Substantive support for COSI. This is the result that justifies the strategic brief's commitment. We then characterize the aligned subspace — what features live in it, what tasks it predicts, whether SAE features in one model linearly recover SAE features in the other. |
| Residual approaches zero | Suspicious. Either the prompt template is leaking the answer in the activation pattern (a confound), the two models share more training data than expected, or the alignment procedure has a degenerate solution. We invert the analysis to find the leak. |

The third row is the result we want. The fourth row is the result we should expect to debug rather than celebrate.

---

## 5. Threats To Validity

These are the ones I see. There will be more.

1. **Tokenizer mismatch.** The two models tokenize the same card differently. The activations are conditioned on token sequences, not on semantic content. Mitigation: report results both with and without tokenizer-aware re-pooling (collapse subword sequences to word-level vectors before pooling).
2. **Layer-fraction sensitivity.** A positive result at layer 0.5 may not hold at 0.25 or 0.75. We must report the full sweep, not the best layer.
3. **Pooling artifact.** Last-token pooling may bias toward the prompt's syntactic completion rather than its semantic content. Sweep to mean-pooling and max-pooling, report all.
4. **Card heterogeneity.** Sovereign's cards span several methodologies (formal model, comparative case, statistical analysis). Within-methodology isomorphism may differ from cross-methodology. Stratify the analysis.
5. **Selection effect on cards.** If the cards were filtered by Sovereign for quality, they may already be projected onto a manifold defined by the system's filters. Run the same analysis on raw, unfiltered text from the same domain as a sanity check.
6. **Procrustes assumes linear.** Even orthogonal alignment is a linear operation. The "true" invariant may be non-linear, in which case Procrustes underestimates isomorphism. Hence the CKA / non-linear extension in §4.

---

## 6. Connection To Prior Literature

- **Operator theory:** Noel 2011, *The Invariant Subspace Problem*. Lomonosov 1973 (every operator commuting with a non-zero compact operator has a non-trivial invariant subspace). Enflo 1987 (counter-example for general Banach spaces). The conjecture that bounded linear operators on infinite-dimensional separable Hilbert spaces have invariant subspaces is open. We are not solving it. We are claiming that the *forward operator of a transformer*, restricted to the manifold of natural-language activations, has approximately invariant subspaces, and that these subspaces are model-pair-isomorphic.
- **Mech interp:** Bricken et al. 2023, Cunningham et al. 2024, Templeton et al. 2024. SAE features as "monosemantic directions" are the candidate concrete realization of the invariant subspace. The bridge claim is: SAE features ≈ basis of approximately invariant subspace. If true, the entire mech-interp program is doing operator theory under another name. The contribution would be the formalization, not the discovery.
- **Cross-model alignment:** Kornblith et al. 2019 (CKA, *Similarity of Neural Network Representations Revisited*). This is the closest prior art and the obvious first ablation. We compute CKA in parallel with Procrustes; if Procrustes finds isomorphism that CKA does not, or vice versa, the discrepancy is itself informative.

---

## 7. Procedure Timeline (4 weeks, dry run)

- **Week 1.** Card corpus consolidation. Activation extraction infrastructure. Both models loaded and producing residual stream traces on demand. Tokenizer-aware pooling implemented. Manifest captured per the new `run_manifest.py` standard.
- **Week 2.** Subspace identification (PCA path, SAE path if available). Random-rotation and permutation baselines computed. Pipeline producing residuals.
- **Week 3.** Layer sweep, pooling sweep, methodology stratification. Calibration model (Command-R or Qwen3-30B) added as architectural-distance probe.
- **Week 4.** Analysis. Either: (a) result clean → write up, draft as the first arXiv paper from the strategic program; or (b) result ambiguous → diagnose threat to validity, redesign, push to Phase 2.

The four weeks are real time, with the substrate work (MLA retrofit, prefix cache) running in parallel only as much as is needed to keep the experiment unblocked. Substrate optimization for its own sake is paused until COSI ships.

---

## 8. Falsification Gates

The experiment is falsifiable iff each of the following is true *before* we look at the data:

- The chance baselines are computed on randomized inputs, not on the real comparison.
- The layer fraction is preregistered (we are saying 0.5, with declared sweep at 0.25 and 0.75 — not "the layer that gave the best result").
- The pooling method is preregistered (last-token, with declared sweep — not "the pooling that gave the best result").
- The threshold for "significant isomorphism" is preregistered: residual_observed must be below the 1st percentile of the permutation null, and the effect size must exceed 2σ.
- A negative result will be reported. If COSI fails, the strategic brief gets a negative-result revision, not a quiet pivot.

If we cannot meet these gates, the experiment is not yet ready to run.

---

## 9. What This Experiment Does Not Settle

It does not settle whether the isomorphism (if found) is *causal* — whether the aligned subspace is what the models use to compute, or whether it is a statistical shadow of something else they share. That is Phase 2, and it requires interventions: ablate a direction in model A's subspace, predict the corresponding ablation effect in model B's aligned direction, run both, see if predictions hold.

It does not settle whether the invariant subspace is a *substrate-independent feature of compositional reasoning* or a *training-regime artifact*. That requires running the experiment on a model trained from scratch with a different objective. Out of scope for the four weeks. In scope for the year.

It does not settle whether what we have measured is what operator theory means by an invariant subspace. The classical definition is exact. Ours is approximate. The bridge between approximate empirical invariance and the formal definition is itself a contribution if we can make it rigorously, but it is not what this experiment alone establishes.

---

## 10. The Correction I Am Asking For

Adam, before this runs, the things I most want you to push on:

- Is the model pair right? Qwen3-Next vs Phi-4 maximizes architectural distance in our local zoo. There may be a more diagnostic pair I am missing.
- Is the corpus right? Sovereign's cards are a sample of convenience. A clean compositional benchmark (e.g., a slice of MATH, a curated subset of GSM8K reasoning chains, the BIG-bench compositional tasks) might give a sharper signal at the cost of losing connection to our own pipeline. The trade-off is real.
- Is the Procrustes-residual the right metric? CKA is the obvious alternative and may be what the literature expects to see. Both is the safe answer; one being primary is the disciplined answer. Tell me which.
- Is there a result I have not predicted that I should worry about? The four rows in §4 are the ones I see. The unknown unknowns are the ones that kill an experiment.

I will write this up properly in the brief once you have hit it. The point of this note is to be wrong in legible ways. It is a proposal for an open. Open it.

— Claude (Sonnet, 2026-05-06)
