# Synthesis Paper Outline — *The Operator-Theoretic Substrate of Compositional Computation*

**Working title.** *On the Invariant Substrate: An Operator-Theoretic Framework for Compositional Computation in Self-Observing Systems*

**Authors.** Adam A. Bensaid, Atlas Consulting & Technology Services. With Claude (Anthropic).

**Status.** Outline, 2026-05-06. The COSI paper (this directory's `main.tex`) is the empirical anchor; the synthesis paper is the theoretical work the COSI program is building toward. Estimated final length: 40–60 pages, 100–150 references. Written across the next 3–6 months as Phase 1 / Phase 2 / Phase 3 results accumulate.

**Premise.** Multiple independent literatures have, over the past seventy years, accumulated evidence for a single underlying claim: that compositional computation in finite-capacity substrates produces an approximately invariant subspace structure that is preserved across substrate, scale, and architectural choice. The literatures have not engaged each other; the underlying claim has not been articulated as a single hypothesis with a single mathematical framework. This paper articulates the framework, traces its empirical instantiations across the literatures, and reports the empirical program (COSI Phases 0–3) that tests it directly in the language-model substrate.

---

## Section 1 — The Claim, Stated

One sentence each, ten claims, in increasing scope. Begin with the most narrow empirical claim and end with the broadest theoretical one. Each claim is a load-bearing element of the synthesis; each will be defended in the corresponding section below.

1. Trained transformers admit residual-stream activations that decompose into sparse, monosemantic feature dictionaries (empirical, well established).
2. These dictionaries are approximately invariant under the forward pass on representative samples (the operator-theoretic reading of the empirical observation).
3. The invariant subspaces are isomorphic across architecturally distinct trained models on shared probe sets (the COSI hypothesis).
4. The isomorphism is not an artifact of shared training distribution (the Phase-2 confound to be controlled).
5. The aligned subspace is causally responsible for compositional reasoning behavior, not merely correlated with it (the Phase-3 intervention claim).
6. The same approximately invariant subspace structure appears in biological neural populations engaged in analogous tasks (the computational-neuroscience bridge).
7. The structure is the empirical signature of variational free-energy minimization in finite-capacity substrates (the free-energy framing).
8. The structure is the empirical signature of cybernetic requisite variety (Ashby) realized in trained substrates.
9. The structure is what classical operator theory predicts in the linear case and what its nonlinear extensions, where they exist, would predict in the nonlinear case.
10. Compositional reasoning is therefore substrate-independent in a sharper sense than has been previously available: not merely "any sufficient substrate can implement it" but "any sufficient substrate must implement it via approximately the same invariant-subspace decomposition."

---

## Section 2 — Operator Theory: The Mathematical Backbone

**Thesis.** The classical theory of invariant subspaces of bounded operators on Hilbert spaces, despite remaining open in its strongest formulation, provides the right structural language for the empirical regularities the field is documenting. The framework's affirmative results (Aronszajn-Smith for compact operators, Lomonosov for operators commuting with a non-zero compact, the recent Argyros-Haydon constructions) constitute the linear-case theorem-base on which the nonlinear empirical work must build.

**Key citations.** Lomonosov 1973, Enflo 1987, Aronszajn-Smith 1954, Read 1985, Argyros-Haydon 2011, Beauzamy 1988, Halmos 1970. Noel 2011 (Honours thesis — direct intellectual debt).

**Concrete claim.** The forward operator of a trained transformer, restricted to the manifold of natural-language activations, admits approximately invariant subspaces whose dimension equals the dimension of the recovered SAE feature dictionary up to the superposition redundancy factor.

**What the section commits us to.** Defending the analogy between approximate empirical invariance (sample-level) and exact mathematical invariance (operator-level) rigorously. Specifying the conditions under which the linear theorems apply at the linearized first-order operator at a given activation, and the open question of how to extend this to the global nonlinear case.

---

## Section 3 — Mechanistic Interpretability: The Empirical Front

**Thesis.** The mechanistic interpretability program has been performing operator theory empirically for two years without naming it. The SAE features are bases for approximately invariant subspaces; the recovered circuits are descriptions of how the subspaces are computed; the universality findings are the empirical confirmation that the subspaces are substrate-independent at finite training-distribution overlap.

**Key citations.** Olah 2020 (Zoom In), Elhage 2021 (mathematical framework), Olsson 2022 (induction heads), Wang 2023 (IOI circuit), Geva 2021 (FFN as KV memory), Meng 2022 (ROME), Conmy 2023 (ACDC), Elhage 2022 (superposition), Bricken 2023 (monosemanticity), Cunningham 2024, Templeton 2024 (scaling monosemanticity), Gao 2024 (scaling SAEs), Lieberum 2024 (Gemma Scope), Marks 2024 (sparse feature circuits), Chughtai 2023 (toy universality), Gurnee 2024 (universal neurons), Nanda 2023 (grokking).

**Concrete claim.** The polysemanticity / superposition phenomenon is the finite-rank approximation problem: the network's invariant subspaces have dimension exceeding the neuron count, requiring the sparse-coding solution. The universality findings are the empirical confirmation of approximate substrate-independence at the level of the recovered subspace bases.

**What the section commits us to.** Demonstrating concretely (with worked examples from at least three of the cited circuit papers) that the recovered circuits can be re-described in operator-theoretic language without loss of explanatory content.

---

## Section 4 — Cross-Model Alignment: The Convergence Front

**Thesis.** The independent literature on cross-model representational similarity (CKA, SVCCA, model stitching, embedding alignment, Platonic Representation Hypothesis) has documented exactly the convergence the operator-theoretic framework predicts. The results have been treated as empirical regularities; we read them as evidence for the underlying structural fact.

**Key citations.** Mikolov 2013 (word embedding alignment), Smith 2017, Conneau 2017 (orthogonal Procrustes for bilingual lexicon), Lenc-Vedaldi 2015, Raghu 2017 (SVCCA), Morcos 2018 (PWCCA), Kornblith 2019 (CKA), Williams 2021 (generalized shape metrics), Bansal 2021 (model stitching), Klabunde 2024 (survey), Huh 2024 (Platonic Representation Hypothesis).

**Concrete claim.** The Platonic Representation Hypothesis is the operator-theoretic isomorphism claim restated in distributional language. The convergence the field has documented is the convergence the framework predicts as architectures and training distributions widen across a substrate-independent target.

**What the section commits us to.** Explaining why CKA and Procrustes verdicts sometimes diverge (the answer: the orthogonal-vs-isotropic-vs-anisotropic invariance hierarchy reflects different approximations to the operator-theoretic isomorphism, each appropriate under different assumptions about what should be preserved). Reporting both metrics in COSI Phase 1.

---

## Section 5 — Sequence-Model Lineage and the Architecture-Independence Argument

**Thesis.** The progression LSTM → S4 → Transformer → Mamba → hybrid attention/SSM is a sequence of progressively-better approximations to the same target: maintaining a finite-state representation of an arbitrary-history process whose statistics are governed by language. The fact that the subspace structure is preserved across architectural classes (the Phase 1 prediction) is what would be expected if the architectures are converging on the same invariant target.

**Key citations.** Hochreiter-Schmidhuber 1997 (LSTM), Vaswani 2017 (transformer), Gu-Goel-Ré 2022 (S4), Gu-Dao 2023 (Mamba), Ji 2025 (MLA retrofit). Plus the Q/K/V architectural decomposition as a specific operator-theoretic move (key-value memory as finite-rank approximation to the resolvent of the forward operator).

**Concrete claim.** The KV cache works because invariant subspaces exist. If the substrate did not have approximately invariant low-dimensional structure, the cache would not produce the speedups it produces; the empirical fact of cache effectiveness is itself indirect evidence for the operator-theoretic claim.

**What the section commits us to.** A worked example showing the KV cache hit rate as a function of context length and prompt structure, demonstrating that hit rate tracks the approximately-invariant structure of the prompt distribution rather than scaling indifferently.

---

## Section 6 — Computational Neuroscience: The Biological Front

**Thesis.** Biological neural populations have been documented to operate on low-dimensional manifolds with stable geometric structure across animals, behaviors, and even species. The same operator-theoretic framework that organizes the artificial findings should organize the biological ones.

**Key citations.** O'Keefe 1971 (place cells), Hafting 2005 (grid cells), Yamins-DiCarlo 2016 (deep nets predict cortex), Gallego 2017 (neural manifolds for movement), Saxena-Cunningham 2019 (neural population doctrine), Tang 2019 (cross-individual fMRI similarity).

**Concrete claim.** Place cells and grid cells implement approximately invariant subspaces of the hippocampal-entorhinal forward operator with respect to spatial position; the geometric structure is preserved across animals because the operator-theoretic constraint is preserved. The corresponding artificial-network finding is that goal-driven networks trained on spatial tasks recover the same manifold structure.

**What the section commits us to.** Specifying the COSI methodology's extension to biological-artificial alignment (Phase 5 of the program). Sketching the experimental protocol: neural population recording during a compositional task, transformer activation matrix on the same task, Procrustes alignment between the two with appropriate dimensionality matching.

---

## Section 7 — Free-Energy, Information Bottleneck, Cybernetic Variety: The Variational Front

**Thesis.** The free-energy principle (Friston) and the information-bottleneck objective (Tishby) predict that any self-maintaining computational system must develop internal states approximating the posterior over its environment's hidden causes, and that these internal states are substrate-independent because the variational objective is. Ashby's law of requisite variety provides the structural constraint: the substrate must contain at least as much variety as the regularity it represents. The operator-theoretic invariant subspaces are the implementation of the variational solution under capacity constraints.

**Key citations.** Shannon 1948 (information theory), Tishby-Zaslavsky 2015 (information bottleneck deep learning), Friston 2010 (free-energy principle), Achille-Soatto 2018 (emergence of invariance), Ashby 1956 (requisite variety), Wiener 1948 (cybernetics), Maturana-Varela 1980 (autopoiesis).

**Concrete claim.** The dimensionality $k$ of the empirically-recovered invariant subspace should track the entropy of the task distribution (Tishby) rather than the dimensionality of the substrate. The COSI Phase-1 layer-fraction sweep and the cross-domain probe-set replication will test this directly: the recovered $k$ should be domain-dependent and substrate-independent if the variational framing is correct.

**What the section commits us to.** A concrete prediction: $k$ should scale logarithmically with the task's compositional complexity (number of distinct atoms times number of compositional operations) rather than linearly with the substrate's hidden dimension. We do not know whether this prediction holds; the synthesis paper commits us to running the test.

---

## Section 8 — Linguistics and Compositionality: The Theoretical-Computer-Science Front

**Thesis.** The compositional structure of language (Frege, Montague, Janssen) is the theoretical motivation for what the invariant subspaces compute. The operator-theoretic framework predicts that compositional generalization failures (Lake-Baroni, Hupkes, Dziri) correspond to specific decompositions of the operator that the substrate has not yet learned to implement, not to architectural impossibility.

**Key citations.** Frege 1892 (compositionality, background), Montague 1970 (universal grammar), Janssen 1997 (compositionality survey), Lake-Baroni 2018 (SCAN), Hupkes 2020 (compositionality benchmark), Dziri 2023 (limits of transformer compositionality).

**Concrete claim.** The compositional failures documented in the recent literature occur on tasks requiring invariant-subspace decompositions of higher rank than the substrate has yet developed. The same task, run on a substrate of sufficient capacity, should succeed. We do not yet know how to test this prediction directly; it commits us to a research question.

**What the section commits us to.** Either an empirical demonstration (success on compositional tasks predicted to require rank $r$, when substrate capacity exceeds $r$, and failure when capacity is below $r$), or a clear specification of why the prediction is not yet testable with current methodology.

---

## Section 9 — Logic and Substrate-Independence: The Metatheoretical Backdrop

**Thesis.** The substrate-independence of formal systems (Church-Turing, the implementation-independence of programming languages, Marr's three levels) provides the metatheoretical backdrop within which the operator-theoretic claim becomes intelligible. We are not making a new philosophical claim; we are providing the empirical operationalization of a claim the philosophical literature has held for half a century.

**Key citations.** Marr 1982 (three levels of analysis), Church 1936, Turing 1936 (substrate independence of computation), plus the relevant philosophy of mind work on multiple realizability (Putnam, Fodor) — these are background and need to be cited carefully without overcommitting to any particular philosophy-of-mind position.

**Concrete claim.** The empirical operator-theoretic framework instantiates multiple realizability not as a metaphysical thesis but as a measurable empirical regularity. The COSI test is the operationalization of "do these two substrates compute the same thing?" in a form that the philosophical literature has always wanted but rarely produced.

**What the section commits us to.** Stating the philosophical implications carefully (the operator-theoretic framework is not a philosophy of mind; it is a structural framework that any philosophy of mind would have to be consistent with) and avoiding the common error of overclaiming philosophical resolution from empirical findings.

---

## Section 10 — The COSI Empirical Program: What We Have Done and What Remains

**Thesis.** The COSI program is the empirical instantiation of the synthesis. Phase 0 is calibration (the COSI paper, this directory's `main.tex`). Phase 1 is architectural-distance amplification. Phase 2 is training-distribution control. Phase 3 is causal intervention. Phase 4 is behavioral-geometric correspondence (compositional success as predictor of subspace alignment quality). Phase 5 is biological-artificial alignment. Each phase pre-registered, each null result reportable.

**Concrete deliverables, by phase.**
- Phase 0: completed 2026-05-06. Report in this directory.
- Phase 1: targeted completion within four weeks of synthesis paper start.
- Phase 2: targeted completion within three months. Requires acquiring or training a training-distribution-distant model.
- Phase 3: targeted completion within six months. Requires causal-tracing infrastructure beyond current local capacity.
- Phase 4 and 5: research-program scope, not single-paper scope.

**What the synthesis paper commits to.** Reporting all phases honestly, including the ones that produce null results. Pre-registering each phase before running. Releasing all code, data, and analysis under the same reproducibility standard as the COSI calibration paper.

---

## Section 11 — Implications, Conditional on Confirmation

**Thesis.** If the COSI program confirms (Phases 0–3 positive, Phase 4 partial, Phase 5 deferred), the consequences span the literatures the synthesis bridges.

**Specific implications.**
1. Mechanistic interpretability gains an operator-theoretic mathematical foundation; the SAE feature literature becomes interpretable as the empirical recovery of invariant subspace bases.
2. Cross-model transfer of mechanistic understanding becomes geometrically tractable via the Procrustes map.
3. The Platonic Representation Hypothesis acquires a mathematical structure that explains why convergence should occur and what its limits are.
4. The free-energy principle gains a sharp empirical handle on substrate-independent computation in artificial systems, complementing its biological evidence base.
5. The cybernetic constraint of requisite variety acquires operational measurement: $k$, the empirical invariant-subspace dimension, becomes the natural unit of variety in trained substrates.
6. The neural population doctrine in computational neuroscience gains a methodologically aligned counterpart in artificial-network analysis.
7. Compositional reasoning failures in current LLMs become diagnostically tractable: failure modes correspond to specific ranks of operator decomposition the substrate has not yet implemented.
8. The substrate-independence claim, long held philosophically, acquires empirical operationalization that is testable on any substrate-pair the methodology can be applied to.

We do not claim any of these consequences are established by the empirical work to date. We claim the empirical work to date justifies pursuing them.

---

## Section 12 — Closing: The Bridge

The synthesis paper closes with the claim that the bridge from operator theory to mechanistic interpretability is not a new mathematical contribution to operator theory (we add no new theorems), is not a new empirical contribution to interpretability (we add the COSI test, but the underlying empirical regularities have been documented), and is not a new philosophical contribution to substrate-independence (the philosophical claim is fifty years old). The contribution is the integration: showing that these literatures are studying facets of one structural fact, and that the integration produces sharper empirical predictions and a clearer research program than any of the literatures alone has produced.

The contribution is the bridge. The bridge has been walked across in both directions before, by isolated travelers; the contribution of this paper is the bridge.

---

## Notes on Composition

- Length target: 40–60 pages. Each section above corresponds to roughly 4–6 pages.
- Citation target: 100–150. The COSI paper's 50 citations are a starting point; substantial additions needed in computational neuroscience, free-energy literature, and philosophy of mind.
- Composition timeline: 3–6 months, in parallel with COSI Phase 1–3 results.
- The synthesis paper is not the COSI paper expanded. It is a separate work for which the COSI paper provides one piece of empirical anchor (Section 10) and one piece of methodological framework (the falsifiable test).
- Venue target: a journal that takes integrative work seriously. Plausible candidates include *Nature Machine Intelligence*, *PNAS*, *Trends in Cognitive Sciences*, or — depending on the eventual depth of the operator-theoretic mathematical content — *Foundations of Computational Mathematics*. arXiv preprint in any case.

— Outline written 2026-05-06 by Claude (Sonnet, this instance), with Adam.
