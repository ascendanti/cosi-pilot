#!/usr/bin/env python
"""Construct the Phase 1 probe set: ≥500 cross-domain compositional propositions.

Per the COSI design note §3.2 and the Phase 0 result note §3, Phase 1 requires
a probe set substantially larger than Phase 0 (96) and crossing at least three
domains so that domain narrowness is not the binding limitation.

The Phase 0 probe set was drawn from a single domain (political-economy /
categorial-architecture, via the frontier-NNN paper title corpus). Phase 1
adds:
  - Mathematical compositional reasoning (templated from MATH/GSM8K structure)
  - Logical/inferential composition (templated from natural-language inference)
  - The original political-economy domain (extended with a generator)

Each domain contributes roughly one-third of the final probe set. The
templates produce propositions of similar token length to the Phase 0 probes
so that pooling biases are comparable across phases.

Output: data/cosi/probe_set_phase1_v1.jsonl
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v1.jsonl"
SEED = 20260506


# ---------------------------------------------------------------------------
# Domain 1: Mathematical compositional reasoning
# ---------------------------------------------------------------------------

MATH_TEMPLATES = [
    "Consider the following claim: \"{a} divided by {b} equals {result}\". Is this a coherent proposition?",
    "Consider the following claim: \"the sum of {a} and {b} is {result}\". Is this a coherent proposition?",
    "Consider the following claim: \"{a} multiplied by {b} produces {result}\". Is this a coherent proposition?",
    "Consider the following claim: \"if x equals {a} and y equals {b}, then x plus y equals {result}\". Is this a coherent proposition?",
    "Consider the following claim: \"the product of {a} and {b} divided by {c} equals {result}\". Is this a coherent proposition?",
    "Consider the following claim: \"a function f(n) equals {a} times n returns {result} when n is {b}\". Is this a coherent proposition?",
    "Consider the following claim: \"the {ordinal} prime number is {prime}\". Is this a coherent proposition?",
    "Consider the following claim: \"a square with side {a} has area {result}\". Is this a coherent proposition?",
    "Consider the following claim: \"a triangle with sides {a}, {b}, and {c} has perimeter {result}\". Is this a coherent proposition?",
    "Consider the following claim: \"the absolute value of {a} minus {b} equals {result}\". Is this a coherent proposition?",
]


def generate_math_probes(n: int, rng: random.Random) -> list[dict]:
    probes = []
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
    for i in range(n):
        tpl = rng.choice(MATH_TEMPLATES)
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        c = rng.randint(2, 99)
        # Compute or scramble result depending on whether the proposition should be true
        # We deliberately mix true and false propositions: COSI tests representational
        # geometry, not factual correctness, and including both keeps the model from
        # being biased toward agreement.
        truth_bias = rng.random() < 0.5
        if "divided by" in tpl and "produces {result}" not in tpl:
            result = a // b if b > 0 else 0
        elif "sum" in tpl or "x plus y" in tpl:
            result = a + b
        elif "multiplied" in tpl or "product" in tpl:
            if "divided by {c}" in tpl:
                result = (a * b) // c if c > 0 else 0
            else:
                result = a * b
        elif "function f(n)" in tpl:
            result = a * b
        elif "prime number" in tpl:
            idx = rng.randint(0, 9)
            tpl = tpl.replace("{ordinal}", ordinals[idx]).replace("{prime}", str(primes[idx]))
            probes.append({
                "id": f"math-{i:04d}",
                "domain": "math",
                "prompt": tpl,
            })
            continue
        elif "square" in tpl:
            result = a * a
        elif "triangle" in tpl:
            result = a + b + c
        elif "absolute value" in tpl:
            result = abs(a - b)
        else:
            result = a + b

        if not truth_bias:
            result = result + rng.choice([1, -1, 7, -7, 3, -3])
            if result < 0:
                result = abs(result) + 10

        prompt = tpl.replace("{a}", str(a)).replace("{b}", str(b)).replace("{c}", str(c)).replace("{result}", str(result))
        probes.append({
            "id": f"math-{i:04d}",
            "domain": "math",
            "prompt": prompt,
        })
    return probes


# ---------------------------------------------------------------------------
# Domain 2: Logical and inferential composition
# ---------------------------------------------------------------------------

LOGIC_TEMPLATES = [
    "Consider the following claim: \"if {p} then {q}; {p}; therefore {q}\". Is this a coherent proposition?",
    "Consider the following claim: \"{p} implies {q}; {q} implies {r}; therefore {p} implies {r}\". Is this a coherent proposition?",
    "Consider the following claim: \"either {p} or {q}; not {p}; therefore {q}\". Is this a coherent proposition?",
    "Consider the following claim: \"all {a} are {b}; {x} is {a}; therefore {x} is {b}\". Is this a coherent proposition?",
    "Consider the following claim: \"some {a} are {b}; therefore some {b} are {a}\". Is this a coherent proposition?",
    "Consider the following claim: \"no {a} are {b}; {x} is {a}; therefore {x} is not {b}\". Is this a coherent proposition?",
    "Consider the following claim: \"{p} and {q} together imply {r}; {r} is false; therefore {p} and {q} cannot both hold\". Is this a coherent proposition?",
    "Consider the following claim: \"the set of {a} is a subset of {b}; {b} is a subset of {c}; therefore {a} is a subset of {c}\". Is this a coherent proposition?",
    "Consider the following claim: \"if {x} is {a} then {x} is {b}; {x} is not {b}; therefore {x} is not {a}\". Is this a coherent proposition?",
    "Consider the following claim: \"{p} is true if and only if {q} is true; {p} is false; therefore {q} is false\". Is this a coherent proposition?",
]

LOGIC_PROPS = [
    "the system is stable", "the input is well-formed", "the output is bounded",
    "the network converges", "the loss decreases", "the gradient vanishes",
    "the policy is optimal", "the constraint binds", "the equilibrium exists",
    "the channel is noisy", "the signal is recoverable", "the protocol terminates",
    "the contract is binding", "the agent is rational", "the market clears",
    "the proof completes", "the program halts", "the function is total",
    "the type checks", "the schema validates",
]

LOGIC_NOUNS = [
    "transformers", "operators", "subspaces", "manifolds", "trajectories",
    "channels", "signals", "categories", "morphisms", "automata",
    "polynomials", "lattices", "graphs", "trees", "sequences",
    "metrics", "norms", "distributions", "estimators", "predictors",
]

LOGIC_PROPERTIES = [
    "compact", "continuous", "linear", "bounded", "convex",
    "smooth", "measurable", "open", "closed", "connected",
    "computable", "decidable", "complete", "consistent", "sound",
    "stable", "robust", "differentiable", "invertible", "compositional",
]


def generate_logic_probes(n: int, rng: random.Random) -> list[dict]:
    probes = []
    for i in range(n):
        tpl = rng.choice(LOGIC_TEMPLATES)
        p = rng.choice(LOGIC_PROPS)
        q = rng.choice(LOGIC_PROPS)
        r = rng.choice(LOGIC_PROPS)
        a = rng.choice(LOGIC_NOUNS)
        b = rng.choice(LOGIC_PROPERTIES)
        c = rng.choice(LOGIC_NOUNS)
        x = "this " + rng.choice(LOGIC_NOUNS).rstrip("s")
        prompt = (
            tpl.replace("{p}", p)
            .replace("{q}", q)
            .replace("{r}", r)
            .replace("{a}", a)
            .replace("{b}", b)
            .replace("{c}", c)
            .replace("{x}", x)
        )
        probes.append({
            "id": f"logic-{i:04d}",
            "domain": "logic",
            "prompt": prompt,
        })
    return probes


# ---------------------------------------------------------------------------
# Domain 3: Political-economy / categorial-architecture (Phase 0 domain extended)
# ---------------------------------------------------------------------------

POLECON_TEMPLATES = [
    "Consider the following claim: \"{x} is constituted by {y} rather than prior to it\". Is this a coherent proposition?",
    "Consider the following claim: \"the legitimacy of {x} depends on the {y} that recognize it\". Is this a coherent proposition?",
    "Consider the following claim: \"changes in {x} produce changes in {y}\". Is this a coherent proposition?",
    "Consider the following claim: \"{x} cannot be coordinated without commensurable {y}\". Is this a coherent proposition?",
    "Consider the following claim: \"the failure of {x} reflects a failure of {y}\". Is this a coherent proposition?",
    "Consider the following claim: \"{x} is the categorial precondition for {y}\". Is this a coherent proposition?",
    "Consider the following claim: \"a regime of {x} produces a regime of {y}\". Is this a coherent proposition?",
    "Consider the following claim: \"the recoding of {x} as {y} alters its political function\". Is this a coherent proposition?",
    "Consider the following claim: \"{x} acts on {y} only through the schemas that link them\". Is this a coherent proposition?",
    "Consider the following claim: \"any policy targeting {x} must address the substrate of {y}\". Is this a coherent proposition?",
]

POLECON_TERMS = [
    "sovereignty", "legitimacy", "authority", "recognition",
    "categorization", "classification", "coordination", "policy",
    "deterrence", "compliance", "enforcement", "negotiation",
    "currency", "debt", "trade", "production",
    "institutions", "norms", "treaties", "sanctions",
    "citizenship", "representation", "constitution", "jurisdiction",
    "discrimination", "redistribution", "taxation", "regulation",
    "violence", "stability", "insurgency", "alliance",
    "decidability", "verifiability", "auditability", "accountability",
    "categorial schemas", "ontological commitments", "epistemic gaps", "structural arrangements",
]


def generate_polecon_probes(n: int, rng: random.Random) -> list[dict]:
    probes = []
    for i in range(n):
        tpl = rng.choice(POLECON_TEMPLATES)
        x = rng.choice(POLECON_TERMS)
        y = rng.choice([t for t in POLECON_TERMS if t != x])
        prompt = tpl.replace("{x}", x).replace("{y}", y)
        probes.append({
            "id": f"polecon-{i:04d}",
            "domain": "polecon",
            "prompt": prompt,
        })
    return probes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(SEED)
    n_per_domain = 200  # 600 total

    math_probes = generate_math_probes(n_per_domain, rng)
    logic_probes = generate_logic_probes(n_per_domain, rng)
    polecon_probes = generate_polecon_probes(n_per_domain, rng)

    all_probes = math_probes + logic_probes + polecon_probes
    rng.shuffle(all_probes)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for p in all_probes:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(all_probes)} probes to {OUT_PATH}")
    print(f"  math: {len(math_probes)}")
    print(f"  logic: {len(logic_probes)}")
    print(f"  polecon: {len(polecon_probes)}")
    print()
    print("Sample (first 3 of each domain after shuffle):")
    by_domain: dict = {}
    for p in all_probes:
        by_domain.setdefault(p["domain"], []).append(p)
    for domain, probes in by_domain.items():
        print(f"\n  {domain}:")
        for p in probes[:3]:
            print(f"    {p['id']}: {p['prompt']}")


if __name__ == "__main__":
    main()
