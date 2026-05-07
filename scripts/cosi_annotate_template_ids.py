#!/usr/bin/env python
"""Annotate each Phase 1 probe with the template_id it was generated from.

Reverse-matches each prompt against the deterministic templates in
build_phase1_probes.py. The match is exact: each prompt was created by
substituting variables into one of the 30 templates (10 math + 10 logic +
10 polecon), so we can recover the template by substring matching.

Output: data/cosi/probe_set_phase1_v2.jsonl with added 'template_id'
field per probe (e.g., 'math-04', 'logic-07', 'polecon-02').
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Import templates from the build script
import build_phase1_probes as bp  # noqa: E402

PROBE_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v1.jsonl"
OUT_PATH = REPO_ROOT / "data" / "cosi" / "probe_set_phase1_v2.jsonl"


def template_to_regex(template: str) -> str:
    """Convert a Python format-string template to a regex that matches any
    substitution of its placeholders.
    """
    pattern = re.escape(template)
    # Replace escaped placeholders with greedy match (prompts are short)
    pattern = re.sub(r"\\\{[a-z_]+\\\}", r".*?", pattern)
    return "^" + pattern + "$"


def main() -> None:
    domain_templates = {
        "math": [(f"math-{i:02d}", t) for i, t in enumerate(bp.MATH_TEMPLATES)],
        "logic": [(f"logic-{i:02d}", t) for i, t in enumerate(bp.LOGIC_TEMPLATES)],
        "polecon": [(f"polecon-{i:02d}", t) for i, t in enumerate(bp.POLECON_TEMPLATES)],
    }
    # Pre-compile regexes
    domain_regexes: dict[str, list[tuple[str, re.Pattern]]] = {}
    for domain, items in domain_templates.items():
        domain_regexes[domain] = [(tid, re.compile(template_to_regex(t))) for tid, t in items]

    probes = []
    with open(PROBE_PATH) as f:
        for line in f:
            probes.append(json.loads(line))

    annotated = []
    matched = 0
    unmatched = 0
    for p in probes:
        domain = p["domain"]
        prompt = p["prompt"]
        match = None
        for tid, regex in domain_regexes[domain]:
            if regex.match(prompt):
                match = tid
                break
        if match is None:
            unmatched += 1
            print(f"UNMATCHED ({domain}): {prompt[:100]}")
            match = f"{domain}-unknown"
        else:
            matched += 1
        annotated.append({**p, "template_id": match})

    print(f"\nMatched: {matched}, Unmatched: {unmatched}, Total: {len(probes)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for p in annotated:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {OUT_PATH}")

    # Summary
    from collections import Counter
    tid_counts = Counter(p["template_id"] for p in annotated)
    print("\nTemplate distribution:")
    for tid in sorted(tid_counts):
        print(f"  {tid}: {tid_counts[tid]}")


if __name__ == "__main__":
    main()
