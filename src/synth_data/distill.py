"""Distill many generated scenarios down into a curated few-shot library.

This is the step that makes synthetic generation useful rather than just noisy volume: not every
generated scenario is a good example. We score each on clarity and realism (LLM-judge), keep only
the strongest few per specialty/intent, and write them as a compact few-shot file the agent's system
prompt can optionally include.
"""

import json
import os
from collections import defaultdict

import anthropic
import yaml

SCORING_PROMPT = """Rate this synthetic patient-call scenario on a 1-5 scale for how realistic and
well-formed it is as a test case (clear patient turns, sensible required_actions/forbidden_phrases,
correct must_escalate flag). Respond with ONLY a JSON object: {"score": <int>, "reason": "<one sentence>"}
"""


def score_scenario(scenario: dict, model: str = "claude-sonnet-4-6") -> dict:
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=model,
        max_tokens=150,
        system=SCORING_PROMPT,
        messages=[{"role": "user", "content": json.dumps(scenario)}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "reason": f"unparseable: {raw[:150]}"}


def distill(scenarios: list[dict], top_n_per_category: int = 2) -> dict:
    """Score all scenarios, keep the top N per category, group into a few-shot library."""
    scored = []
    for s in scenarios:
        result = score_scenario(s)
        scored.append((s, result["score"], result.get("reason", "")))

    by_category = defaultdict(list)
    for scenario, score, reason in scored:
        by_category[scenario.get("category", "unknown")].append((scenario, score, reason))

    library = {}
    for category, items in by_category.items():
        items.sort(key=lambda x: x[1], reverse=True)
        kept = items[:top_n_per_category]
        library[category] = [
            {"scenario": s, "quality_score": score, "reason": reason}
            for s, score, reason in kept
        ]

    return library


def write_library(library: dict, path: str = "data/few_shot_library.yaml"):
    with open(path, "w") as f:
        yaml.safe_dump(library, f, sort_keys=False)
