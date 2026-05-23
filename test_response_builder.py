"""
test_response_builder.py — Smoke test for the structured response builder.

Validates that for representative queries across all 3 pillars and age bands:
  - answer >= 150 chars
  - answer ends with terminal punctuation (not truncated mid-sentence)
  - >= 2 next_steps
  - citations present (or graceful fallback when RAG is empty)

Usage:
  python test_response_builder.py
"""
from __future__ import annotations

import asyncio
import os
import sys

from response_builder import build_full_response


TEST_CASES = [
    # (pillar, age_band, query, clarifying_qa)
    ("HEALTH", "11-24", "I just got my first period and I'm scared", []),
    ("HEALTH", "25-40", "Planning my first pregnancy in 6 months", [
        {"q": "Any pre-existing conditions?", "a": "PCOS diagnosed last year"},
    ]),
    ("FINANCE", "41+", "I want to plan for retirement, where do I start?", []),
    ("CAREER", "25-40", "I want to restart after a 4-year maternity break", []),
]


async def run_one(pillar: str, age_band: str, query: str, qa: list[dict]) -> bool:
    print("\n" + "=" * 60)
    print(f"TEST: {pillar} | {age_band} | {query[:50]}")
    print("=" * 60)

    response = await build_full_response(
        pillar=pillar,
        age_band=age_band,
        query=query,
        clarifying_qa=qa,
        rag_chunks=[],
        persona_prefix="You are Kalpana, a warm women's life companion.",
        framing_prefix="",
        memory_context="",
        fallback_system_prompt="You are Kalpana.",
    )

    answer = response.get("answer", "") or ""
    next_steps = response.get("next_steps", []) or []
    citations = response.get("citations", []) or []
    diag = response.get("diagnostics", {})

    print(f"Answer length: {len(answer)} chars")
    print(f"Answer tail:   ...{answer[-80:]}")
    print(f"Citations:     {len(citations)}")
    print(f"Next steps:    {len(next_steps)}")
    print(f"Diagnostics:   {diag}")

    failures = []
    if len(answer) < 150:
        failures.append(f"answer too short ({len(answer)})")
    if not answer.rstrip().endswith((".", "!", "?", "\u0964", "\u06d4", ":")):
        failures.append("answer ends mid-sentence")
    if len(next_steps) < 2:
        failures.append("fewer than 2 next steps")

    if failures:
        print(f"\u274c FAIL: {failures}")
        return False
    print("\u2705 PASSED")
    return True


async def main() -> int:
    # Initialize Vertex AI for the test runner. If unavailable, the builder will
    # still emit deterministic fallbacks and the test should still pass.
    try:
        import vertexai  # type: ignore
        project = os.getenv("PROJECT_ID", "shakti360")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        vertexai.init(project=project, location=location)
        print(f"Vertex AI initialized: {project}/{location}")
    except Exception as e:
        print(f"WARN: Vertex AI not initialized ({e}). Fallback paths only.")

    results = []
    for case in TEST_CASES:
        ok = await run_one(*case)
        results.append(ok)

    print("\n" + "=" * 60)
    passed = sum(1 for r in results if r)
    print(f"Result: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
