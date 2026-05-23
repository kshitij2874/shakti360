"""
validators.py — Three parallel validation checks for orchestrator.
1. citation_check — factual claims must cite RAG sources
2. age_appropriateness_check — no inappropriate content for age band
3. domain_safety_check — no medical diagnosis, no financial guarantees
"""
from __future__ import annotations

import logging
import re
from typing import Any

from observability import observe

logger = logging.getLogger("shakti.validators")

# ── Age-inappropriate content patterns ──
INAPPROPRIATE_YOUNG: list[str] = [
    r"sexual\s+intercourse",
    r"contraceptive\s+pills",
    r"abortion\s+procedure",
    r"substance\s+abuse\s+methods",
    r"alcohol\s+consumption\s+guide",
    r"suicide\s+methods",
]

SAFETY_VIOLATIONS: list[str] = [
    r"i\s+diagnose\s+you",
    r"you\s+(have|are\s+suffering\s+from)\s+\w+\s+(disease|disorder|syndrome|cancer)",
    r"guaranteed\s+returns",
    r"you\s+will\s+(definitely|certainly)\s+(earn|make|get)\s+\d+%",
    r"buy\s+this\s+stock",
    r"invest\s+all\s+your\s+money",
    r"stop\s+taking\s+your\s+medication",
    r"no\s+need\s+to\s+see\s+a\s+doctor",
]

SAFE_FALLBACK = (
    "I want to make sure you get the most accurate information. "
    "For this topic, I'd recommend consulting official sources or reaching out to:\n"
    "• Women's Helpline: 181\n"
    "• Police: 1091\n"
    "• Emergency: 112\n\n"
    "These services are available 24/7 and can provide verified guidance."
)


@observe("citation_check")
async def citation_check(
    response_text: str,
    rag_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Verify that factual claims in the response are grounded in RAG sources.
    Returns: {passed: bool, citations_found: int, details: str}
    """
    if not rag_sources:
        return {
            "passed": False,
            "citations_found": 0,
            "details": "No RAG sources available to validate against.",
        }

    # Extract source identifiers from RAG chunks
    source_texts = []
    source_names = []
    for src in rag_sources:
        source_texts.append(src.get("content", "").lower())
        source_names.append(src.get("source", "unknown"))

    # Check if response content is grounded in source material
    response_lower = response_text.lower()
    sentences = [s.strip() for s in re.split(r'[.!?]', response_text) if len(s.strip()) > 20]

    grounded_count = 0
    total_claims = max(len(sentences), 1)

    for sentence in sentences:
        sentence_lower = sentence.lower()
        # Check if key terms from the sentence appear in any source
        words = set(re.findall(r'\b\w{4,}\b', sentence_lower))
        for source_text in source_texts:
            overlap = sum(1 for w in words if w in source_text)
            if overlap >= min(3, len(words)):
                grounded_count += 1
                break

    grounding_ratio = grounded_count / total_claims
    passed = grounding_ratio >= 0.5  # At least 50% of claims should be grounded

    return {
        "passed": passed,
        "citations_found": grounded_count,
        "total_claims": total_claims,
        "grounding_ratio": round(grounding_ratio, 2),
        "sources_used": source_names,
        "details": f"{grounded_count}/{total_claims} claims grounded in sources.",
    }


@observe("age_appropriateness_check")
async def age_appropriateness_check(
    response_text: str,
    age_band: str,
) -> dict[str, Any]:
    """
    Check response for age-inappropriate content.
    Stricter for 11-24 age band.
    Returns: {passed: bool, details: str}
    """
    violations = []

    if age_band == "11-24":
        for pattern in INAPPROPRIATE_YOUNG:
            if re.search(pattern, response_text, re.IGNORECASE):
                violations.append(pattern)

    if violations:
        return {
            "passed": False,
            "violations": len(violations),
            "details": f"Content may not be appropriate for age band {age_band}.",
        }

    return {
        "passed": True,
        "violations": 0,
        "details": f"Content is appropriate for age band {age_band}.",
    }


@observe("domain_safety_check")
async def domain_safety_check(
    response_text: str,
    pillar: str,
) -> dict[str, Any]:
    """
    Check for safety violations: no medical diagnosis, no financial guarantees.
    Returns: {passed: bool, details: str}
    """
    violations = []

    for pattern in SAFETY_VIOLATIONS:
        if re.search(pattern, response_text, re.IGNORECASE):
            violations.append(pattern)

    # Additional pillar-specific checks
    if pillar == "HEALTH":
        # Should not prescribe specific medications
        med_patterns = [
            r"take\s+\d+\s*mg\s+of\s+\w+",
            r"prescription:\s+\w+",
        ]
        for p in med_patterns:
            if re.search(p, response_text, re.IGNORECASE):
                violations.append("medication_prescription")

    elif pillar == "FINANCE":
        # Should not guarantee returns
        fin_patterns = [
            r"guaranteed\s+\d+%\s+returns",
            r"risk-free\s+profit",
            r"you\s+cannot\s+lose",
        ]
        for p in fin_patterns:
            if re.search(p, response_text, re.IGNORECASE):
                violations.append("financial_guarantee")

    if violations:
        return {
            "passed": False,
            "violations": len(violations),
            "details": f"Safety concern detected in {pillar} response.",
        }

    return {
        "passed": True,
        "violations": 0,
        "details": f"{pillar} response passes safety checks.",
    }


async def run_all_validations(
    response_text: str,
    rag_sources: list[dict[str, Any]],
    age_band: str,
    pillar: str,
) -> dict[str, Any]:
    """
    Run all 3 validation checks in parallel.
    Returns combined result with pass/fail and safe fallback if needed.
    """
    import asyncio

    citation_result, age_result, safety_result = await asyncio.gather(
        citation_check(response_text, rag_sources),
        age_appropriateness_check(response_text, age_band),
        domain_safety_check(response_text, pillar),
    )

    all_passed = (
        citation_result["passed"]
        and age_result["passed"]
        and safety_result["passed"]
    )

    result = {
        "all_passed": all_passed,
        "citation_check": citation_result,
        "age_appropriateness_check": age_result,
        "domain_safety_check": safety_result,
    }

    if not all_passed:
        result["fallback_response"] = SAFE_FALLBACK
        failures = []
        if not citation_result["passed"]:
            failures.append("citation")
        if not age_result["passed"]:
            failures.append("age_appropriateness")
        if not safety_result["passed"]:
            failures.append("domain_safety")
        result["failed_checks"] = failures
        logger.warning(f"Validation FAILED: {failures}")

    return result
