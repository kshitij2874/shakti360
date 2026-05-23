"""
orchestrator.py — Root ADK agent with exactly 3 responsibilities:
1. INTENT CLASSIFICATION → HEALTH | FINANCE | CAREER
2. ROUTING → dispatch to the correct sub-agent
3. VALIDATION → 3 parallel checks before returning response
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from observability import observe, TraceContext, metrics
from validators import run_all_validations, SAFE_FALLBACK
from memory import get_user_memory_context, ShortTermMemory
from cache import get_cached_response, set_cached_response
import agents.health as health_agent
import agents.finance as finance_agent
import agents.career as career_agent

logger = logging.getLogger("shakti.orchestrator")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

CLASSIFIER_PROMPT = (
    "You are an intent classifier for a women's life companion. "
    "Given a user query, return EXACTLY one word: HEALTH, FINANCE, or CAREER. "
    "No other output."
)

SUB_AGENTS = {
    "HEALTH": health_agent,
    "FINANCE": finance_agent,
    "CAREER": career_agent,
}


# ═══════════════════════════════════════════════════════
# 1. INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════

@observe("orchestrator_classify")
async def classify_intent(query: str) -> str:
    """
    Classify user query into HEALTH | FINANCE | CAREER.
    Uses gemini-2.5-flash with max_output_tokens=5.
    """
    try:
        from vertexai.generative_models import GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold  # type: ignore
        model = GenerativeModel(DEFAULT_MODEL)
        
        # Configure safety settings to avoid blocking benign queries
        safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        ]
        
        response = model.generate_content(
            f"{CLASSIFIER_PROMPT}\n\nUser query: {query}",
            generation_config={"max_output_tokens": 5, "temperature": 0.0},
            safety_settings=safety_settings,
        )
        intent = response.text.strip().upper()
        if intent in ("HEALTH", "FINANCE", "CAREER"):
            return intent
        logger.warning(f"Unexpected classifier output: '{intent}' — falling back to keyword matching")
    except Exception as e:
        logger.warning(f"LLM classifier failed: {e} — using keyword fallback")

    # Keyword-based fallback
    q = query.lower()
    health_kw = ["health", "period", "pregnant", "doctor", "clinic", "menstrual", "menopause",
                 "medicine", "hospital", "symptoms", "pain", "disease", "diet", "nutrition",
                 "puberty", "antenatal", "maternity", "jsy", "pmjay", "breastfeed"]
    finance_kw = ["money", "invest", "sip", "loan", "tax", "savings", "scheme", "pension",
                  "sukanya", "elss", "mutual fund", "insurance", "bank", "emi", "budget",
                  "scss", "nps", "interest rate", "return"]
    career_kw = ["job", "career", "scholarship", "course", "study", "college", "intern",
                 "resume", "interview", "skill", "returnship", "mentor", "business",
                 "entrepreneur", "startup", "freelance", "training", "work", "back to work", "working", "employment",
                 "consulting", "consultant", "retirement", "retire"]

    scores = {
        "HEALTH": sum(1 for kw in health_kw if kw in q),
        "FINANCE": sum(1 for kw in finance_kw if kw in q),
        "CAREER": sum(1 for kw in career_kw if kw in q),
    }

    intent = max(scores, key=scores.get)  # type: ignore
    if scores[intent] == 0:
        intent = "HEALTH"  # Default
    return intent


# ═══════════════════════════════════════════════════════
# 2. ROUTING
# ═══════════════════════════════════════════════════════

@observe("orchestrator_route")
async def route_to_agent(
    query: str,
    intent: str,
    age_band: str,
    user_memory_context: str,
    session_id: str,
) -> dict[str, Any]:
    """Route to the corresponding sub-agent."""
    agent = SUB_AGENTS.get(intent)
    if not agent:
        logger.error(f"Unknown intent: {intent}")
        return {"response": SAFE_FALLBACK, "citations": [], "tools_to_call": [], "rag_sources": []}

    result = await agent.run(
        query=query,
        age_band=age_band,
        user_memory_context=user_memory_context,
        session_id=session_id,
    )
    return result


# ═══════════════════════════════════════════════════════
# 3. VALIDATION
# ═══════════════════════════════════════════════════════

@observe("orchestrator_validate")
async def validate_response(
    response_text: str,
    rag_sources: list[dict[str, Any]],
    age_band: str,
    pillar: str,
) -> dict[str, Any]:
    """Run 3 parallel validation checks on the sub-agent response."""
    return await run_all_validations(
        response_text=response_text,
        rag_sources=rag_sources,
        age_band=age_band,
        pillar=pillar,
    )


# ═══════════════════════════════════════════════════════
# ORCHESTRATOR MAIN FLOW
# ═══════════════════════════════════════════════════════

async def process_query(
    user_id: str,
    age_band: str,
    query: str,
    session_id: str,
) -> dict[str, Any]:
    """
    Main orchestrator flow:
    1. Check cache
    2. Classify intent
    3. Get memory context
    4. Route to sub-agent
    5. Validate response
    6. Cache result
    7. Return structured response
    """
    start_time = time.time()

    # Initialize trace
    trace = TraceContext(
        session_id=session_id,
        user_id=user_id,
        metadata={"age_band": age_band},
    )
    trace.start_trace("shakti_orchestrator")

    steps: list[dict[str, str]] = []

    # Step 1: Classify intent
    steps.append({"step": "classifying", "detail": "Classifying intent..."})
    intent = await classify_intent(query)
    steps.append({"step": "classified", "detail": f"Intent: {intent}"})

    # Check cache
    cached = await get_cached_response(query, age_band, intent)
    if cached:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_session(latency_ms, intent, age_band, cost=0.0)
        return {**cached, "from_cache": True, "latency_ms": round(latency_ms, 1)}

    # Step 2: Get memory context
    steps.append({"step": "memory", "detail": "Loading user context..."})
    memory_context = await get_user_memory_context(user_id, query)

    # Step 3: Route to sub-agent
    steps.append({"step": "routing", "detail": f"Retrieving {intent.lower()} docs for {age_band}..."})
    agent_result = await route_to_agent(
        query=query,
        intent=intent,
        age_band=age_band,
        user_memory_context=memory_context,
        session_id=session_id,
    )

    response_text = agent_result.get("response", "")
    citations = agent_result.get("citations", [])
    tools_to_call = agent_result.get("tools_to_call", [])
    rag_sources = agent_result.get("rag_sources", [])

    # Step 4: Validate response
    steps.append({"step": "validating", "detail": "Validating response..."})
    validation = await validate_response(
        response_text=response_text,
        rag_sources=rag_sources,
        age_band=age_band,
        pillar=intent,
    )

    # Check validation and construct warnings
    validation_warnings = []
    if not validation["all_passed"]:
        failed_checks = validation.get("failed_checks", [])
        for chk in failed_checks:
            detail = ""
            if chk == "citation":
                detail = validation["citation_check"].get("details", "")
            elif chk == "age_appropriateness":
                detail = validation["age_appropriateness_check"].get("details", "")
            elif chk == "domain_safety":
                detail = validation["domain_safety_check"].get("details", "")
            validation_warnings.append(f"{chk}: {detail}")
            logger.warning(f"Validation WARNING - check '{chk}' failed: {detail}")

    # Determine mode: STRICT_VALIDATION flag
    strict_val_env = os.getenv("STRICT_VALIDATION", "false").lower()
    is_strict = strict_val_env in ("true", "1", "yes")

    hallucination_passed = validation["all_passed"]
    if not hallucination_passed:
        if is_strict:
            response_text = validation.get("fallback_response", SAFE_FALLBACK)
            logger.warning(f"Strict validation FAILED, using fallback: {validation.get('failed_checks', [])}")
        else:
            logger.info(f"Validation FAILED in soft mode (STRICT_VALIDATION=false). Warnings: {validation_warnings}")

    latency_ms = (time.time() - start_time) * 1000

    # Build final response
    result = {
        "response": response_text,
        "sub_agent": intent,
        "pillar": intent,
        "age_band": age_band,
        "citations": citations,
        "tools_to_call": tools_to_call,
        "awaiting_approval": len(tools_to_call) > 0,
        "session_id": session_id,
        "validation": {
            "all_passed": validation["all_passed"],
            "citation_check": validation["citation_check"]["passed"],
            "age_check": validation["age_appropriateness_check"]["passed"],
            "safety_check": validation["domain_safety_check"]["passed"],
        },
        "validation_warnings": validation_warnings,
        "latency_ms": round(latency_ms, 1),
        "from_cache": False,
        "steps": steps,
    }

    # Record metrics
    metrics.record_session(
        latency_ms=latency_ms,
        pillar=intent,
        age_band=age_band,
        hallucination_passed=hallucination_passed,
    )

    # Cache the result (without steps/latency for consistency)
    cache_data = {k: v for k, v in result.items() if k not in ("steps", "latency_ms", "from_cache")}
    await set_cached_response(query, age_band, intent, cache_data)

    # Log trace
    trace.add_generation(
        name=f"{intent.lower()}_response",
        model=agent_result.get("model_used", DEFAULT_MODEL),
        input_text=query,
        output_text=response_text,
        metadata={
            "age_band": age_band,
            "pillar": intent,
            "citations_count": len(citations),
            "hallucination_check_passed": hallucination_passed,
            "latency_ms": round(latency_ms, 1),
        },
    )
    trace_url = trace.finalize()
    if trace_url:
        result["trace_url"] = trace_url

    return result
