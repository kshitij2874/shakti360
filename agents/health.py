"""
agents/health.py — Health sub-agent.
Model: MedGemma (if available) or gemini-2.5-flash fallback.
Never diagnoses — navigates, explains, suggests clinics, flags when to see a doctor.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from observability import observe
from rag import retrieve, format_context
from tools import get_helplines
from dual_user import get_framing_for_user
from persona import get_persona_prefix
from response_builder import build_full_response

logger = logging.getLogger("shakti.agents.health")

SYSTEM_PROMPT = (
    "You are a women's health navigator. You DO NOT diagnose. "
    "You navigate — explaining concepts, suggesting clinics, "
    "flagging when to see a doctor. Cite every fact from provided context. "
    "If no context, say 'I need verified info on this — please consult "
    "[appropriate helpline/source]'. "
    "Be warm, empathetic, and culturally sensitive for Indian women. "
    "Use simple language. Always mention relevant helplines when safety is a concern."
)


DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _get_model_name() -> str:
    """Return model to use — MedGemma if enabled, else GEMINI_MODEL from env."""
    if os.getenv("USE_MEDGEMMA", "false").lower() == "true":
        return "medgemma"
    return DEFAULT_MODEL


@observe("health_agent_run")
async def run(
    query: str,
    age_band: str,
    user_memory_context: str = "",
    session_id: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    """
    Execute the health agent.
    Returns: {response, citations, tools_to_call, rag_sources}
    """
    # 1. Retrieve RAG context filtered by pillar=HEALTH + age_band
    rag_chunks = await retrieve(query=query, pillar="HEALTH", age_band=age_band)

    # 2. Build dual-user framing + age-tuned persona
    user_ctx, framing_prefix = await get_framing_for_user(user_id)
    persona_prefix = await get_persona_prefix(user_id)
    combined_persona = (persona_prefix + "\n\n" + SYSTEM_PROMPT).strip()

    # 3. Build structured response with no-truncation guarantees
    model_name = _get_model_name()
    structured = await build_full_response(
        pillar="HEALTH",
        age_band=age_band,
        query=query,
        clarifying_qa=[],  # already inlined into query by orchestrator
        rag_chunks=rag_chunks,
        persona_prefix=combined_persona,
        framing_prefix=framing_prefix,
        memory_context=user_memory_context,
        fallback_system_prompt=SYSTEM_PROMPT,
        model_name=(model_name if model_name != "medgemma" else DEFAULT_MODEL),
    )

    response_text = structured["answer"]
    citation_chips = structured["citations"]
    next_steps = structured["next_steps"]

    # 4. Detect if tool calls are needed (cheap rule-based)
    tools_to_call: list[dict[str, Any]] = []
    query_lower = query.lower()
    if any(w in query_lower for w in ["clinic", "hospital", "doctor near", "nearby"]):
        tools_to_call.append({
            "tool": "find_nearby_clinic",
            "params": {},
            "reason": "User is looking for nearby healthcare facilities.",
        })
    if any(w in query_lower for w in ["police", "safety", "violence", "harassment"]):
        tools_to_call.append({
            "tool": "find_nearby_police",
            "params": {},
            "reason": "User may need safety assistance.",
        })
    scheme_keywords = ["jsy", "janani", "pmjay", "ayushman", "pmmvy", "maternity scheme"]
    for kw in scheme_keywords:
        if kw in query_lower:
            tools_to_call.append({
                "tool": "lookup_scheme",
                "params": {"keyword": kw},
                "reason": f"User asked about government scheme: {kw}",
            })
            break
    
    # Health assessment tool - when user describes symptoms or asks for health check
    if any(w in query_lower for w in ["symptom", "health check", "risk assessment", "health assessment", "am i at risk", "health risk"]):
        tools_to_call.append({
            "tool": "assess_health_risk",
            "params": {"age": 25, "gender": "female", "symptoms": [], "conditions": [], "lifestyle": {}},
            "reason": "User is asking for a health risk assessment.",
        })
    
    # Mental wellbeing tool - when user expresses stress, anxiety, mood concerns
    if any(w in query_lower for w in ["stress", "anxious", "depressed", "mood", "mental health", "wellbeing", "overwhelmed", "burnout"]):
        tools_to_call.append({
            "tool": "mental_wellbeing_check",
            "params": {"mood": "neutral", "stress_level": "moderate", "sleep_quality": "fair", "social_support": "moderate", "recent_changes": []},
            "reason": "User is expressing mental health concerns.",
        })
    
    # Tavily web search - when query needs up-to-date health information
    # Triggers on: latest research, new treatments, recent findings, current statistics
    if any(w in query_lower for w in ["latest", "recent", "new treatment", "research", "study", "statistics", "current", "update", "news", "vaccine", "outbreak", "pandemic"]):
        tools_to_call.append({
            "tool": "tavily_search_health",
            "params": {"query": query, "max_results": 5},
            "reason": "Query needs up-to-date health information that may not be in RAG.",
        })

    # 5. Extract legacy citation strings (for validators) from RAG sources
    citations = []
    for chunk in rag_chunks:
        source_ref = chunk.get("source_ref", "")
        if source_ref and source_ref not in citations:
            citations.append(source_ref)

    return {
        "response": response_text,
        "citations": citations,
        "citation_chips": citation_chips,
        "next_steps": next_steps,
        "tools_to_call": tools_to_call,
        "rag_sources": rag_chunks,
        "model_used": structured.get("model_used", model_name),
        "pillar": "HEALTH",
        "diagnostics": structured.get("diagnostics", {}),
    }
