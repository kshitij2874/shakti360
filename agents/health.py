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
    "You are a women's health navigator for India. You do NOT diagnose. "
    "Be direct: lead with the key fact, then explain. "
    "Acknowledge her specific situation briefly — one sentence max — then get to what she needs to know. "
    "No generic reassurances. No fluff. Cite every factual claim from the provided context. "
    "If context doesn't cover something, say so clearly and point to a specific helpline or resource. "
    "Mention helplines (181, 112) only when safety is genuinely at risk — not as a default footer."
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

    # 4. Detect if tool calls are needed (conservative — only on EXPLICIT requests)
    tools_to_call: list[dict[str, Any]] = []
    query_lower = query.lower()
    
    # Only add tools when user EXPLICITLY asks for them
    if any(phrase in query_lower for phrase in ["find clinic", "find hospital", "clinic near", "hospital near", "doctor near me", "nearby clinic", "nearby hospital"]):
        tools_to_call.append({
            "tool": "find_nearby_clinic",
            "params": {},
            "reason": "User is explicitly looking for nearby healthcare facilities.",
        })
    if any(phrase in query_lower for phrase in ["find police", "police near", "police station near", "need help", "emergency help"]):
        tools_to_call.append({
            "tool": "find_nearby_police",
            "params": {},
            "reason": "User explicitly needs safety assistance.",
        })
    scheme_keywords = ["jsy", "janani suraksha", "pmjay", "ayushman bharat", "pmmvy", "matru vandana"]
    for kw in scheme_keywords:
        if kw in query_lower:
            tools_to_call.append({
                "tool": "lookup_scheme",
                "params": {"keyword": kw.split()[0]},
                "reason": f"User asked about government scheme: {kw}",
            })
            break
    
    # Health assessment — only when user explicitly asks for it
    if any(phrase in query_lower for phrase in ["health risk assessment", "assess my health", "health check up", "am i at risk"]):
        tools_to_call.append({
            "tool": "assess_health_risk",
            "params": {"age": 25, "gender": "female", "symptoms": [], "conditions": [], "lifestyle": {}},
            "reason": "User is explicitly asking for a health risk assessment.",
        })
    
    # Mental wellbeing — only when user explicitly expresses these concerns
    if any(phrase in query_lower for phrase in ["i feel stressed", "i am anxious", "i am depressed", "mental health help", "i feel overwhelmed", "burnout"]):
        tools_to_call.append({
            "tool": "mental_wellbeing_check",
            "params": {"mood": "neutral", "stress_level": "moderate", "sleep_quality": "fair", "social_support": "moderate", "recent_changes": []},
            "reason": "User is explicitly expressing mental health concerns.",
        })
    
    # Tavily web search — ONLY when user explicitly asks for latest/current info
    if any(phrase in query_lower for phrase in ["latest research on", "latest studies on", "current outbreak", "current pandemic", "new vaccine for", "recent medical breakthrough"]):
        tools_to_call.append({
            "tool": "tavily_search_health",
            "params": {"query": query, "max_results": 5},
            "reason": "User explicitly asks for up-to-date health research.",
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
