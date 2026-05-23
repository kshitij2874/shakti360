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
    context = format_context(rag_chunks)

    # 2. Build dual-user framing
    user_ctx, framing_prefix = await get_framing_for_user(user_id)

    # 3. Build prompt
    helplines = get_helplines()
    prompt = (
        f"{framing_prefix}"
        f"{SYSTEM_PROMPT}\n\n"
        f"User age band: {age_band}\n"
        f"User memory context: {user_memory_context}\n\n"
        f"{context}\n\n"
        f"Emergency helplines: Women's Helpline {helplines['women_helpline']}, "
        f"Ambulance {helplines['ambulance']}, Emergency {helplines['emergency']}\n\n"
        f"User query: {query}\n\n"
        "Respond helpfully, cite sources, and suggest next steps."
    )

    # 3. Call LLM
    model_name = _get_model_name()
    response_text = ""
    tools_to_call: list[dict[str, Any]] = []

    try:
        from vertexai.generative_models import GenerativeModel  # type: ignore
        model = GenerativeModel(model_name if model_name != "medgemma" else DEFAULT_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 800, "temperature": 0.3},
        )
        response_text = response.text.strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        response_text = (
            f"I'd like to help with your health question about '{query}'. "
            f"Based on available information for women in the {age_band} age group:\n\n"
        )
        # Build response from RAG context
        if rag_chunks:
            for chunk in rag_chunks[:2]:
                response_text += f"• {chunk['content'][:200]}...\n\n"
            response_text += f"Source: {rag_chunks[0].get('source_ref', 'Government health guidelines')}\n\n"
        response_text += (
            "For detailed medical guidance, please consult a healthcare professional.\n"
            f"Women's Helpline: {helplines['women_helpline']} | Ambulance: {helplines['ambulance']}"
        )

    # 4. Detect if tool calls are needed
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
    # Check for scheme lookups
    scheme_keywords = ["jsy", "janani", "pmjay", "ayushman", "pmmvy", "maternity scheme"]
    for kw in scheme_keywords:
        if kw in query_lower:
            tools_to_call.append({
                "tool": "lookup_scheme",
                "params": {"keyword": kw},
                "reason": f"User asked about government scheme: {kw}",
            })
            break

    # 5. Extract citations from RAG sources
    citations = []
    for chunk in rag_chunks:
        source_ref = chunk.get("source_ref", "")
        if source_ref and source_ref not in citations:
            citations.append(source_ref)

    return {
        "response": response_text,
        "citations": citations,
        "tools_to_call": tools_to_call,
        "rag_sources": rag_chunks,
        "model_used": model_name,
        "pillar": "HEALTH",
    }
