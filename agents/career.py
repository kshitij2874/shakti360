"""
agents/career.py — Career sub-agent.
Model: gemini-2.5-flash
Practical, India-specific guidance. Cites sources for schemes/scholarships.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from observability import observe
from rag import retrieve, format_context
from tools import get_helplines
from dual_user import get_framing_for_user
from response_builder import build_full_response

logger = logging.getLogger("shakti.agents.career")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a women's career advisor. Give practical, India-specific guidance. "
    "Cite sources for any scheme, scholarship, or program mentioned. "
    "Be motivating and action-oriented. Suggest concrete next steps with "
    "links or contact points where available."
)


@observe("career_agent_run")
async def run(
    query: str,
    age_band: str,
    user_memory_context: str = "",
    session_id: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    """
    Execute the career agent.
    Returns: {response, citations, tools_to_call, rag_sources}
    """
    # 1. Retrieve RAG context
    rag_chunks = await retrieve(query=query, pillar="CAREER", age_band=age_band)

    # 2. Build dual-user framing
    user_ctx, framing_prefix = await get_framing_for_user(user_id)

    # 3. Build structured response with no-truncation guarantees
    structured = await build_full_response(
        pillar="CAREER",
        age_band=age_band,
        query=query,
        clarifying_qa=[],
        rag_chunks=rag_chunks,
        persona_prefix=SYSTEM_PROMPT,
        framing_prefix=framing_prefix,
        memory_context=user_memory_context,
        fallback_system_prompt=SYSTEM_PROMPT,
        model_name=DEFAULT_MODEL,
    )

    response_text = structured["answer"]
    citation_chips = structured["citations"]
    next_steps = structured["next_steps"]

    # 4. Detect tool calls (rule-based)
    tools_to_call: list[dict[str, Any]] = []
    query_lower = query.lower()
    scheme_keywords = {
        "pragati": "pragati", "scholarship": "pragati",
        "pmkvy": "pmkvy", "skill": "pmkvy",
        "mudra": "mudra", "business loan": "mudra",
        "standup": "standup", "startup": "standup",
    }
    for keyword, scheme_key in scheme_keywords.items():
        if keyword in query_lower:
            tools_to_call.append({
                "tool": "lookup_scheme",
                "params": {"keyword": scheme_key},
                "reason": f"User asked about scheme/program: {scheme_key}",
            })
            break

    # 5. Legacy citation strings
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
        "model_used": structured.get("model_used", DEFAULT_MODEL),
        "pillar": "CAREER",
        "diagnostics": structured.get("diagnostics", {}),
    }
