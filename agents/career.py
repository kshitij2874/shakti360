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
    context = format_context(rag_chunks)

    # 2. Build dual-user framing
    user_ctx, framing_prefix = await get_framing_for_user(user_id)

    # 3. Build prompt
    prompt = (
        f"{framing_prefix}"
        f"{SYSTEM_PROMPT}\n\n"
        f"User age band: {age_band}\n"
        f"User memory context: {user_memory_context}\n\n"
        f"{context}\n\n"
        f"User query: {query}\n\n"
        "Respond with practical career guidance. Cite every scheme, "
        "scholarship, or program. Suggest concrete next steps."
    )

    # 3. Call LLM
    response_text = ""
    tools_to_call: list[dict[str, Any]] = []

    try:
        from vertexai.generative_models import GenerativeModel  # type: ignore
        model = GenerativeModel(DEFAULT_MODEL)
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 800, "temperature": 0.3},
        )
        response_text = response.text.strip()
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        response_text = (
            f"Here's career guidance for your question on '{query}' "
            f"for the {age_band} age group:\n\n"
        )
        if rag_chunks:
            for chunk in rag_chunks[:2]:
                response_text += f"• {chunk['content'][:200]}...\n\n"
            response_text += f"Source: {rag_chunks[0].get('source_ref', 'Government career resources')}\n\n"
        response_text += "For more guidance, visit scholarships.gov.in or the National Career Service portal (ncs.gov.in)."

    # 4. Detect tool calls
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

    # 5. Extract citations
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
        "model_used": DEFAULT_MODEL,
        "pillar": "CAREER",
    }
