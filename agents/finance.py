"""
agents/finance.py — Finance sub-agent.
Model: gemini-2.5-flash
Explains in plain language. Never promises returns. Always cites scheme/source.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import asyncio

from observability import observe
from rag import retrieve, format_context, web_search_fallback
from tools import get_helplines
from dual_user import get_user_profile, build_user_context, build_framing_prefix
from persona import build_persona_prefix
from response_builder import build_full_response

logger = logging.getLogger("shakti.agents.finance")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a women's financial guide for India. "
    "Be direct: lead with the answer, then the numbers. "
    "Use concrete figures — ₹ amounts, percentages, timelines. No vague encouragement. "
    "Cite every scheme, rule, or rate by name and source. Never promise or imply guaranteed returns. "
    "End with one specific, actionable next step."
)


@observe("finance_agent_run")
async def run(
    query: str,
    age_band: str,
    user_memory_context: str = "",
    session_id: str = "",
    user_id: str = "",
    answer_tier: str = "reasoning",
) -> dict[str, Any]:
    """
    Execute the finance agent.
    Returns: {response, citations, tools_to_call, rag_sources}
    """
    # 1. Retrieve RAG context + fetch profile concurrently (one profile read, not two)
    async def _profile():
        return await get_user_profile(user_id) if user_id else {}
    rag_chunks, profile = await asyncio.gather(
        retrieve(query=query, pillar="FINANCE", age_band=age_band),
        _profile(),
    )

    # Web fallback when the local corpus has nothing relevant
    if not rag_chunks:
        rag_chunks = await web_search_fallback(query, "FINANCE")

    # 2. Build dual-user framing + age-tuned persona from the single profile
    framing_prefix = build_framing_prefix(build_user_context(profile))
    persona_prefix = build_persona_prefix(profile)
    combined_persona = (persona_prefix + "\n\n" + SYSTEM_PROMPT).strip()

    # 3. Build structured response with no-truncation guarantees
    structured = await build_full_response(
        pillar="FINANCE",
        age_band=age_band,
        query=query,
        clarifying_qa=[],
        rag_chunks=rag_chunks,
        persona_prefix=combined_persona,
        framing_prefix=framing_prefix,
        memory_context=user_memory_context,
        fallback_system_prompt=SYSTEM_PROMPT,
        model_name=DEFAULT_MODEL,
        language=profile.get("language", "English"),
        answer_tier=answer_tier,
    )

    response_text = structured["answer"]
    citation_chips = structured["citations"]
    next_steps = structured["next_steps"]

    # 4. Detect tool calls (conservative — only on EXPLICIT requests)
    tools_to_call: list[dict[str, Any]] = []
    query_lower = query.lower()
    
    # SIP calculator — only when user explicitly asks to calculate
    if any(phrase in query_lower for phrase in ["calculate sip", "sip calculator", "sip returns", "how much will my sip"]):
        import re
        amounts = re.findall(r'\u20b9?\s*(\d+[,\d]*)', query)
        monthly = float(amounts[0].replace(",", "")) if amounts else 5000.0
        years_match = re.findall(r'(\d+)\s*year', query_lower)
        years = int(years_match[0]) if years_match else 10
        tools_to_call.append({
            "tool": "calculate_sip",
            "params": {"monthly_amount": monthly, "annual_return_pct": 12.0, "years": years},
            "reason": f"Calculate SIP returns for \u20b9{monthly:,.0f}/month over {years} years.",
        })

    # Scheme lookup — only when user explicitly names a scheme
    scheme_keywords = {
        "sukanya samriddhi": "sukanya", "sukanya yojana": "sukanya",
        "senior citizen savings": "scss", "scss scheme": "scss",
        "national pension": "nps", "nps scheme": "nps",
        "mudra loan": "mudra", "mudra yojana": "mudra",
        "stand up india": "standup", "standup india": "standup",
        "atal pension": "apy", "apy scheme": "apy",
    }
    for keyword, scheme_key in scheme_keywords.items():
        if keyword in query_lower:
            tools_to_call.append({
                "tool": "lookup_scheme",
                "params": {"keyword": scheme_key},
                "reason": f"User asked about scheme: {scheme_key}",
            })
            break
    
    # Budget analysis — only when user explicitly asks for it
    if any(phrase in query_lower for phrase in ["analyze my budget", "budget analysis", "where does my money go", "track my expenses", "spending analysis"]):
        tools_to_call.append({
            "tool": "analyze_budget",
            "params": {"monthly_income": 50000, "expenses": {}, "savings_goal": None},
            "reason": "User is explicitly asking for budget analysis.",
        })
    
    # Scheme eligibility — only when user explicitly asks
    if any(phrase in query_lower for phrase in ["which schemes am i eligible", "what schemes do i qualify", "eligible for which schemes", "scheme eligibility"]):
        tools_to_call.append({
            "tool": "check_scheme_eligibility",
            "params": {"user_profile": {}, "scheme_type": "all"},
            "reason": "User is explicitly asking about scheme eligibility.",
        })
    
    # Tavily web search — ONLY when user explicitly asks for latest/current info
    if any(phrase in query_lower for phrase in ["latest interest rate", "current interest rate", "latest market update", "stock market today", "rbi announcement", "sebi circular"]):
        tools_to_call.append({
            "tool": "tavily_search_finance",
            "params": {"query": query, "max_results": 5},
            "reason": "User explicitly asks for up-to-date financial information.",
        })

    # 5. Legacy citation strings (for validators)
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
        "pillar": "FINANCE",
        "diagnostics": structured.get("diagnostics", {}),
    }
