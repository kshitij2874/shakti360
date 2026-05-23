"""
agents/finance.py — Finance sub-agent.
Model: gemini-2.5-flash
Explains in plain language. Never promises returns. Always cites scheme/source.
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

logger = logging.getLogger("shakti.agents.finance")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a women's financial literacy guide. "
    "Explain in plain language. Never promise returns. "
    "Always cite the scheme or source. Suggest concrete next steps. "
    "Be encouraging and practical. Use Indian Rupee (₹) for amounts. "
    "Tailor advice to the user's age band."
)


@observe("finance_agent_run")
async def run(
    query: str,
    age_band: str,
    user_memory_context: str = "",
    session_id: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    """
    Execute the finance agent.
    Returns: {response, citations, tools_to_call, rag_sources}
    """
    # 1. Retrieve RAG context
    rag_chunks = await retrieve(query=query, pillar="FINANCE", age_band=age_band)

    # 2. Build dual-user framing + age-tuned persona
    user_ctx, framing_prefix = await get_framing_for_user(user_id)
    persona_prefix = await get_persona_prefix(user_id)
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
    )

    response_text = structured["answer"]
    citation_chips = structured["citations"]
    next_steps = structured["next_steps"]

    # 4. Detect tool calls (rule-based)
    tools_to_call: list[dict[str, Any]] = []
    query_lower = query.lower()
    if any(w in query_lower for w in ["sip calculat", "how much will", "returns on sip", "calculate sip"]):
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

    scheme_keywords = {
        "sukanya": "sukanya", "samriddhi": "sukanya",
        "scss": "scss", "senior citizen": "scss",
        "nps": "nps", "pension": "nps",
        "mudra": "mudra", "loan": "mudra",
        "standup": "standup", "stand up": "standup",
        "apy": "apy", "atal pension": "apy",
    }
    for keyword, scheme_key in scheme_keywords.items():
        if keyword in query_lower:
            tools_to_call.append({
                "tool": "lookup_scheme",
                "params": {"keyword": scheme_key},
                "reason": f"User asked about scheme: {scheme_key}",
            })
            break
    
    # Budget analysis tool - when user asks about budget, spending, expenses
    if any(w in query_lower for w in ["budget", "spending", "expenses", "where does my money go", "track expenses", "financial health"]):
        tools_to_call.append({
            "tool": "analyze_budget",
            "params": {"monthly_income": 50000, "expenses": {}, "savings_goal": None},
            "reason": "User is asking for budget analysis and spending insights.",
        })
    
    # Scheme eligibility checker - when user asks what schemes they qualify for
    if any(w in query_lower for w in ["eligible", "qualify", "which schemes", "what schemes", "benefits available"]):
        tools_to_call.append({
            "tool": "check_scheme_eligibility",
            "params": {"user_profile": {}, "scheme_type": "all"},
            "reason": "User is asking about scheme eligibility.",
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
