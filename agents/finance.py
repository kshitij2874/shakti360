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
) -> dict[str, Any]:
    """
    Execute the finance agent.
    Returns: {response, citations, tools_to_call, rag_sources}
    """
    # 1. Retrieve RAG context
    rag_chunks = await retrieve(query=query, pillar="FINANCE", age_band=age_band)
    context = format_context(rag_chunks)

    # 2. Build prompt
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User age band: {age_band}\n"
        f"User memory context: {user_memory_context}\n\n"
        f"{context}\n\n"
        f"User query: {query}\n\n"
        "Respond helpfully with plain-language financial guidance. "
        "Cite sources. Suggest concrete next steps."
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
            f"Here's what I found about your finance question on '{query}' "
            f"for the {age_band} age group:\n\n"
        )
        if rag_chunks:
            for chunk in rag_chunks[:2]:
                response_text += f"• {chunk['content'][:200]}...\n\n"
            response_text += f"Source: {rag_chunks[0].get('source_ref', 'Government financial guidelines')}\n\n"
        response_text += (
            "Remember: past returns don't guarantee future performance. "
            "For personalised advice, consult a SEBI-registered financial advisor."
        )

    # 4. Detect tool calls
    query_lower = query.lower()
    # SIP calculator
    if any(w in query_lower for w in ["sip calculat", "how much will", "returns on sip", "calculate sip"]):
        # Try to extract numbers
        import re
        amounts = re.findall(r'₹?\s*(\d+[,\d]*)', query)
        monthly = float(amounts[0].replace(",", "")) if amounts else 5000.0
        years_match = re.findall(r'(\d+)\s*year', query_lower)
        years = int(years_match[0]) if years_match else 10
        tools_to_call.append({
            "tool": "calculate_sip",
            "params": {"monthly_amount": monthly, "annual_return_pct": 12.0, "years": years},
            "reason": f"Calculate SIP returns for ₹{monthly:,.0f}/month over {years} years.",
        })

    # Scheme lookups
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
        "pillar": "FINANCE",
    }
