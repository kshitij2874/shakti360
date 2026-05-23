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
from persona import get_persona_prefix
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

    # 2. Build dual-user framing + age-tuned persona
    user_ctx, framing_prefix = await get_framing_for_user(user_id)
    persona_prefix = await get_persona_prefix(user_id)
    combined_persona = (persona_prefix + "\n\n" + SYSTEM_PROMPT).strip()

    # 3. Build structured response with no-truncation guarantees
    structured = await build_full_response(
        pillar="CAREER",
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
    
    # Job search tool - when user asks about finding jobs, job listings, opportunities
    if any(w in query_lower for w in ["find jobs", "job listings", "job openings", "job search", "looking for work", "employment opportunities", "hiring", "vacancies"]):
        # Extract job type from query
        job_type = "software"  # default
        if any(w in query_lower for w in ["data", "analytics", "machine learning", "ai"]):
            job_type = "data"
        elif any(w in query_lower for w in ["marketing", "digital", "content", "social media"]):
            job_type = "marketing"
        elif any(w in query_lower for w in ["finance", "accounting", "banking"]):
            job_type = "finance"
        elif any(w in query_lower for w in ["health", "medical", "clinical"]):
            job_type = "healthcare"
        
        tools_to_call.append({
            "tool": "search_jobs_web",
            "params": {"query": job_type, "location": "India", "num_results": 5},
            "reason": "User is looking for job opportunities.",
        })
    
    # Scheme eligibility checker - when user asks what programs they qualify for
    if any(w in query_lower for w in ["eligible", "qualify", "which programs", "what programs", "benefits available"]):
        tools_to_call.append({
            "tool": "check_scheme_eligibility",
            "params": {"user_profile": {}, "scheme_type": "career"},
            "reason": "User is asking about career program eligibility.",
        })
    
    # Tavily web search - when query needs up-to-date career/job market information
    # Triggers on: latest trends, current market, recent opportunities, news, updates
    if any(w in query_lower for w in ["latest", "recent", "current", "trend", "market", "demand", "salary", "hiring", "layoff", "remote", "hybrid", "news", "update", "industry"]):
        tools_to_call.append({
            "tool": "tavily_search_career",
            "params": {"query": query, "max_results": 5},
            "reason": "Query needs up-to-date career/job market information that may not be in RAG.",
        })

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
