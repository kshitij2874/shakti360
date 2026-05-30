"""
agents/career.py — Career sub-agent.
Model: gemini-2.5-flash
Practical, India-specific guidance. Cites sources for schemes/scholarships.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import asyncio

from observability import observe
from rag import retrieve, format_context
from tools import get_helplines
from dual_user import get_user_profile, build_user_context, build_framing_prefix
from persona import build_persona_prefix
from response_builder import build_full_response

logger = logging.getLogger("shakti.agents.career")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a women's career advisor for India. "
    "Be direct and practical: lead with the most relevant option or action. "
    "Name specific schemes, scholarships, or programs — not generic advice. "
    "Cite sources. No generic motivation. End with a concrete next step "
    "that includes a contact point, website, or specific action."
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
    # 1. Retrieve RAG context + fetch profile concurrently (one profile read, not two)
    async def _profile():
        return await get_user_profile(user_id) if user_id else {}
    rag_chunks, profile = await asyncio.gather(
        retrieve(query=query, pillar="CAREER", age_band=age_band),
        _profile(),
    )

    # 2. Build dual-user framing + age-tuned persona from the single profile
    framing_prefix = build_framing_prefix(build_user_context(profile))
    persona_prefix = build_persona_prefix(profile)
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
        language=profile.get("language", "English"),
    )

    response_text = structured["answer"]
    citation_chips = structured["citations"]
    next_steps = structured["next_steps"]

    # 4. Detect tool calls (conservative — only on EXPLICIT requests)
    tools_to_call: list[dict[str, Any]] = []
    query_lower = query.lower()
    
    # Scheme lookup — only when user explicitly names a scheme
    scheme_keywords = {
        "pragati scholarship": "pragati", "aicte pragati": "pragati",
        "pmkvy scheme": "pmkvy", "kaushal vikas": "pmkvy",
        "mudra loan": "mudra", "mudra yojana": "mudra",
        "stand up india": "standup", "standup india": "standup",
    }
    for keyword, scheme_key in scheme_keywords.items():
        if keyword in query_lower:
            tools_to_call.append({
                "tool": "lookup_scheme",
                "params": {"keyword": scheme_key},
                "reason": f"User asked about scheme/program: {scheme_key}",
            })
            break
    
    # Job search — only when user explicitly asks to find jobs
    if any(phrase in query_lower for phrase in ["find jobs", "search jobs", "job listings", "job openings", "looking for jobs", "job search"]):
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
            "reason": "User is explicitly looking for job opportunities.",
        })
    
    # Scheme eligibility — only when user explicitly asks
    if any(phrase in query_lower for phrase in ["which programs am i eligible", "what programs do i qualify", "program eligibility", "career program eligibility"]):
        tools_to_call.append({
            "tool": "check_scheme_eligibility",
            "params": {"user_profile": {}, "scheme_type": "career"},
            "reason": "User is explicitly asking about career program eligibility.",
        })
    
    # Tavily web search — ONLY when user explicitly asks for latest/current info
    if any(phrase in query_lower for phrase in ["latest job market", "current job trends", "salary trends", "hiring trends", "industry trends"]):
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
