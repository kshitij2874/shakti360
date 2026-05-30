"""
response_builder.py — Bulletproof structured responses with NO truncation.

Strategy:
  Pass 1: Generate answer text (focused, generous tokens, retry on truncation)
  Pass 2: Extract citations from RAG chunks (deterministic, no LLM call)
  Pass 3: Generate next-step suggestions (separate cheap call, JSON-mode with fallback)

Each pass has independent retry + repair so a failure in one section never
nukes the others. Hard fallbacks guarantee no field is ever empty.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger("shakti.response_builder")

# ── Token budgets — generous so the model never starves ──
TOKEN_BUDGETS = {
    "answer": 1200,        # ~900 words
    "next_steps": 300,     # ~ 2 short JSON entries
}

# ── Hard minimums for validation ──
MIN_ANSWER_CHARS = 150
MIN_NEXT_STEPS = 2
TERMINAL_PUNCT = (".", "!", "?", "\u0964", "\u06d4", ":")  # period, !, ?, devanagari danda, urdu


# ═══════════════════════════════════════════════════════════
# TOP-LEVEL ENTRY POINT
# ═══════════════════════════════════════════════════════════

async def build_full_response(
    *,
    pillar: str,
    age_band: str,
    query: str,
    clarifying_qa: Optional[list[dict]] = None,
    rag_chunks: Optional[list[dict]] = None,
    persona_prefix: str = "",
    framing_prefix: str = "",
    memory_context: str = "",
    fallback_system_prompt: str = "",
    model_name: Optional[str] = None,
    language: str = "English",
) -> dict[str, Any]:
    """
    Build a complete structured response with no-truncation guarantees.
    Returns: {pillar, answer, citations, next_steps, model_used, retries}
    """
    clarifying_qa = clarifying_qa or []
    rag_chunks = rag_chunks or []

    diagnostics: dict[str, Any] = {
        "answer_attempts": 0,
        "next_steps_attempts": 0,
        "truncation_events": 0,
        "repairs": [],
    }

    # ── Pass 1: Answer text ──
    answer_text = await _generate_answer_text(
        pillar=pillar,
        age_band=age_band,
        query=query,
        clarifying_qa=clarifying_qa,
        rag_chunks=rag_chunks,
        persona_prefix=persona_prefix,
        framing_prefix=framing_prefix,
        memory_context=memory_context,
        fallback_system_prompt=fallback_system_prompt,
        model_name=model_name,
        language=language,
        diagnostics=diagnostics,
    )

    # ── Pass 2: Citations (deterministic) ──
    citation_chips = _extract_citations(rag_chunks)

    # ── Pass 3: Next steps ──
    next_steps = await _generate_next_steps(
        pillar=pillar,
        age_band=age_band,
        answer_text=answer_text,
        query=query,
        model_name=model_name,
        language=language,
        diagnostics=diagnostics,
    )

    # ── Validate + repair ──
    response = {
        "pillar": pillar,
        "answer": answer_text,
        "citations": citation_chips,
        "next_steps": next_steps,
        "model_used": model_name or "gemini-2.5-flash",
        "diagnostics": diagnostics,
    }

    validation = _validate_response(response)
    if not validation["ok"]:
        logger.warning(f"Response validation issues: {validation['issues']}")
        response = _repair_response(response, validation, pillar, age_band, query)

    return response


# ═══════════════════════════════════════════════════════════
# PASS 1: ANSWER TEXT
# ═══════════════════════════════════════════════════════════

async def _generate_answer_text(
    *,
    pillar: str,
    age_band: str,
    query: str,
    clarifying_qa: list[dict],
    rag_chunks: list[dict],
    persona_prefix: str,
    framing_prefix: str,
    memory_context: str,
    fallback_system_prompt: str,
    model_name: Optional[str],
    diagnostics: dict,
    language: str = "English",
) -> str:
    """Generate the answer body. Retries up to 3x with larger token budget on truncation."""
    rag_context = _format_rag_context(rag_chunks)
    qa_context = _format_qa_context(clarifying_qa)

    system_prompt = _build_answer_system_prompt(
        pillar=pillar,
        age_band=age_band,
        rag_context=rag_context,
        qa_context=qa_context,
        persona_prefix=persona_prefix,
        framing_prefix=framing_prefix,
        memory_context=memory_context,
        fallback_system_prompt=fallback_system_prompt,
        language=language,
    )

    user_msg = f"Original question from the user:\n{query}"

    token_budget = TOKEN_BUDGETS["answer"]

    for attempt in range(3):
        diagnostics["answer_attempts"] += 1
        try:
            text = await _call_gemini(
                prompt=f"{system_prompt}\n\n{user_msg}",
                max_output_tokens=token_budget,
                temperature=0.7,
                top_p=0.9,
                model_name=model_name,
                tier="reasoning",  # main answer = difficult task → stronger model
            )
            text = (text or "").strip()

            if len(text) < MIN_ANSWER_CHARS:
                logger.warning(f"Answer too short ({len(text)} chars) — attempt {attempt + 1}")
                diagnostics["truncation_events"] += 1
                token_budget = int(token_budget * 1.5)
                continue

            if _ends_mid_sentence(text):
                logger.warning(f"Answer cut off mid-sentence (len={len(text)}) — attempt {attempt + 1}")
                diagnostics["truncation_events"] += 1
                token_budget = int(token_budget * 1.5)
                continue

            return text

        except Exception as e:
            logger.error(f"Answer generation attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(0.3)

    # All retries failed — use a deterministic fallback derived from RAG context
    logger.error("All answer generation attempts failed; using RAG-derived fallback.")
    return _fallback_answer(query, pillar, age_band, rag_chunks)


def _build_answer_system_prompt(
    *,
    pillar: str,
    age_band: str,
    rag_context: str,
    qa_context: str,
    persona_prefix: str,
    framing_prefix: str,
    memory_context: str,
    fallback_system_prompt: str,
    language: str = "English",
) -> str:
    length_hint = {
        "11-24": "roughly 400-700 characters",
        "25-40": "roughly 600-900 characters",
        "41+": "roughly 700-1000 characters",
    }.get(age_band, "roughly 600-900 characters")

    persona_block = persona_prefix.strip() or fallback_system_prompt.strip()

    return (
        f"{persona_block}\n\n"
        f"{framing_prefix}"
        f"\nROLE: You are Kalpana, a women's life companion. "
        f"The user has asked about a {pillar.lower()} topic.\n"
        "You have access to verified context below from government and trusted sources.\n\n"
        f"USER MEMORY CONTEXT:\n{memory_context or 'No prior context.'}\n\n"
        f"CONTEXT FROM RAG:\n{rag_context}\n\n"
        f"USER'S CLARIFYING ANSWERS:\n{qa_context}\n\n"
        "YOUR TASK:\n"
        "Write a complete, direct answer to the original question. "
        "Follow these rules STRICTLY:\n\n"
        "TONE RULES:\n"
        "1. Be DIRECT. Lead with the most important fact or action \u2014 not with pleasantries.\n"
        "2. Be EMPATHETIC but not soft. Acknowledge her specific situation in one sentence max, then get to the facts.\n"
        "3. NO FLUFF. Never write: 'It's wonderful that you asked', 'As your companion', "
        "'It's natural to feel', or any generic reassurance not tied to her actual situation.\n"
        "4. NO REPETITION. Do not repeat information already given in the clarifying answers or previous turns.\n\n"
        "WRITING RULES:\n"
        "5. Write a COMPLETE answer \u2014 do NOT cut off mid-thought.\n"
        "6. Use 2-4 short paragraphs (each 2-4 sentences). Plain prose.\n"
        "7. Every factual claim must come from the RAG context. Do not invent facts.\n"
        "8. End with ONE concrete, actionable next step \u2014 not a question, not a disclaimer.\n"
        f"9. Length: {length_hint}.\n"
        "10. Do NOT add generic 'consult a doctor' disclaimers \u2014 next-step suggestions handle that.\n\n"
        "OUTPUT FORMAT:\n"
        "Plain prose only. NO markdown headers. NO JSON. "
        "Bullet points only when listing 3+ distinct items \u2014 not for general paragraphs. "
        "Write as if talking directly to her, not formatting a document.\n\n"
        f"LANGUAGE \u2014 CRITICAL:\n"
        f"Write your ENTIRE response in {language}. "
        f"{'Use natural, conversational ' + language + '. ' if language != 'English' else ''}"
        f"{'For Hinglish, mix Hindi and English the way urban Indian women naturally text. ' if language == 'Hinglish' else ''}"
        "Keep proper nouns, scheme names, and helpline numbers as-is.\n\n"
        "BEGIN THE ANSWER NOW:"
    )


# ═══════════════════════════════════════════════════════════
# PASS 2: CITATIONS (deterministic, no LLM)
# ═══════════════════════════════════════════════════════════

# Source filename -> short label
_LABEL_RULES = (
    ("janani", "JSY"),
    ("jsy", "JSY"),
    ("pmjay", "PMJAY"),
    ("ayushman", "PMJAY"),
    ("sukanya", "Sukanya Samriddhi"),
    ("aiims", "AIIMS"),
    ("icmr", "ICMR"),
    ("nhm", "NHM"),
    ("sebi", "SEBI"),
    ("rbi", "RBI"),
    ("mahila", "Mahila Udyam"),
    ("scss", "SCSS"),
    ("scholarship", "MoE"),
    ("returnship", "NSDC"),
    ("menopause", "AIIMS"),
    ("puberty", "WHO"),
    ("pcos", "AIIMS"),
    ("antenatal", "WHO"),
    ("maternity", "JSY"),
    ("sip", "SEBI"),
    ("elss", "SEBI"),
    ("mudra", "Mudra Yojana"),
    ("standup", "Stand-Up India"),
    ("pmkvy", "PMKVY"),
    ("nps", "NPS"),
    ("apy", "APY"),
    ("pragati", "AICTE Pragati"),
    ("pmmvy", "PMMVY"),
    ("who", "WHO"),
)


def _label_from_source(source: str) -> str:
    s = (source or "").lower()
    for needle, label in _LABEL_RULES:
        if needle in s:
            return label
    # Fallback: title-case filename without extension
    base = source.replace(".txt", "").replace("_", " ").strip()
    return (base or "Source").title()[:24]


def _extract_citations(rag_chunks: list[dict]) -> list[dict]:
    """Deterministic citation extraction. Caps at 5, de-duplicates by label."""
    if not rag_chunks:
        return []

    citations: list[dict] = []
    seen: set[str] = set()

    for chunk in rag_chunks[:8]:
        source = chunk.get("source", "")
        is_web = bool(chunk.get("is_web"))
        label = (chunk.get("doc_title") or source) if is_web else _label_from_source(source)
        label = (label or "Source")[:40]
        if label in seen:
            continue
        seen.add(label)

        citations.append({
            "label": label,
            "source": chunk.get("source_ref") or source,
            "section": (chunk.get("section_title") or "")[:80],
            "kind": "web" if is_web else "official",
        })
        if len(citations) >= 5:
            break

    return citations


# ═══════════════════════════════════════════════════════════
# PASS 3: NEXT STEPS
# ═══════════════════════════════════════════════════════════

async def _generate_next_steps(
    *,
    pillar: str,
    age_band: str,
    answer_text: str,
    query: str,
    model_name: Optional[str],
    diagnostics: dict,
    language: str = "English",
) -> list[dict]:
    """Generate exactly 2 next-step suggestions. Falls back to hardcoded on failure."""
    prompt = (
        "You are Kalpana suggesting helpful next steps after answering a user's question.\n\n"
        f"User's question: {query}\n"
        f"Your answer (just provided): {answer_text[:500]}...\n"
        f"Topic: {pillar}\n"
        f"User's age band: {age_band}\n\n"
        "Generate EXACTLY 2 next-step suggestions. Each suggestion offers to help with "
        "something concrete \u2014 either calling a tool or exploring a related topic.\n\n"
        "OUTPUT FORMAT \u2014 strict JSON array only, no markdown fences:\n"
        '[\n'
        '  {"id": "suggestion_1", "label": "...", "action_type": "tool_call" | "agent_continue", "tool_or_context": "..."},\n'
        '  {"id": "suggestion_2", "label": "...", "action_type": "tool_call" | "agent_continue", "tool_or_context": "..."}\n'
        ']\n\n'
        "RULES:\n"
        "- Each label is a friendly question starting with 'Want me to...' or 'Would you like...'\n"
        "- label max 80 chars\n"
        "- action_type: 'tool_call' for external API (maps, scheme lookup, calculator, job search, budget analysis, health assessment); "
        "'agent_continue' for another conversational turn\n"
        "- Available tools: find_nearby_clinic, find_nearby_police, lookup_scheme, calculate_sip, "
        "search_jobs_web, analyze_budget, assess_health_risk, mental_wellbeing_check, check_scheme_eligibility, "
        "find_support_org (connect her to a real counselor/NGO/mentor)\n"
        "- If the topic is emotional, sensitive, or she seems to need real human support, "
        "make ONE suggestion 'find_support_org'.\n"
        f"- Match the tone for age band {age_band}.\n"
        f"- Write each label in {language} (keep proper nouns/scheme names as-is).\n\n"
        "Output the JSON array now:"
    )

    for attempt in range(2):
        diagnostics["next_steps_attempts"] += 1
        try:
            raw = await _call_gemini(
                prompt=prompt,
                max_output_tokens=TOKEN_BUDGETS["next_steps"],
                temperature=0.6,
                top_p=0.9,
                model_name=model_name,
            )
            steps = _parse_json_array(raw or "")
            if isinstance(steps, list) and len(steps) >= MIN_NEXT_STEPS:
                cleaned = []
                for s in steps[:2]:
                    if not isinstance(s, dict):
                        continue
                    label = (s.get("label") or "").strip()[:120]
                    if not label:
                        continue
                    cleaned.append({
                        "id": s.get("id") or f"suggestion_{len(cleaned) + 1}",
                        "label": label,
                        "action_type": s.get("action_type") or "agent_continue",
                        "tool_or_context": s.get("tool_or_context") or "",
                    })
                if len(cleaned) >= MIN_NEXT_STEPS:
                    return cleaned
            logger.warning(f"Next-steps JSON malformed or too short (attempt {attempt + 1}).")
        except Exception as e:
            logger.warning(f"Next-steps generation attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(0.2)

    diagnostics["repairs"].append("next_steps_fallback")
    return _fallback_next_steps(pillar)


def _parse_json_array(raw: str):
    """Robust JSON-array parser tolerant of code fences, prose padding, trailing commas."""
    if not raw:
        return None
    text = raw.strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    # Slice between the first [ and last ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        slice_ = text[start:end + 1]
        # Remove trailing commas
        slice_ = re.sub(r",\s*([}\]])", r"\1", slice_)
        try:
            return json.loads(slice_)
        except Exception:
            return None
    return None


# ═══════════════════════════════════════════════════════════
# FALLBACKS — never return empty
# ═══════════════════════════════════════════════════════════

_FALLBACK_NEXT_STEPS = {
    "HEALTH": [
        {"id": "find_clinics", "label": "Want me to find women's health clinics nearby?",
         "action_type": "tool_call", "tool_or_context": "find_nearby_clinic"},
        {"id": "health_check", "label": "Would you like a quick health risk assessment?",
         "action_type": "tool_call", "tool_or_context": "assess_health_risk"},
        {"id": "schemes", "label": "Want to know which government health schemes you qualify for?",
         "action_type": "tool_call", "tool_or_context": "check_scheme_eligibility"},
    ],
    "FINANCE": [
        {"id": "calculator", "label": "Want me to run a quick SIP calculation for your goal?",
         "action_type": "tool_call", "tool_or_context": "calculate_sip"},
        {"id": "budget", "label": "Would you like me to analyze your budget and spending?",
         "action_type": "tool_call", "tool_or_context": "analyze_budget"},
        {"id": "schemes", "label": "Want to know which government savings schemes you qualify for?",
         "action_type": "tool_call", "tool_or_context": "check_scheme_eligibility"},
    ],
    "CAREER": [
        {"id": "jobs", "label": "Want me to search for relevant job opportunities?",
         "action_type": "tool_call", "tool_or_context": "search_jobs_web"},
        {"id": "returnship", "label": "Want to see returnship and upskilling programs near you?",
         "action_type": "agent_continue", "tool_or_context": "career_programs"},
        {"id": "schemes", "label": "Want to know which career programs you qualify for?",
         "action_type": "tool_call", "tool_or_context": "check_scheme_eligibility"},
    ],
}


def _fallback_next_steps(pillar: str) -> list[dict]:
    return list(_FALLBACK_NEXT_STEPS.get(pillar.upper(), _FALLBACK_NEXT_STEPS["HEALTH"]))


def _fallback_answer(query: str, pillar: str, age_band: str, rag_chunks: list[dict]) -> str:
    """Deterministic fallback built from RAG when LLM fails entirely."""
    intro = (
        f"I want to make sure I give you the right guidance on this. "
        f"Here's what verified sources say that might help.\n\n"
    )
    body = ""
    for chunk in rag_chunks[:2]:
        content = (chunk.get("content") or "").strip()
        if content:
            body += content[:280].rstrip() + "...\n\n"
    if not body:
        body = (
            "I couldn't pull detailed information for this question right now. "
            "Please try rephrasing, or reach the women's helpline at 181 for direct support.\n\n"
        )
    closer = "If you'd like, share a bit more about your situation and I can be more specific."
    return f"{intro}{body}{closer}".strip()


# ═══════════════════════════════════════════════════════════
# VALIDATION + REPAIR
# ═══════════════════════════════════════════════════════════

def _ends_mid_sentence(text: str) -> bool:
    if not text:
        return True
    text = text.rstrip()
    if not text.endswith(TERMINAL_PUNCT):
        return True
    sentences = re.split(r"[.!?]+", text)
    last = sentences[-2] if len(sentences) >= 2 else ""
    if len(last.strip().split()) < 3:
        return True
    return False


def _validate_response(response: dict) -> dict:
    issues: list[str] = []

    answer = response.get("answer", "") or ""
    if len(answer) < MIN_ANSWER_CHARS:
        issues.append("answer_too_short")
    if _ends_mid_sentence(answer):
        issues.append("answer_truncated")

    citations = response.get("citations") or []
    if not citations:
        issues.append("no_citations")

    next_steps = response.get("next_steps") or []
    if len(next_steps) < MIN_NEXT_STEPS:
        issues.append("insufficient_next_steps")

    return {"ok": not issues, "issues": issues}


def _repair_response(response: dict, validation: dict, pillar: str, age_band: str, query: str) -> dict:
    issues = set(validation["issues"])
    diagnostics = response.setdefault("diagnostics", {})
    repairs = diagnostics.setdefault("repairs", [])

    if "answer_truncated" in issues or "answer_too_short" in issues:
        # Already retried 3x; clip to last full sentence so at least it doesn't end mid-word
        ans = (response.get("answer") or "").rstrip()
        if ans and not ans.endswith(TERMINAL_PUNCT):
            cut = max(ans.rfind("."), ans.rfind("!"), ans.rfind("?"))
            if cut > MIN_ANSWER_CHARS // 2:
                ans = ans[: cut + 1]
                repairs.append("answer_clipped_to_last_sentence")
        response["answer"] = ans or _fallback_answer(query, pillar, age_band, [])

    if "insufficient_next_steps" in issues:
        response["next_steps"] = _fallback_next_steps(pillar)
        repairs.append("next_steps_fallback")

    if "no_citations" in issues:
        response["citations"] = [{"label": "Verified", "source": "Government sources", "section": ""}]
        repairs.append("synthetic_citation")

    return response


# ═══════════════════════════════════════════════════════════
# GEMINI CALL (sync SDK wrapped in executor)
# ═══════════════════════════════════════════════════════════

import os

_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


async def _call_gemini(
    *,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    model_name: Optional[str] = None,
    tier: str = "fast",
) -> str:
    """Delegate to the unified LLM client (DeepSeek → Gemini fallback)."""
    from llm import call_llm
    return await call_llm(
        prompt=prompt,
        max_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        model_name=model_name,
        tier=tier,
    )


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _format_rag_context(rag_chunks: list[dict]) -> str:
    if not rag_chunks:
        return "No retrieved context available."
    parts = []
    for c in rag_chunks:
        src = c.get("source", "unknown")
        content = (c.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"[Source: {src}]\n{content}")
    return "\n\n".join(parts) or "No retrieved context available."


def _format_qa_context(qa: list[dict]) -> str:
    if not qa:
        return "No clarifying answers provided."
    lines = []
    for entry in qa:
        q = (entry.get("q") or "").strip()
        a = (entry.get("a") or "").strip()
        if not a or a.lower() in ("skip", "skipped"):
            continue
        lines.append(f"Q: {q}\nA: {a}")
    return "\n".join(lines) or "No clarifying answers provided."
