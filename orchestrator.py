"""
orchestrator.py — Root ADK agent with exactly 3 responsibilities:
1. INTENT CLASSIFICATION → HEALTH | FINANCE | CAREER
2. ROUTING → dispatch to the correct sub-agent
3. VALIDATION → 3 parallel checks before returning response
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

from observability import observe, TraceContext, metrics
from validators import run_all_validations, SAFE_FALLBACK
from memory import get_user_memory_context, ShortTermMemory
from cache import get_cached_response, set_cached_response
from clarifier import (
    get_clarifying_questions,
    is_yes_no,
    is_doc_request,
    build_clarification_context,
    opening_kalpana_line,
    filter_already_answered,
)
from dual_user import get_user_profile
import agents.health as health_agent
import agents.finance as finance_agent
import agents.career as career_agent

logger = logging.getLogger("shakti.orchestrator")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Conversation state (in-memory) ──
# Keyed by session_id. Holds clarifying phase between turns.
_conversation_state: dict[str, dict[str, Any]] = {}


def _get_session_state(session_id: str) -> dict[str, Any]:
    return _conversation_state.get(session_id, {})


def _set_session_state(session_id: str, state: dict[str, Any]) -> None:
    _conversation_state[session_id] = state


def _clear_session_state(session_id: str) -> None:
    _conversation_state.pop(session_id, None)


# How many recent answers count as the "same conversation" before we reset
# and re-clarify a new topic
_TOPIC_RESET_AFTER_TURNS = 0  # 0 = reset on first user msg after an answer


CLASSIFIER_PROMPT = (
    "You are an intent classifier for a women's life companion. "
    "Given a user query, return EXACTLY one word: HEALTH, FINANCE, or CAREER. "
    "No other output."
)

SUB_AGENTS = {
    "HEALTH": health_agent,
    "FINANCE": finance_agent,
    "CAREER": career_agent,
}


# ═══════════════════════════════════════════════════════
# 1. INTENT CLASSIFICATION
# ═══════════════════════════════════════════════════════

@observe("orchestrator_classify")
async def classify_intent(query: str) -> str:
    """
    Classify user query into HEALTH | FINANCE | CAREER.
    Uses gemini-2.5-flash with max_output_tokens=5.
    """
    try:
        from vertexai.generative_models import GenerativeModel, SafetySetting, HarmCategory, HarmBlockThreshold  # type: ignore
        model = GenerativeModel(DEFAULT_MODEL)
        
        # Configure safety settings to avoid blocking benign queries
        safety_settings = [
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=HarmBlockThreshold.BLOCK_NONE),
            SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=HarmBlockThreshold.BLOCK_NONE),
        ]
        
        response = model.generate_content(
            f"{CLASSIFIER_PROMPT}\n\nUser query: {query}",
            generation_config={"max_output_tokens": 5, "temperature": 0.0},
            safety_settings=safety_settings,
        )
        intent = response.text.strip().upper()
        if intent in ("HEALTH", "FINANCE", "CAREER"):
            return intent
        logger.warning(f"Unexpected classifier output: '{intent}' — falling back to keyword matching")
    except Exception as e:
        logger.warning(f"LLM classifier failed: {e} — using keyword fallback")

    # Keyword-based fallback
    q = query.lower()
    health_kw = ["health", "period", "pregnant", "doctor", "clinic", "menstrual", "menopause",
                 "medicine", "hospital", "symptoms", "pain", "disease", "diet", "nutrition",
                 "puberty", "antenatal", "maternity", "jsy", "pmjay", "breastfeed"]
    finance_kw = ["money", "invest", "sip", "loan", "tax", "savings", "scheme", "pension",
                  "sukanya", "elss", "mutual fund", "insurance", "bank", "emi", "budget",
                  "scss", "nps", "interest rate", "return"]
    career_kw = ["job", "career", "scholarship", "course", "study", "college", "intern",
                 "resume", "interview", "skill", "returnship", "mentor", "business",
                 "entrepreneur", "startup", "freelance", "training", "work", "back to work", "working", "employment",
                 "consulting", "consultant", "retirement", "retire"]

    scores = {
        "HEALTH": sum(1 for kw in health_kw if kw in q),
        "FINANCE": sum(1 for kw in finance_kw if kw in q),
        "CAREER": sum(1 for kw in career_kw if kw in q),
    }

    intent = max(scores, key=scores.get)  # type: ignore
    if scores[intent] == 0:
        # No keywords matched — use LLM to classify instead of defaulting to HEALTH
        try:
            from clarifier import _call_gemini as _gemini
            prompt = (
                "Classify this user query into EXACTLY one word: HEALTH, FINANCE, or CAREER.\n"
                "HEALTH = medical, body, symptoms, pregnancy, mental health, clinics\n"
                "FINANCE = money, savings, investments, schemes, insurance, tax\n"
                "CAREER = jobs, skills, education, scholarships, business, work\n\n"
                f"Query: \"{query}\"\n\n"
                "Reply with ONLY one word:"
            )
            raw = (await _gemini(prompt, max_tokens=5, temperature=0.0)).strip().upper()
            if raw in ("HEALTH", "FINANCE", "CAREER"):
                return raw
        except Exception:
            pass
        # If LLM also fails, default to CAREER (most general) instead of HEALTH
        intent = "CAREER"
    return intent


# ═══════════════════════════════════════════════════════
# 2. ROUTING
# ═══════════════════════════════════════════════════════

@observe("orchestrator_route")
async def route_to_agent(
    query: str,
    intent: str,
    age_band: str,
    user_memory_context: str,
    session_id: str,
    user_id: str = "",
) -> dict[str, Any]:
    """Route to the corresponding sub-agent."""
    agent = SUB_AGENTS.get(intent)
    if not agent:
        logger.error(f"Unknown intent: {intent}")
        return {"response": SAFE_FALLBACK, "citations": [], "tools_to_call": [], "rag_sources": []}

    result = await agent.run(
        query=query,
        age_band=age_band,
        user_memory_context=user_memory_context,
        session_id=session_id,
        user_id=user_id,
    )
    return result


# ═══════════════════════════════════════════════════════
# 3. VALIDATION
# ═══════════════════════════════════════════════════════

@observe("orchestrator_validate")
async def validate_response(
    response_text: str,
    rag_sources: list[dict[str, Any]],
    age_band: str,
    pillar: str,
) -> dict[str, Any]:
    """Run 3 parallel validation checks on the sub-agent response."""
    return await run_all_validations(
        response_text=response_text,
        rag_sources=rag_sources,
        age_band=age_band,
        pillar=pillar,
    )


# ═══════════════════════════════════════════════════════
# ORCHESTRATOR MAIN FLOW
# ═══════════════════════════════════════════════════════

async def process_query(
    user_id: str,
    age_band: str,
    query: str,
    session_id: str,
) -> dict[str, Any]:
    """
    Main orchestrator flow with clarifying questions:
      1. If no session state -> classify, store clarifying questions, ask first one
      2. If phase=clarifying -> store this answer, ask next OR move to answering
      3. If phase=answering or done -> reset for new topic (next user msg = new convo)
    Answer phase:
      4. Check cache, get memory, route to agent, validate, return.
    """
    start_time = time.time()
    state = _get_session_state(session_id)
    phase = state.get("phase")

    # ── STATE: NEW CONVERSATION ──
    if not phase or phase == "done":
        # Classify the intent up front so we can ask pillar-tuned questions
        intent = await classify_intent(query)
        template_qs = get_clarifying_questions(intent, age_band, query)

        # Fetch profile once for filtering + framing
        profile = await get_user_profile(user_id) if user_id else {}

        # Smart filter — drop questions the user already answered in their
        # original message or that the profile implies
        clarifying_qs = await filter_already_answered(template_qs, query, profile)

        # If filter returned 0 questions OR no template available, skip straight to answering
        if not clarifying_qs:
            logger.info(f"Skipping clarifying — user query already covers all questions: '{query[:80]}'")
            _set_session_state(session_id, {
                "phase": "answering_inline",
                "pillar": intent,
                "original_query": query,
                "qa_pairs": [],
            })
            return await _generate_full_answer(
                user_id=user_id,
                age_band=age_band,
                query=query,
                pillar=intent,
                qa_pairs=[],
                session_id=session_id,
                start_time=start_time,
            )

        name = profile.get("preferred_name") or ""
        opener = opening_kalpana_line(intent, name=name)

        # Store state and return the first clarifying question
        _set_session_state(session_id, {
            "phase": "clarifying",
            "pillar": intent,
            "original_query": query,
            "clarifying_questions": clarifying_qs,
            "qa_pairs": [],
            "asked_index": 0,
        })

        first_q = clarifying_qs[0]
        latency_ms = (time.time() - start_time) * 1000

        return {
            "type": "clarifying_question",
            "response": f"{opener}\n\n{first_q}",
            "kalpana_says": opener,
            "question": first_q,
            "is_yes_no": is_yes_no(first_q),
            "is_doc_request": is_doc_request(first_q),
            "question_index": 0,
            "questions_total": len(clarifying_qs),
            "pillar": intent,
            "sub_agent": intent,
            "age_band": age_band,
            "session_id": session_id,
            "citations": [],
            "tools_to_call": [],
            "awaiting_approval": False,
            "from_cache": False,
            "latency_ms": round(latency_ms, 1),
            "steps": [
                {"step": "classified", "detail": f"Intent: {intent}"},
                {"step": "clarifying", "detail": "Asking clarifying questions..."},
            ],
        }

    # ── STATE: CLARIFYING ── store this answer, advance or finalize
    if phase == "clarifying":
        clarifying_qs = state.get("clarifying_questions", [])
        idx = state.get("asked_index", 0)
        qa_pairs = state.get("qa_pairs", [])

        if idx < len(clarifying_qs):
            qa_pairs.append({
                "q": clarifying_qs[idx],
                "a": query,
            })

        new_idx = idx + 1

        if new_idx < len(clarifying_qs):
            # Ask the next one
            next_q = clarifying_qs[new_idx]
            _set_session_state(session_id, {
                **state,
                "qa_pairs": qa_pairs,
                "asked_index": new_idx,
            })
            latency_ms = (time.time() - start_time) * 1000
            return {
                "type": "clarifying_question",
                "response": next_q,
                "question": next_q,
                "is_yes_no": is_yes_no(next_q),
                "is_doc_request": is_doc_request(next_q),
                "question_index": new_idx,
                "questions_total": len(clarifying_qs),
                "pillar": state.get("pillar"),
                "sub_agent": state.get("pillar"),
                "age_band": age_band,
                "session_id": session_id,
                "citations": [],
                "tools_to_call": [],
                "awaiting_approval": False,
                "from_cache": False,
                "latency_ms": round(latency_ms, 1),
                "steps": [{"step": "clarifying", "detail": f"Question {new_idx + 1} of {len(clarifying_qs)}"}],
            }

        # All clarifying questions answered — synthesize the real answer
        _set_session_state(session_id, {
            **state,
            "qa_pairs": qa_pairs,
            "phase": "answering",
        })

        return await _generate_full_answer(
            user_id=user_id,
            age_band=age_band,
            query=state.get("original_query", query),
            pillar=state.get("pillar"),
            qa_pairs=qa_pairs,
            session_id=session_id,
            start_time=start_time,
        )

    # ── STATE: DONE ── decide: follow-up on same topic, or new topic?
    last_pillar = state.get("last_pillar")
    last_query = state.get("last_query")
    last_answer = state.get("last_answer_preview", "")

    is_followup = False
    if last_pillar and last_query:
        is_followup = await _is_followup(
            new_query=query,
            last_query=last_query,
            last_pillar=last_pillar,
        )

    if is_followup:
        # Same topic — skip clarifying, but RE-CLASSIFY the new query's intent
        # so we route to the correct agent (not the old one)
        new_intent = await classify_intent(query)
        logger.info(f"Follow-up detected. Re-classified intent: {new_intent} (was {last_pillar})")
        
        # Build conversation history into the enriched query
        prior_qa = state.get("qa_pairs", []) or []
        followup_history = list(prior_qa)
        if last_answer:
            followup_history.append({
                "q": f"(previous answer about {last_pillar.lower()})",
                "a": last_answer,
            })
        followup_history.append({
            "q": "User's follow-up",
            "a": query,
        })

        _set_session_state(session_id, {
            **state,
            "phase": "answering",
            "pillar": new_intent,  # Use the NEW intent, not the old one
        })

        return await _generate_full_answer(
            user_id=user_id,
            age_band=age_band,
            query=query,  # Use the NEW query, not the old one
            pillar=new_intent,  # Route to the correct agent
            qa_pairs=followup_history,
            session_id=session_id,
            start_time=start_time,
        )

    # New topic — reset and re-clarify
    logger.info(f"New topic detected, resetting clarifying flow: '{query[:60]}'")
    _clear_session_state(session_id)
    return await process_query(user_id=user_id, age_band=age_band, query=query, session_id=session_id)


async def _is_followup(*, new_query: str, last_query: str, last_pillar: str) -> bool:
    """Conservative follow-up detection.
    
    Only returns True when the new message is clearly a continuation of the
    SAME topic — not when it's a fresh question about a different pillar.
    
    Uses LLM as primary signal. Cheap heuristics are only for obvious cases
    like single-word affirmations.
    """
    q = (new_query or "").strip().lower()
    if not q:
        return False

    # Only the most obvious single-word follow-ups — NOT "why", "how", "more"
    # which could be about anything
    obvious_followups = (
        "yes", "no", "ok", "okay", "sure", "thanks", "thank you",
        "go on", "continue", "next", "skip", "done",
    )
    if q in obvious_followups:
        return True

    # Very short referential phrases that ONLY make sense in context
    strictly_referential = (
        "tell me more about that",
        "what else",
        "anything else",
        "and then",
        "what about that",
    )
    if q in strictly_referential:
        return True

    # For everything else — use LLM to decide. This is the primary signal.
    try:
        from clarifier import _call_gemini as _gemini
        prompt = (
            "A user is chatting with a women's life companion AI.\n"
            f"Their PREVIOUS topic was: {last_pillar}\n"
            f"Their PREVIOUS question: \"{last_query[:200]}\"\n"
            f"Their NEW message: \"{new_query[:200]}\"\n\n"
            "Is the NEW message clearly about the SAME topic as the previous "
            "question? Or is it a fresh, unrelated question about a different "
            "topic (e.g. switching from health to finance, or from career to health)?\n\n"
            "Reply with EXACTLY one word: 'followup' if same topic, 'new' if different topic."
        )
        raw = (await _gemini(prompt, max_tokens=5, temperature=0.0)).strip().lower()
        logger.info(f"Follow-up LLM decision: '{raw}' for query '{new_query[:60]}'")
        return raw.startswith("followup")
    except Exception as e:
        logger.warning(f"Follow-up LLM check failed: {e}")
        # If LLM fails, default to NOT follow-up (safer to re-clarify)
        return False


async def _generate_full_answer(
    user_id: str,
    age_band: str,
    query: str,
    pillar: Optional[str],
    qa_pairs: list[dict[str, str]],
    session_id: str,
    start_time: float,
) -> dict[str, Any]:
    """Synthesise the final answer using the original query + clarifying Q&A."""
    # Initialize trace
    trace = TraceContext(
        session_id=session_id,
        user_id=user_id,
        metadata={"age_band": age_band},
    )
    trace.start_trace("shakti_orchestrator")

    steps: list[dict[str, str]] = []

    # Step 1: Classify intent (use stored pillar if available, else reclassify)
    intent = (pillar or "").upper() if pillar else await classify_intent(query)
    if intent not in ("HEALTH", "FINANCE", "CAREER"):
        intent = await classify_intent(query)
    steps.append({"step": "classified", "detail": f"Intent: {intent}"})

    # Build enriched query string for cache key and agent input
    enriched_query = build_clarification_context(query, qa_pairs)

    # Check cache (using enriched query so different clarifications => different cache)
    cached = await get_cached_response(enriched_query, age_band, intent)
    if cached:
        latency_ms = (time.time() - start_time) * 1000
        metrics.record_session(latency_ms, intent, age_band, cost=0.0)
        prior_state = _get_session_state(session_id)
        _set_session_state(session_id, {
            **prior_state,
            "phase": "done",
            "last_pillar": intent,
            "last_query": prior_state.get("original_query") or query,
            "last_answer_preview": (cached.get("response") or "")[:600],
        })
        return {**cached, "from_cache": True, "latency_ms": round(latency_ms, 1), "type": "answer"}

    # Step 2: Get memory context
    steps.append({"step": "memory", "detail": "Loading user context..."})
    memory_context = await get_user_memory_context(user_id, query)

    # Step 3: Route to sub-agent with enriched query
    steps.append({"step": "routing", "detail": f"Retrieving {intent.lower()} docs for {age_band}..."})
    agent_result = await route_to_agent(
        query=enriched_query,
        intent=intent,
        age_band=age_band,
        user_memory_context=memory_context,
        session_id=session_id,
        user_id=user_id,
    )

    response_text = agent_result.get("response", "")
    citations = agent_result.get("citations", [])
    citation_chips = agent_result.get("citation_chips", [])
    next_steps = agent_result.get("next_steps", [])
    tools_to_call = agent_result.get("tools_to_call", [])
    rag_sources = agent_result.get("rag_sources", [])
    diagnostics = agent_result.get("diagnostics", {})

    # Step 4: Validate response
    steps.append({"step": "validating", "detail": "Validating response..."})
    validation = await validate_response(
        response_text=response_text,
        rag_sources=rag_sources,
        age_band=age_band,
        pillar=intent,
    )

    # Check validation and construct warnings
    validation_warnings = []
    if not validation["all_passed"]:
        failed_checks = validation.get("failed_checks", [])
        for chk in failed_checks:
            detail = ""
            if chk == "citation":
                detail = validation["citation_check"].get("details", "")
            elif chk == "age_appropriateness":
                detail = validation["age_appropriateness_check"].get("details", "")
            elif chk == "domain_safety":
                detail = validation["domain_safety_check"].get("details", "")
            validation_warnings.append(f"{chk}: {detail}")
            logger.warning(f"Validation WARNING - check '{chk}' failed: {detail}")

    # Determine mode: STRICT_VALIDATION flag
    strict_val_env = os.getenv("STRICT_VALIDATION", "false").lower()
    is_strict = strict_val_env in ("true", "1", "yes")

    hallucination_passed = validation["all_passed"]
    if not hallucination_passed:
        if is_strict:
            response_text = validation.get("fallback_response", SAFE_FALLBACK)
            logger.warning(f"Strict validation FAILED, using fallback: {validation.get('failed_checks', [])}")
        else:
            logger.info(f"Validation FAILED in soft mode (STRICT_VALIDATION=false). Warnings: {validation_warnings}")

    latency_ms = (time.time() - start_time) * 1000

    # Build final response
    result = {
        "type": "answer",
        "response": response_text,
        "sub_agent": intent,
        "pillar": intent,
        "age_band": age_band,
        "citations": citations,
        "citation_chips": citation_chips,
        "next_steps": next_steps,
        "tools_to_call": tools_to_call,
        "awaiting_approval": len(tools_to_call) > 0,
        "session_id": session_id,
        "validation": {
            "all_passed": validation["all_passed"],
            "citation_check": validation["citation_check"]["passed"],
            "age_check": validation["age_appropriateness_check"]["passed"],
            "safety_check": validation["domain_safety_check"]["passed"],
        },
        "validation_warnings": validation_warnings,
        "latency_ms": round(latency_ms, 1),
        "from_cache": False,
        "steps": steps,
        "clarifying_qa": qa_pairs,
        "response_diagnostics": diagnostics,
    }

    # Record metrics
    metrics.record_session(
        latency_ms=latency_ms,
        pillar=intent,
        age_band=age_band,
        hallucination_passed=hallucination_passed,
    )

    # Cache the result (without steps/latency for consistency)
    cache_data = {k: v for k, v in result.items() if k not in ("steps", "latency_ms", "from_cache")}
    await set_cached_response(enriched_query, age_band, intent, cache_data)

    # Mark conversation as done and remember the answered topic so a follow-up
    # user message can be detected and routed without re-clarifying.
    prior_state = _get_session_state(session_id)
    _set_session_state(session_id, {
        **prior_state,
        "phase": "done",
        "last_pillar": intent,
        "last_query": prior_state.get("original_query") or query,
        "last_answer_preview": (response_text or "")[:600],
    })

    # Log trace
    trace.add_generation(
        name=f"{intent.lower()}_response",
        model=agent_result.get("model_used", DEFAULT_MODEL),
        input_text=enriched_query,
        output_text=response_text,
        metadata={
            "age_band": age_band,
            "pillar": intent,
            "citations_count": len(citations),
            "hallucination_check_passed": hallucination_passed,
            "latency_ms": round(latency_ms, 1),
            "clarifying_qa_count": len(qa_pairs),
        },
    )
    trace_url = trace.finalize()
    if trace_url:
        result["trace_url"] = trace_url

    return result
