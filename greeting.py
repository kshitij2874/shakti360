"""
greeting.py — Personalized contextual greetings for returning users.

Two enhancements over a basic 'last topic' greeting:
  1. extract_life_context() distills a memorable life-context phrase from the
     last session (e.g. "She was preparing to move to Hyderabad alone").
  2. Greeting prompt references this specific life-context, not just the topic,
     so Kalpana feels like a friend remembering — not a CRM auto-message.

Performance note: life_context is best computed at write-time (after each
final answer) and stored on the session doc, so /greeting doesn't pay an
extra LLM call per visit. We try the stored value first, then fall back
to live extraction.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("shakti.greeting")

_db = None
_firestore_available = False


def _get_db():
    global _db, _firestore_available
    if _db is not None:
        return _db
    try:
        from google.cloud import firestore  # type: ignore
        _db = firestore.Client()
        _firestore_available = True
        return _db
    except Exception as e:
        logger.warning(f"Firestore unavailable for greeting: {e}")
        _firestore_available = False
        return None


# ── Local fallback for session history ──
_local_sessions: list[dict] = []


def _add_local_session(user_id: str, session_data: dict) -> None:
    session_data["user_id"] = user_id
    _local_sessions.append(session_data)
    if len(_local_sessions) > 50:
        _local_sessions.pop(0)


def _get_last_local_session(user_id: str) -> Optional[dict]:
    user_sessions = [s for s in _local_sessions if s.get("user_id") == user_id]
    if not user_sessions:
        return None
    return sorted(user_sessions, key=lambda x: x.get("created_at", 0), reverse=True)[0]


# ═══════════════════════════════════════════════════════════
# LIFE-CONTEXT EXTRACTION
# ═══════════════════════════════════════════════════════════

async def _call_gemini(prompt: str, max_tokens: int = 80, temperature: float = 0.4) -> str:
    """Run the sync Vertex AI SDK in a thread, return text or empty string on failure."""
    try:
        import vertexai  # type: ignore
        from vertexai.generative_models import GenerativeModel  # type: ignore
    except Exception as e:
        logger.warning(f"Vertex AI SDK unavailable: {e}")
        return ""

    def _do() -> str:
        try:
            vertexai.init(
                project=os.getenv("PROJECT_ID", "shakti360"),
                location=os.getenv("VERTEX_LOCATION", "us-central1"),
            )
            model = GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
            resp = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            return (resp.text or "").strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"Gemini call failed in greeting: {e}")
            return ""

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do)


async def extract_life_context(session_data: dict) -> str:
    """Extract ONE memorable life-context fact from a past conversation.

    Returns a short phrase (<= 25 words) Kalpana can reference next time, or
    empty string if extraction fails / the session has no usable content.
    """
    if not session_data:
        return ""

    query = (session_data.get("query") or "").strip()
    pillar = (session_data.get("pillar") or "").strip()
    if not query:
        return ""

    # Pre-existing fields may carry useful detail
    answer = (session_data.get("answer") or session_data.get("response_preview") or "")[:300]
    clarifying = session_data.get("clarifying_qa") or session_data.get("clarifying_answers") or []
    clarifying_str = ""
    if isinstance(clarifying, list):
        bits = []
        for qa in clarifying[:4]:
            if isinstance(qa, dict):
                q = (qa.get("q") or "").strip()
                a = (qa.get("a") or "").strip()
                if a and a.lower() not in ("skip", "skipped"):
                    bits.append(f"  - {q} -> {a}")
        clarifying_str = "\n".join(bits)

    prompt = (
        "Read this past conversation snapshot and extract ONE memorable "
        "life-context fact about the user \u2014 something a caring friend "
        "would remember and bring up next time.\n\n"
        f"Topic: {pillar}\n"
        f"Original question: {query}\n"
        + (f"Clarifying details from the user:\n{clarifying_str}\n" if clarifying_str else "")
        + (f"Answer summary: {answer}\n" if answer else "")
        + "\nExamples of GOOD life-context phrases:\n"
        "- \"She was preparing to move to Hyderabad alone for the first time\"\n"
        "- \"She started planning her first pregnancy in 6 months\"\n"
        "- \"She was choosing between two career switches after maternity leave\"\n"
        "- \"She was setting up her first SIP at age 26\"\n"
        "- \"She was navigating early menopause symptoms\"\n\n"
        "Rules:\n"
        "- ONE sentence, max 25 words.\n"
        "- Third person (she/her), even if the user used 'I'.\n"
        "- Concrete and specific. Avoid generic phrases like 'asking about health'.\n"
        "- If there is not enough detail to be specific, return an empty string.\n\n"
        "Return ONLY the phrase, no preamble or quotes."
    )

    text = await _call_gemini(prompt, max_tokens=60, temperature=0.3)
    # Strip stray bullets/labels
    text = text.replace("\n", " ").strip()
    if text.lower().startswith("life-context"):
        text = text.split(":", 1)[-1].strip()
    return text[:200]


# ═══════════════════════════════════════════════════════════
# GREETING GENERATION
# ═══════════════════════════════════════════════════════════

async def generate_greeting(user_id: str) -> dict[str, Any]:
    """Generate a contextual greeting based on user profile + last session."""
    db = _get_db()

    # ── Profile ──
    profile = None
    if db:
        try:
            doc = db.collection("user_profiles").document(user_id).get()
            if doc.exists:
                profile = doc.to_dict() or {}
        except Exception as e:
            logger.warning(f"Firestore profile read error: {e}")

    if not profile:
        return {
            "greeting": "Welcome to Kalpana! Let's get you set up.",
            "has_context": False,
            "is_first_time": True,
        }

    if not profile.get("onboarding_complete"):
        return {
            "greeting": "Welcome back. Let's finish setting up your profile first.",
            "has_context": False,
            "is_first_time": True,
        }

    name = profile.get("preferred_name", "friend")
    style = profile.get("communication_style", "Like a supportive friend")
    language = profile.get("language", "English")
    age_band = profile.get("age_band", "25-40")

    # ── Last session ──
    last_session_data: Optional[dict] = None
    last_session_id: Optional[str] = None

    if db:
        try:
            from google.cloud import firestore as gcloud_firestore  # type: ignore
            sessions = (
                db.collection("sessions")
                .where("user_id", "==", user_id)
                .order_by("created_at", direction=gcloud_firestore.Query.DESCENDING)
                .limit(1)
                .stream()
            )
            for s in sessions:
                last_session_data = s.to_dict() or {}
                last_session_id = s.id
                break
        except Exception as e:
            logger.warning(f"Could not fetch last session from Firestore: {e}")

    if not last_session_data:
        local = _get_last_local_session(user_id)
        if local:
            last_session_data = local
            last_session_id = local.get("session_id")

    if not last_session_data:
        return {
            "greeting": f"Welcome, {name}. What's on your mind today?",
            "has_context": False,
            "is_first_time": False,
            "name": name,
        }

    last_pillar = (last_session_data.get("pillar") or "").lower()
    last_query = last_session_data.get("query", "")

    # ── Life-context: prefer pre-computed, else extract live ──
    life_context = (last_session_data.get("life_context") or "").strip()
    if not life_context:
        try:
            life_context = await extract_life_context(last_session_data)
        except Exception as e:
            logger.warning(f"Live life_context extraction failed: {e}")
            life_context = ""

    # ── Build greeting via Gemini, weaving in the life context ──
    greeting_text = await _generate_greeting_text(
        name=name,
        style=style,
        language=language,
        age_band=age_band,
        life_context=life_context,
        last_pillar=last_pillar,
        last_query=last_query,
    )

    if not greeting_text:
        # Deterministic fallback
        if life_context:
            greeting_text = (
                f"Welcome back, {name}. Last time, {life_context} "
                f"\u2014 how's that going? Want to pick up where we left off?"
            )
        else:
            greeting_text = (
                f"Welcome back, {name}. Last time we were exploring {last_pillar} "
                f"\u2014 would you like to continue or start something new?"
            )

    return {
        "greeting": greeting_text,
        "has_context": True,
        "is_first_time": False,
        "name": name,
        "last_pillar": last_pillar,
        "last_query": last_query[:200],
        "life_context": life_context,
        "last_session_id": last_session_id,
    }


async def _generate_greeting_text(
    *,
    name: str,
    style: str,
    language: str,
    age_band: str,
    life_context: str,
    last_pillar: str,
    last_query: str,
) -> str:
    """Compose the warm 2-sentence greeting via Gemini."""
    context_line = (
        f"Last life-context to remember: {life_context}"
        if life_context else
        f"Last topic discussed: {last_pillar} \u2014 '{last_query[:120]}'"
    )

    prompt = (
        "Generate a warm 2-sentence greeting for a returning user.\n\n"
        f"Name: {name}\n"
        f"Tone style: {style}\n"
        f"Preferred language: {language}\n"
        f"Age band: {age_band}\n"
        f"{context_line}\n\n"
        "The greeting MUST:\n"
        "1. Address her by name once.\n"
        "2. Reference the specific life-context naturally \u2014 the way a friend would, "
        "not a CRM ('Last time you were getting ready to...', 'When we last talked, "
        "you were planning...').\n"
        "3. Ask gently how it's going OR if she wants to continue.\n\n"
        "Example pattern:\n"
        "\"Welcome back, Priya. Last time you were getting ready to move to Hyderabad "
        "\u2014 how's that going? Did you check out those clinics we found?\"\n\n"
        "Constraints:\n"
        "- Max 220 characters total.\n"
        "- No exclamation marks. No emojis.\n"
        "- If language is Hindi or Hinglish, mix Hindi words naturally.\n"
        "- Do NOT quote the user's question verbatim.\n"
        "- Output ONLY the greeting text, no preamble or quotes."
    )

    return await _call_gemini(prompt, max_tokens=140, temperature=0.7)
