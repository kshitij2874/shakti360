"""
greeting.py — Generate personalized contextual greetings for returning users.
Uses Vertex AI Gemini to craft a warm, persona-aware greeting that references
the user's last conversation.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

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
    """Store session locally for greeting fallback."""
    session_data["user_id"] = user_id
    _local_sessions.append(session_data)
    # Keep only last 50
    if len(_local_sessions) > 50:
        _local_sessions.pop(0)


def _get_last_local_session(user_id: str) -> Optional[dict]:
    """Get last session from local store."""
    user_sessions = [s for s in _local_sessions if s.get("user_id") == user_id]
    if not user_sessions:
        return None
    return sorted(user_sessions, key=lambda x: x.get("created_at", 0), reverse=True)[0]


async def generate_greeting(user_id: str) -> dict:
    """Generate a contextual greeting based on user profile + last session."""
    db = _get_db()

    # ── Fetch user profile ──
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
            "greeting": "Welcome to ShaktiAgent! Let's get you set up.",
            "has_context": False,
            "is_first_time": True,
        }

    if not profile.get("onboarding_complete"):
        return {
            "greeting": "Welcome back! Let's finish setting up your profile first.",
            "has_context": False,
            "is_first_time": True,
        }

    name = profile.get("preferred_name", "friend")
    style = profile.get("communication_style", "Like a supportive friend")
    language = profile.get("language", "English")

    # ── Fetch last session ──
    last_session_data = None
    last_session_id = None

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

    # Fallback to local
    if not last_session_data:
        local = _get_last_local_session(user_id)
        if local:
            last_session_data = local
            last_session_id = local.get("session_id")

    if not last_session_data:
        return {
            "greeting": f"Welcome, {name}! What's on your mind today?",
            "has_context": False,
            "is_first_time": False,
            "name": name,
        }

    last_pillar = (last_session_data.get("pillar") or "").lower()
    last_query = last_session_data.get("query", "")

    # ── Generate contextual greeting via Gemini ──
    greeting_text = ""

    try:
        import vertexai  # type: ignore
        from vertexai.generative_models import GenerativeModel  # type: ignore

        vertexai.init(
            project=os.getenv("PROJECT_ID", "shakti360"),
            location=os.getenv("VERTEX_LOCATION", "us-central1"),
        )
        model = GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))

        prompt = (
            f"Generate a warm, brief greeting (max 2 sentences) for a returning user.\n\n"
            f"Context:\n"
            f"- User's name: {name}\n"
            f"- Tone style: {style}\n"
            f"- Preferred language: {language}\n"
            f"- Last topic discussed: {last_pillar}\n"
            f"- Last query: \"{last_query[:150]}\"\n\n"
            f"Rules:\n"
            f"- Make it feel like a friend remembering, not a CRM\n"
            f"- Gently reference what they last talked about (don't quote verbatim)\n"
            f"- End with: would they like to continue or start fresh?\n"
            f"- NO exclamation marks, NO emojis\n"
            f"- If language is Hindi/Hinglish, naturally mix Hindi words\n"
            f"- Keep under 200 characters\n"
            f"- Respond ONLY with the greeting text, nothing else"
        )

        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 120, "temperature": 0.7},
        )
        greeting_text = (response.text or "").strip().strip('"').strip("'")

    except Exception as e:
        logger.error(f"Gemini greeting generation failed: {e}")

    if not greeting_text:
        greeting_text = (
            f"Welcome back, {name}. Last time we were exploring {last_pillar} "
            f"— would you like to continue or start something new?"
        )

    return {
        "greeting": greeting_text,
        "has_context": True,
        "is_first_time": False,
        "name": name,
        "last_pillar": last_pillar,
        "last_query": last_query[:200],
        "last_session_id": last_session_id,
    }
