"""
onboarding.py — Conversational onboarding for first-time users.
5-turn flow that builds user persona, saves to Firestore user_profiles.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger("shakti.onboarding")

# ── Firestore client (lazy init) ──
_firestore_client = None
_firestore_available = False


def _get_firestore():
    global _firestore_client, _firestore_available
    if _firestore_client is not None:
        return _firestore_client
    try:
        from google.cloud import firestore  # type: ignore
        _firestore_client = firestore.Client()
        _firestore_available = True
        return _firestore_client
    except Exception as e:
        logger.warning(f"Firestore unavailable for onboarding: {e}")
        _firestore_available = False
        return None


# ── Local fallback ──
_local_profiles: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════
# ONBOARDING QUESTIONS
# ═══════════════════════════════════════════════════════════

ONBOARDING_QUESTIONS = [
    {
        "id": "age_band",
        "question": "Hi! I'm Shakti \u2014 your companion for life's big moments. "
                    "To help you better, may I ask which age range you're in?",
        "type": "choice",
        "options": ["11-24", "25-40", "41+"],
    },
    {
        "id": "preferred_name",
        "question": "Lovely to meet you! What should I call you?",
        "type": "text",
    },
    {
        "id": "language",
        "question": "Which language are you most comfortable in?",
        "type": "choice",
        "options": ["English", "Hindi", "Hinglish", "Tamil", "Telugu", "Marathi", "Bengali"],
    },
    {
        "id": "top_focus",
        "question": "What's most on your mind right now?",
        "type": "choice",
        "options": ["Health & Wellness", "Money & Finances", "Career & Growth", "All of these"],
    },
    {
        "id": "communication_style",
        "question": "How do you prefer I talk to you?",
        "type": "choice",
        "options": [
            "Like a wise older sister",
            "Like a professional advisor",
            "Like a supportive friend",
            "Just give me the facts",
        ],
    },
]


# ═══════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════

class OnboardingAnswer(BaseModel):
    question_id: str
    answer: str
    document_uri: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════

async def get_onboarding_state(user_id: str) -> dict:
    """Get current onboarding state for a user."""
    db = _get_firestore()
    if db:
        try:
            doc = db.collection("user_profiles").document(user_id).get()
            if doc.exists:
                data = doc.to_dict()
                return {
                    "complete": data.get("onboarding_complete", False),
                    "step": data.get("onboarding_step", 0),
                    "answers": data.get("onboarding_answers", {}),
                    "profile": data,
                }
        except Exception as e:
            logger.warning(f"Firestore read error: {e}")

    # Local fallback
    profile = _local_profiles.get(user_id)
    if profile:
        return {
            "complete": profile.get("onboarding_complete", False),
            "step": profile.get("onboarding_step", 0),
            "answers": profile.get("onboarding_answers", {}),
            "profile": profile,
        }

    return {"complete": False, "step": 0, "answers": {}, "profile": None}


async def save_onboarding_answer(user_id: str, answer: OnboardingAnswer) -> dict:
    """Save an onboarding answer and return the next question or completion state."""
    state = await get_onboarding_state(user_id)
    answers = state["answers"]
    answers[answer.question_id] = answer.answer

    new_step = state["step"] + 1
    is_complete = new_step >= len(ONBOARDING_QUESTIONS)

    update = {
        "onboarding_answers": answers,
        "onboarding_step": new_step,
        "onboarding_complete": is_complete,
        "updated_at": time.time(),
    }

    # Promote answers to top-level fields for easy access by agents
    field_map = {
        "age_band": "age_band",
        "preferred_name": "preferred_name",
        "language": "language",
        "communication_style": "communication_style",
        "top_focus": "top_focus",
    }
    if answer.question_id in field_map:
        update[field_map[answer.question_id]] = answer.answer

    if is_complete:
        update["onboarding_completed_at"] = time.time()

    # Save to Firestore
    db = _get_firestore()
    if db:
        try:
            from google.cloud import firestore as gcloud_firestore  # type: ignore
            ref = db.collection("user_profiles").document(user_id)
            update["updated_at"] = gcloud_firestore.SERVER_TIMESTAMP
            ref.set(update, merge=True)
            logger.info(f"Onboarding answer saved (Firestore): user={user_id}, q={answer.question_id}")
        except Exception as e:
            logger.warning(f"Firestore save error: {e}")
            # Local fallback
            if user_id not in _local_profiles:
                _local_profiles[user_id] = {}
            _local_profiles[user_id].update(update)
    else:
        # Local fallback
        if user_id not in _local_profiles:
            _local_profiles[user_id] = {}
        _local_profiles[user_id].update(update)

    return {
        "complete": is_complete,
        "next_step": new_step,
        "total_steps": len(ONBOARDING_QUESTIONS),
        "next_question": ONBOARDING_QUESTIONS[new_step] if not is_complete else None,
    }
