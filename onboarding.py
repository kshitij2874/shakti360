"""
onboarding.py — Conversational chat-style onboarding for Kalpana.
Dual-path: user can onboard for themselves OR for someone they care about.
Branching based on previous answers via show_if conditionals.
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
# ONBOARDING FLOW — Conversational, with branching
# ═══════════════════════════════════════════════════════════

ONBOARDING_FLOW = [
    {
        "id": "user_mode",
        "kalpana_says": "Hi, I'm Kalpana. Here for you. \U0001F338\n\n"
                        "First, tell me \u2014 are you here for yourself, or someone you care about?",
        "type": "choice",
        "options": [
            {"value": "self", "label": "For myself"},
            {"value": "other", "label": "For someone I care about"},
        ],
    },
    {
        "id": "role_self",
        "kalpana_says": "Lovely. Which of these feels closest to you right now?",
        "type": "choice",
        "show_if": {"user_mode": "self"},
        "options": [
            {"value": "student", "label": "Student"},
            {"value": "working", "label": "Working woman"},
            {"value": "new_mother", "label": "New mother"},
            {"value": "homemaker", "label": "Homemaker"},
            {"value": "entrepreneur", "label": "Entrepreneur"},
            {"value": "retired", "label": "Retired / mid-life"},
        ],
    },
    {
        "id": "role_other",
        "kalpana_says": "That's beautiful. Who are you here for?",
        "type": "choice",
        "show_if": {"user_mode": "other"},
        "options": [
            {"value": "wife", "label": "My wife"},
            {"value": "daughter", "label": "My daughter"},
            {"value": "mother", "label": "My mother"},
            {"value": "sister", "label": "My sister"},
            {"value": "friend", "label": "A friend"},
            {"value": "colleague", "label": "A colleague"},
        ],
    },
    {
        "id": "relation_self",
        "kalpana_says": "And how do you describe yourself?",
        "type": "choice",
        "show_if": {"user_mode": "other"},
        "options": [
            {"value": "husband", "label": "Husband"},
            {"value": "father", "label": "Father"},
            {"value": "brother", "label": "Brother"},
            {"value": "son", "label": "Son"},
            {"value": "friend", "label": "Friend"},
            {"value": "other", "label": "Someone who cares"},
        ],
    },
    {
        "id": "age_band",
        "kalpana_says_self": "Which age group are you in? I'll adjust how I talk based on this.",
        "kalpana_says_other": "Which age group is she in? I'll tune my guidance for her life stage.",
        "type": "choice",
        "options": [
            {"value": "11-24", "label": "11 \u2013 24 \u00b7 Young adult"},
            {"value": "25-40", "label": "25 \u2013 40 \u00b7 Working age"},
            {"value": "41+", "label": "41+ \u00b7 Mid-life & beyond"},
        ],
    },
    {
        "id": "preferred_name",
        "kalpana_says_self": "What should I call you?",
        "kalpana_says_other": "And what's her name? (or what should I call her in our conversations)",
        "type": "text",
        "placeholder": "Just a first name is fine",
    },
    {
        "id": "language",
        "kalpana_says": "Which language are you most comfortable in?",
        "type": "choice",
        "options": [
            {"value": "English", "label": "English"},
            {"value": "Hindi", "label": "\u0939\u093f\u0902\u0926\u0940"},
            {"value": "Hinglish", "label": "Hinglish"},
            {"value": "Tamil", "label": "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd"},
            {"value": "Telugu", "label": "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41"},
            {"value": "Marathi", "label": "\u092e\u0930\u093e\u0920\u0940"},
            {"value": "Bengali", "label": "\u09ac\u09be\u0982\u09b2\u09be"},
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
# FLOW LOGIC — Visibility filtering and next-question lookup
# ═══════════════════════════════════════════════════════════

def get_visible_questions(answers: dict) -> list[dict]:
    """Filter the flow based on conditional show_if rules."""
    visible = []
    for q in ONBOARDING_FLOW:
        show_if = q.get("show_if")
        if show_if:
            if all(answers.get(k) == v for k, v in show_if.items()):
                visible.append(q)
        else:
            visible.append(q)
    return visible


def _resolve_kalpana_says(q: dict, answers: dict) -> dict:
    """Pick the right kalpana_says variant based on user_mode."""
    if "kalpana_says_self" in q or "kalpana_says_other" in q:
        user_mode = answers.get("user_mode", "self")
        resolved = {k: v for k, v in q.items() if not k.startswith("kalpana_says_")}
        resolved["kalpana_says"] = q.get(
            f"kalpana_says_{user_mode}",
            q.get("kalpana_says_self", "")
        )
        return resolved
    return q


def get_current_question(answers: dict) -> Optional[dict]:
    """Return the next unanswered visible question, with variant text resolved."""
    visible = get_visible_questions(answers)
    for q in visible:
        if q["id"] not in answers:
            return _resolve_kalpana_says(q, answers)
    return None  # Complete


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
                data = doc.to_dict() or {}
                return {
                    "complete": data.get("onboarding_complete", False),
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
            "answers": profile.get("onboarding_answers", {}),
            "profile": profile,
        }

    return {"complete": False, "answers": {}, "profile": {}}


async def save_onboarding_answer(user_id: str, answer: OnboardingAnswer) -> dict:
    """Save an answer and return the next question (or completion)."""
    state = await get_onboarding_state(user_id)
    answers = state["answers"]
    answers[answer.question_id] = answer.answer

    # Determine if onboarding is now complete based on the new answers
    next_q = get_current_question(answers)
    is_complete = next_q is None

    # Build promoted fields for easy access by agents
    promoted: dict = {
        "onboarding_answers": answers,
        "onboarding_complete": is_complete,
        "updated_at": time.time(),
    }

    # Promote key fields to top level
    for key in ("user_mode", "age_band", "preferred_name", "language"):
        if key in answers:
            promoted[key] = answers[key]

    # Determine subject (who Kalpana is helping) and asker role
    if answers.get("user_mode") == "self":
        if "role_self" in answers:
            promoted["subject_role"] = answers["role_self"]
        promoted["asker_role"] = "self"
    elif answers.get("user_mode") == "other":
        if "role_other" in answers:
            promoted["subject_role"] = answers["role_other"]
        if "relation_self" in answers:
            promoted["asker_role"] = answers["relation_self"]

    if is_complete:
        promoted["onboarding_completed_at"] = time.time()

    # Save to Firestore
    db = _get_firestore()
    if db:
        try:
            from google.cloud import firestore as gcloud_firestore  # type: ignore
            ref = db.collection("user_profiles").document(user_id)
            promoted["updated_at"] = gcloud_firestore.SERVER_TIMESTAMP
            ref.set(promoted, merge=True)
            logger.info(f"Onboarding answer saved (Firestore): user={user_id}, q={answer.question_id}, complete={is_complete}")
        except Exception as e:
            logger.warning(f"Firestore save error: {e}")
            # Local fallback
            if user_id not in _local_profiles:
                _local_profiles[user_id] = {}
            _local_profiles[user_id].update(promoted)
    else:
        if user_id not in _local_profiles:
            _local_profiles[user_id] = {}
        _local_profiles[user_id].update(promoted)

    return {
        "complete": is_complete,
        "next_question": next_q,
        "answers_so_far": answers,
    }


# ═══════════════════════════════════════════════════════════
# BACKWARDS COMPAT — legacy import shim
# ═══════════════════════════════════════════════════════════

# Old code in main.py imports ONBOARDING_QUESTIONS — keep alias
ONBOARDING_QUESTIONS = ONBOARDING_FLOW
