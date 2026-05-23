"""
dual_user.py — Dual-user mode framing for Kalpana agent responses.

When a user onboards in "other" mode (e.g. a husband asking about his wife),
every agent response must be framed in the third person ABOUT the subject,
not addressed TO her.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("shakti.dual_user")

# ── Firestore client (lazy) ──
_firestore_client = None


def _get_firestore():
    global _firestore_client
    if _firestore_client is not None:
        return _firestore_client
    try:
        from google.cloud import firestore  # type: ignore
        _firestore_client = firestore.Client()
        return _firestore_client
    except Exception as e:
        logger.warning(f"Firestore unavailable in dual_user: {e}")
        return None


async def get_user_profile(user_id: str) -> dict[str, Any]:
    """Fetch user_profiles doc from Firestore. Returns {} if missing."""
    if not user_id:
        return {}
    db = _get_firestore()
    if not db:
        return {}
    try:
        doc = db.collection("user_profiles").document(user_id).get()
        if doc.exists:
            return doc.to_dict() or {}
    except Exception as e:
        logger.warning(f"Profile fetch failed for {user_id}: {e}")
    return {}


# ── Friendly labels for raw role keys ──
_SUBJECT_ROLE_LABEL = {
    "student": "student",
    "working": "working woman",
    "new_mother": "new mother",
    "homemaker": "homemaker",
    "entrepreneur": "entrepreneur",
    "retired": "woman in mid-life",
    "wife": "wife",
    "daughter": "daughter",
    "mother": "mother",
    "sister": "sister",
    "friend": "friend",
    "colleague": "colleague",
}

_ASKER_ROLE_LABEL = {
    "husband": "husband",
    "father": "father",
    "brother": "brother",
    "son": "son",
    "friend": "friend",
    "other": "someone who cares about her",
    "self": "self",
}


def _label(map_: dict, key: str, fallback: str = "") -> str:
    if not key:
        return fallback
    return map_.get(key, key.replace("_", " ")) or fallback


def build_user_context(profile: dict) -> dict[str, Any]:
    """Build context describing who is asking and who they're asking for."""
    user_mode = (profile or {}).get("user_mode", "self")
    raw_name = (profile or {}).get("preferred_name")
    asker_role_raw = (profile or {}).get("asker_role", "self")
    subject_role_raw = (profile or {}).get("subject_role", "")

    subject_role = _label(_SUBJECT_ROLE_LABEL, subject_role_raw)
    asker_role = _label(_ASKER_ROLE_LABEL, asker_role_raw)

    if user_mode == "self":
        name = raw_name or "friend"
        return {
            "frame": "first_person",
            "user_mode": "self",
            "addressee": name,
            "subject_name": name,
            "subject_role": subject_role,
            "subject_pronoun": "you",
            "subject_possessive": "your",
            "instruction": (
                f"You are talking directly to {name}"
                + (f", a {subject_role}" if subject_role else "")
                + ". Address her warmly by name occasionally. "
                "Use second person — 'you' and 'your' — throughout. "
                "Never refer to her in the third person."
            ),
        }

    # user_mode == "other"
    name = raw_name or "her"
    return {
        "frame": "third_person",
        "user_mode": "other",
        "addressee": asker_role,
        "subject_name": name,
        "subject_role": subject_role,
        "asker_role": asker_role,
        "subject_pronoun": "she",
        "subject_possessive": "her",
        "instruction": (
            f"You are speaking to a {asker_role} who is asking on behalf of {name}"
            + (f", a {subject_role}" if subject_role else "")
            + ". Frame ALL guidance about "
            + f"{name} in the third person — e.g. \"{name} should...\", "
            "\"her doctor...\", \"she may want to...\". "
            f"Never address {name} directly as \"you\". "
            f"Be warm and respectful of the {asker_role}'s care and concern; "
            "briefly acknowledge that asking on her behalf matters. "
            "Where appropriate, suggest a concrete next step the asker can take "
            "(e.g. \"You could share this with her\", \"Encourage her to...\")."
        ),
    }


def build_framing_prefix(user_context: dict) -> str:
    """Build the framing instruction block to prepend to agent system prompts."""
    if not user_context:
        return ""

    extras = ""
    if user_context.get("user_mode") == "other":
        extras = (
            f"\nSubject name: {user_context.get('subject_name', 'her')}"
            f"\nSubject role: {user_context.get('subject_role', 'unknown')}"
            f"\nAsker relationship: {user_context.get('asker_role', 'someone who cares')}"
        )
    elif user_context.get("user_mode") == "self":
        extras = (
            f"\nUser name: {user_context.get('subject_name', 'friend')}"
            f"\nLife stage: {user_context.get('subject_role', 'unknown')}"
        )

    return (
        "\n=== USER FRAMING CONTEXT ===\n"
        f"{user_context['instruction']}"
        f"{extras}\n"
        "CRITICAL: Match this framing throughout your entire response. "
        "Do not mix first-person and third-person.\n"
        "=============================\n"
    )


async def get_framing_for_user(user_id: Optional[str]) -> tuple[dict, str]:
    """Convenience helper: fetch profile and return (context, prefix)."""
    profile = await get_user_profile(user_id) if user_id else {}
    ctx = build_user_context(profile)
    prefix = build_framing_prefix(ctx)
    return ctx, prefix
