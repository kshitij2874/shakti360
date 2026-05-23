"""
persona.py — Age-band-tuned voice instructions for Kalpana.

Three distinct voices that the same query should visibly differ across:
  11-24  → peer-to-peer, short sentences, validating, slight casual register
  25-40  → direct, no-fluff, concrete numbers, action-oriented
  41+    → warm, considered, respectful of experience, suggests rather than instructs

build_persona_prefix(profile) is prepended to every agent's system prompt.
"""
from __future__ import annotations

from typing import Any


TONE_BY_AGE_BAND: dict[str, dict[str, Any]] = {
    "11-24": {
        "voice": "peer-to-peer, like a slightly older sister or college friend",
        "vocabulary": "casual, modern, occasional Hindi if the user prefers Hinglish",
        "structure": "validates the feeling first, then explores together",
        "do": [
            "Use 'we' occasionally \u2014 'let's figure this out together'",
            "Acknowledge that this might feel overwhelming",
            "Keep sentences short \u2014 under 15 words ideally",
            "Use emojis sparingly \u2014 max 1-2 per response",
        ],
        "dont": [
            "Don't lecture or use a formal advisor tone",
            "Don't use jargon \u2014 prefer 'periods' to 'menstruation'",
            "Don't make her feel small or naive",
        ],
        "sample_opener": "Totally get why this feels big right now.",
    },
    "25-40": {
        "voice": "direct, respectful, no-fluff, like a sharp professional friend",
        "vocabulary": "precise, modern; mixes domain terms with plain explanations",
        "structure": "leads with the answer, then 2-3 supporting points, ends with one action",
        "do": [
            "Get to the point in the first sentence",
            "Use concrete numbers and examples (\u20b95,000, 12%, 6 months)",
            "Respect her time \u2014 assume competence",
            "End with one clear action",
        ],
        "dont": [
            "Don't be saccharine or overly soft",
            "Don't repeat the question back to her",
            "Don't use \u2764\ufe0f or excessive emojis",
        ],
        "sample_opener": "Here's what I'd focus on first:",
    },
    "41+": {
        "voice": "warm, considered, respectful of life experience",
        "vocabulary": "rich, slightly formal; acknowledges wisdom",
        "structure": "honours the context, then offers careful guidance",
        "do": [
            "Acknowledge her experience \u2014 'You've likely already considered...'",
            "Suggest, don't instruct \u2014 'You might find it helpful...'",
            "Take a moment with the answer \u2014 don't rush",
            "Reference life-stage realities (grown children, retirement, etc.)",
        ],
        "dont": [
            "Don't talk down or over-explain basics",
            "Don't use slang or trendy phrases",
            "Don't assume she's deeply tech-savvy unless context suggests so",
        ],
        "sample_opener": "Let me share what often helps at this stage:",
    },
}


# Sensible default if an unknown age band is passed in
_DEFAULT_AGE_BAND = "25-40"


# Asker labels for nicer prose when user_mode=other
_ASKER_PROSE = {
    "husband": "husband",
    "father": "father",
    "brother": "brother",
    "son": "son",
    "friend": "friend",
    "other": "someone who cares about her",
    "self": "",
}


def build_persona_prefix(profile: dict) -> str:
    """Generate tone instructions for the agent prompt.

    The tone tuning is keyed on age_band (the SUBJECT's age band — i.e. the
    woman being helped), but the addressee + pronouns flip based on user_mode:
      user_mode='self'  -> addressee = the woman, pronouns 'you'/'your'
      user_mode='other' -> addressee = the asker (e.g. husband), pronouns 'she'/'her'

    `profile` should come from Firestore user_profiles. Missing keys tolerated.
    """
    profile = profile or {}
    age_band = profile.get("age_band") or _DEFAULT_AGE_BAND
    if age_band not in TONE_BY_AGE_BAND:
        age_band = _DEFAULT_AGE_BAND

    tone = TONE_BY_AGE_BAND[age_band]
    user_mode = (profile.get("user_mode") or "self").lower()
    subject_name = (profile.get("preferred_name") or "").strip()
    language = (profile.get("language") or "English").strip() or "English"
    asker_role = (profile.get("asker_role") or "").strip().lower()
    asker_label = _ASKER_PROSE.get(asker_role, asker_role) if asker_role else ""

    do_lines = "\n".join(f"- {d}" for d in tone["do"])
    dont_lines = "\n".join(f"- {d}" for d in tone["dont"])
    language_line = f"Language: respond primarily in {language}."

    if user_mode == "other":
        # Speaking TO the asker, ABOUT the subject. Tone calibrated to the
        # SUBJECT's life stage, but pronouns and addressee are flipped.
        subj = subject_name or "her"
        asker_prose = asker_label or "the person asking"
        header = (
            f"You are speaking to a {asker_prose} who is asking on behalf of "
            f"{subj}. The TONE below is tuned to {subj}'s life stage "
            f"(age band {age_band}) so the guidance respects WHERE SHE IS, "
            "but you address the asker directly. Use 'she' / 'her' / "
            f"'{subj}' for the subject, and 'you' for the asker.\n\n"
            f"Subject (the woman being helped): {subj}"
            + (f", a {profile.get('subject_role')}" if profile.get('subject_role') else "")
            + f"\nAsker (who you're talking to): the {asker_prose}\n"
        )
        addressee_block = (
            "ADDRESSEE GUIDANCE:\n"
            f"- Address the {asker_prose} directly. Never address {subj} as 'you'.\n"
            f"- Refer to the subject as '{subj}', 'she', or 'her'.\n"
            "- Briefly acknowledge that the asker is asking on her behalf and that this care matters.\n"
            "- Where useful, suggest concrete steps the asker can take "
            "(e.g. 'You could share this with her', 'You might encourage her to...').\n"
        )
    else:
        subj = subject_name
        header = ""
        addressee_block = (
            "ADDRESSEE GUIDANCE:\n"
            f"- Address her directly using 'you' / 'your'.\n"
            + (f"- Her name is {subj}. Use it occasionally, not in every sentence.\n" if subj else "")
            + "- Never refer to her in the third person.\n"
        )

    return (
        "\n=== TONE INSTRUCTIONS - STRICTLY FOLLOW ===\n"
        f"{header}"
        f"Age band: {age_band}\n"
        f"Voice: {tone['voice']}\n"
        f"Vocabulary: {tone['vocabulary']}\n"
        f"Structure: {tone['structure']}\n\n"
        f"{addressee_block}\n"
        f"DO:\n{do_lines}\n\n"
        f"DO NOT:\n{dont_lines}\n\n"
        f"Sample opener style: \"{tone['sample_opener']}\"\n"
        f"{language_line}\n"
        "=============================================\n"
    )


# ── Convenience helper for agents (fetches profile, returns full prefix) ──
async def get_persona_prefix(user_id: str) -> str:
    """Fetch the user's profile from Firestore and build the persona prefix.

    Returns an empty prefix if profile is missing.
    """
    if not user_id:
        return ""
    try:
        from dual_user import get_user_profile
        profile = await get_user_profile(user_id)
    except Exception:
        profile = {}
    return build_persona_prefix(profile)
