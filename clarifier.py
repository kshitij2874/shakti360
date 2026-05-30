"""
clarifier.py — Generates pillar + age-specific clarifying questions
that Kalpana asks BEFORE producing a final answer.

Each turn of clarifying yields a single question. The orchestrator collects
answers across turns and then synthesises the final response.

filter_already_answered() uses the LLM to drop templated questions whose
answers are already obvious from the user's original query / profile, so
Kalpana doesn't ask 'do you have any conditions?' when the user just said
'I have PCOS and am planning a pregnancy'.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("shakti.clarifier")

# ── Templates keyed by (pillar, age_band) ──
CLARIFYING_QUESTION_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "HEALTH": {
        "11-24": [
            "Do you take any medicines regularly or have any health conditions? If yes, which ones?",
            "Have you spoken to a doctor about this \u2014 and if so, what did they say?",
            "How long has this been bothering you \u2014 is it new or ongoing?",
        ],
        "25-40": [
            "Do you have any ongoing health conditions or take medicines regularly? If yes, please name them.",
            "Have you seen a doctor for this? Any recent reports or test results?",
            "Which city are you in? I can suggest nearby options.",
        ],
        "41+": [
            "Do you have any ongoing conditions like BP, diabetes, or thyroid \u2014 and if so, what medicines are you taking?",
            "Are you currently seeing a doctor for anything?",
            "Any recent reports or prescriptions to share? (You can skip this for now.)",
        ],
    },
    "FINANCE": {
        "11-24": [
            "Do you earn or save anything regularly \u2014 pocket money, part-time work? If yes, roughly how much?",
            "Do you already have a bank account?",
            "What's your main goal \u2014 college fees, gadget, emergency savings, or something else?",
        ],
        "25-40": [
            "What's your monthly income range roughly? (just for context)",
            "Do you have any existing investments \u2014 SIP, FD, EPF, or others? If yes, what type?",
            "What's your main goal \u2014 home, child's education, retirement, or something else?",
        ],
        "41+": [
            "Do you have any existing savings or investments \u2014 FD, mutual funds, pension? If yes, what type?",
            "Are you focused more on preserving what you have or growing it?",
            "Any dependents you're planning for \u2014 children, parents?",
        ],
    },
    "CAREER": {
        "11-24": [
            "Are you in school, college, or just graduated?",
            "Any field you're drawn to, or still exploring?",
            "Preparing for any entrance exam, scholarship, or certification?",
        ],
        "25-40": [
            "Are you currently working, on a break, or looking for work?",
            "How many years of work experience do you have?",
            "Are you looking to switch fields, upskill, or restart after a break?",
        ],
        "41+": [
            "What's your current or most recent role?",
            "Are you thinking of switching fields, consulting independently, or starting something new?",
            "Any specific skills or industries you're drawn to?",
        ],
    },
}


# ── Yes/No detection — gives the UI a hint to show quick-reply pills ──
_YES_NO_HINTS = (
    "do you ", "have you ", "are you ", "is this ", "any ", "do they "
)


def is_yes_no(question: str) -> bool:
    """Best-effort: does this clarifying question accept a yes/no answer?"""
    q = (question or "").lower().strip()
    if not q:
        return False
    # Multi-part or "what/which/how" questions are not yes/no
    if any(w in q for w in ("which ", "what ", "how ", "where ")):
        return False
    return any(q.startswith(h) for h in _YES_NO_HINTS)


# ── Doc-request detection — UI shows an upload affordance instead of yes/no pills ──
_DOC_PATTERNS = (
    "report", "reports", "prescription", "prescriptions", "scan", "scans",
    "x-ray", "xray", "mri", "ultrasound", "lab result", "blood test",
    "bank statement", "salary slip", "payslip", "itr", "tax return", "resume",
    "cv", "marksheet", "certificate", "document", "documents", "share",
)


def is_doc_request(question: str) -> bool:
    """Does this clarifying question explicitly ask the user to share a document?"""
    q = (question or "").lower()
    if not q:
        return False
    if "share" in q and any(d in q for d in _DOC_PATTERNS):
        return True
    if "upload" in q:
        return True
    if any(p in q for p in ("recent report", "any report", "any prescription", "recent prescription",
                            "any document", "any scan", "lab result", "blood test")):
        return True
    return False


def get_clarifying_questions(pillar: str, age_band: str, query: str = "") -> list[str]:
    """Return up to 3 clarifying questions for this pillar + age band."""
    pillar = (pillar or "").upper()
    questions = CLARIFYING_QUESTION_TEMPLATES.get(pillar, {}).get(age_band, [])
    return list(questions[:3])


def build_clarification_context(
    original_query: str,
    qa_pairs: list[dict[str, str]],
) -> str:
    """Format the original query + clarifying Q/As into a context block for the LLM."""
    lines = [f"Original question: {original_query}"]
    if qa_pairs:
        lines.append("\nClarifying details from the user:")
        for qa in qa_pairs:
            q = (qa.get("q") or "").strip()
            a = (qa.get("a") or "").strip()
            if not a or a.lower() in ("skip", "skipped", "-"):
                continue
            lines.append(f"  Q: {q}")
            lines.append(f"  A: {a}")
    return "\n".join(lines)


async def generate_contextual_questions(
    pillar: str,
    age_band: str,
    query: str,
    profile: Optional[dict] = None,
) -> list[str]:
    """
    Use the LLM to generate 1 precise clarifying question based on what the user actually said.
    Acknowledges their specific situation — feels like a friend asking, not a bot checklist.
    Falls back to a single filtered template on failure.
    """
    profile = profile or {}
    language = profile.get("language", "English")
    profile_hint = ""
    if profile:
        bits = []
        if profile.get("preferred_name"):
            bits.append(f"User's name: {profile['preferred_name']}")
        if profile.get("subject_role"):
            bits.append(f"Life stage: {profile['subject_role']}")
        if bits:
            profile_hint = "\n".join(bits) + "\n"

    prompt = (
        "You are Kalpana, a warm and direct women's life companion.\n"
        f"A user just wrote: \"{query}\"\n\n"
        f"Topic: {pillar} | Age group: {age_band}\n"
        f"{profile_hint}\n"
        "What is the ONE most important thing still missing from their message that would "
        "genuinely change your advice?\n\n"
        "Rules:\n"
        "- Write exactly 1 question (2 only if a second is truly critical)\n"
        "- Do NOT ask about anything the user already mentioned\n"
        "- Acknowledge something specific from their message — show you read it\n"
        "- Sound like a caring, direct friend — not a medical form or checklist\n"
        "- Keep it under 25 words\n"
        f"- Write the question(s) in {language} (keep proper nouns as-is)\n"
        "- If the user gave enough context to answer fully, output: SKIP\n\n"
        "Output only the question(s), one per line. No preamble, no numbering."
    )

    try:
        raw = await _call_gemini(prompt, max_tokens=80, temperature=0.4)
        raw = (raw or "").strip()

        if not raw or raw.upper().startswith("SKIP"):
            return []

        questions = [
            q.strip().lstrip("1234567890.-) ").strip()
            for q in raw.split("\n")
            if q.strip() and len(q.strip()) > 8
        ]
        valid = [q for q in questions if q and not q.upper().startswith("SKIP")]
        return valid[:2]

    except Exception as e:
        logger.warning(f"Contextual question generation failed: {e}")
        templates = get_clarifying_questions(pillar, age_band, query)
        return templates[:1]


async def _call_gemini(prompt: str, max_tokens: int = 120, temperature: float = 0.2) -> str:
    """Delegate to the unified LLM client (DeepSeek → Gemini fallback)."""
    try:
        from llm import call_llm
        return await call_llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        logger.warning(f"Clarifier LLM call failed: {e}")
        return ""


async def filter_already_answered(
    questions: list[str],
    original_query: str,
    profile: Optional[dict] = None,
) -> list[str]:
    """Drop templated questions whose answers are already obvious from the
    user's original query or profile.

    Returns a list of questions to actually ask. Always returns at least 1
    question when there's something useful left; returns an empty list when
    the user has essentially given full context already.
    On any failure, returns the full input list unchanged.
    """
    if not questions:
        return []
    query = (original_query or "").strip()
    if not query:
        return list(questions)

    profile = profile or {}
    profile_hint = ""
    if profile:
        bits = []
        if profile.get("preferred_name"):
            bits.append(f"name={profile['preferred_name']}")
        if profile.get("age_band"):
            bits.append(f"age_band={profile['age_band']}")
        if profile.get("subject_role"):
            bits.append(f"life_stage={profile['subject_role']}")
        if profile.get("language"):
            bits.append(f"language={profile['language']}")
        if bits:
            profile_hint = "Known profile: " + ", ".join(bits) + "\n"

    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions))
    prompt = (
        "You are helping a women's-life-companion AI decide which clarifying "
        "questions are actually still useful to ask.\n\n"
        f"{profile_hint}"
        f"User's original message: \"{query}\"\n\n"
        "Candidate clarifying questions:\n"
        f"{numbered}\n\n"
        "Task: return a JSON array of the INDICES of questions that are STILL "
        "USEFUL to ask \u2014 i.e. the user has NOT already answered them in "
        "their original message and the profile does not already imply the "
        "answer. Drop questions that would feel redundant or annoying.\n\n"
        "Rules:\n"
        "- Output ONLY a JSON array of integers, e.g. [0, 2]\n"
        "- Empty array [] if every question is already answered\n"
        "- Keep at most 3 indices\n"
        "- Preserve the original order"
    )

    raw = await _call_gemini(prompt, max_tokens=60, temperature=0.0)
    if not raw:
        return list(questions[:3])

    # Strip code fences and find the JSON array
    text = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return list(questions[:3])

    try:
        indices = json.loads(text[start:end + 1])
    except Exception:
        return list(questions[:3])

    if not isinstance(indices, list):
        return list(questions[:3])

    out = []
    seen: set[int] = set()
    for idx in indices:
        try:
            i = int(idx)
        except Exception:
            continue
        if 0 <= i < len(questions) and i not in seen:
            seen.add(i)
            out.append(questions[i])
        if len(out) >= 3:
            break

    return out


def opening_kalpana_line(pillar: str, name: str = "") -> str:
    """A short warm opener Kalpana says before asking the first clarifying question."""
    pillar = (pillar or "").upper()
    addr = f" {name}" if name else ""
    if pillar == "HEALTH":
        return (
            f"I'm here for you{addr}. Let's take this one step at a time \u2014 "
            "I'll ask a couple of quick questions so I can guide you better."
        )
    if pillar == "FINANCE":
        return (
            f"Got it{addr}. Before I give you any numbers, let me understand your situation "
            "with a couple of quick questions."
        )
    if pillar == "CAREER":
        return (
            f"Lovely{addr}. Let me ask a couple of short questions so I can tailor "
            "this to where you actually are."
        )
    return (
        f"I'm here for you{addr}. Let me ask a couple of quick questions first."
    )
