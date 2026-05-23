"""
memory.py — 3-tier memory layer (Tencent Agent Memory pattern).
  short_term: in-session only (Python dict in request scope)
  episodic:   Firestore collection "user_memory_episodic"
  semantic:   Firestore vector collection "user_memory_semantic"
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Optional

from observability import observe

logger = logging.getLogger("shakti.memory")

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
        logger.info("Firestore memory layer connected")
        return _firestore_client
    except Exception as e:
        logger.warning(f"Firestore unavailable for memory — using local: {e}")
        _firestore_available = False
        return None


# ── Local fallback storage ──
_local_episodic: dict[str, list[dict[str, Any]]] = {}
_local_semantic: dict[str, list[dict[str, Any]]] = {}


# ═══════════════════════════════════════════════════════════
# SHORT-TERM MEMORY (in-session only)
# ═══════════════════════════════════════════════════════════

class ShortTermMemory:
    """In-session memory — lives only during the request lifecycle."""

    def __init__(self):
        self._store: dict[str, Any] = {}
        self.conversation: list[dict[str, str]] = []

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def add_turn(self, role: str, content: str) -> None:
        self.conversation.append({"role": role, "content": content})

    def get_conversation(self) -> list[dict[str, str]]:
        return self.conversation

    def clear(self) -> None:
        self._store.clear()
        self.conversation.clear()


# ═══════════════════════════════════════════════════════════
# EPISODIC MEMORY (life events, per user)
# ═══════════════════════════════════════════════════════════

@observe("memory_episodic_save")
async def save_episodic_memory(
    user_id: str,
    event_type: str,
    content: str,
) -> str:
    """Save an episodic memory (life event, preference, concern)."""
    memory_id = str(uuid.uuid4())[:8]
    entry = {
        "id": memory_id,
        "user_id": user_id,
        "event_type": event_type,
        "content": content,
        "timestamp": time.time(),
    }

    db = _get_firestore()
    if db:
        try:
            db.collection("user_memory_episodic").document(memory_id).set(entry)
            logger.info(f"Episodic memory saved (Firestore): {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Firestore episodic save error: {e}")

    # Local fallback
    if user_id not in _local_episodic:
        _local_episodic[user_id] = []
    _local_episodic[user_id].append(entry)
    logger.info(f"Episodic memory saved (local): {memory_id}")
    return memory_id


async def get_episodic_memories(
    user_id: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Retrieve recent episodic memories for a user."""
    db = _get_firestore()
    if db:
        try:
            docs = (
                db.collection("user_memory_episodic")
                .where("user_id", "==", user_id)
                .order_by("timestamp", direction="DESCENDING")
                .limit(limit)
                .stream()
            )
            return [doc.to_dict() for doc in docs]
        except Exception as e:
            logger.warning(f"Firestore episodic read error: {e}")

    # Local fallback
    entries = _local_episodic.get(user_id, [])
    return sorted(entries, key=lambda x: x["timestamp"], reverse=True)[:limit]


# ═══════════════════════════════════════════════════════════
# SEMANTIC MEMORY (learnings + embeddings, per user)
# ═══════════════════════════════════════════════════════════

async def _get_embedding(text: str) -> list[float]:
    """Generate embedding using Vertex AI or return mock."""
    try:
        from google.cloud import aiplatform  # type: ignore
        from vertexai.language_models import TextEmbeddingModel  # type: ignore
        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    except Exception as e:
        logger.warning(f"Embedding generation failed, using mock: {e}")
        import hashlib
        h = hashlib.sha256(text.encode()).hexdigest()
        # Generate deterministic mock embedding
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]


@observe("memory_semantic_save")
async def save_semantic_memory(
    user_id: str,
    learning: str,
) -> str:
    """Save a semantic memory with embedding."""
    memory_id = str(uuid.uuid4())[:8]
    embedding = await _get_embedding(learning)

    entry = {
        "id": memory_id,
        "user_id": user_id,
        "learning": learning,
        "embedding": embedding,
        "timestamp": time.time(),
    }

    db = _get_firestore()
    if db:
        try:
            db.collection("user_memory_semantic").document(memory_id).set(entry)
            logger.info(f"Semantic memory saved (Firestore): {memory_id}")
            return memory_id
        except Exception as e:
            logger.warning(f"Firestore semantic save error: {e}")

    # Local fallback
    if user_id not in _local_semantic:
        _local_semantic[user_id] = []
    _local_semantic[user_id].append(entry)
    logger.info(f"Semantic memory saved (local): {memory_id}")
    return memory_id


async def get_relevant_memories(
    user_id: str,
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve top-k semantically relevant memories for a user."""
    query_embedding = await _get_embedding(query)

    def cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # Try Firestore
    db = _get_firestore()
    memories = []
    if db:
        try:
            docs = (
                db.collection("user_memory_semantic")
                .where("user_id", "==", user_id)
                .stream()
            )
            for doc in docs:
                data = doc.to_dict()
                if "embedding" in data:
                    sim = cosine_sim(query_embedding, data["embedding"])
                    memories.append({**data, "similarity": sim})
        except Exception as e:
            logger.warning(f"Firestore semantic read error: {e}")

    # Local fallback
    if not memories:
        for entry in _local_semantic.get(user_id, []):
            if "embedding" in entry:
                sim = cosine_sim(query_embedding, entry["embedding"])
                memories.append({**entry, "similarity": sim})

    # Sort by similarity, return top-k
    memories.sort(key=lambda x: x.get("similarity", 0), reverse=True)
    return memories[:top_k]


# ═══════════════════════════════════════════════════════════
# MEMORY EXTRACTOR (async post-session)
# ═══════════════════════════════════════════════════════════

@observe("memory_extract")
async def extract_and_save_memories(
    user_id: str,
    conversation: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Post-session: extract life events, preferences, and concerns
    from conversation and save to episodic + semantic memory.
    Uses gemini-2.5-flash for extraction.
    """
    if not conversation:
        return {"extracted": 0}

    conv_text = "\n".join(f"{t['role']}: {t['content']}" for t in conversation)

    extraction_prompt = (
        "Analyze this conversation and extract:\n"
        "1. Life events (pregnancy, job change, health issue, etc.)\n"
        "2. Preferences (communication style, topics of interest)\n"
        "3. Recurring concerns\n\n"
        "Return as JSON: {\"events\": [...], \"preferences\": [...], \"concerns\": [...]}\n"
        "Each item should be a short descriptive string.\n"
        "If nothing notable, return empty arrays.\n\n"
        f"Conversation:\n{conv_text}"
    )

    extracted: dict[str, list[str]] = {"events": [], "preferences": [], "concerns": []}

    try:
        import vertexai  # type: ignore
        from vertexai.generative_models import GenerativeModel  # type: ignore
        import json

        model = GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        response = model.generate_content(
            extraction_prompt,
            generation_config={"max_output_tokens": 300, "temperature": 0.1},
        )
        raw = response.text.strip()
        # Parse JSON from response
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        extracted = json.loads(raw)
    except Exception as e:
        logger.warning(f"LLM memory extraction failed — using keyword fallback: {e}")
        # Simple keyword extraction fallback
        for turn in conversation:
            content = turn.get("content", "").lower()
            if any(w in content for w in ["pregnant", "baby", "period", "health"]):
                extracted["events"].append(f"Health topic discussed: {content[:80]}")
            if any(w in content for w in ["invest", "save", "loan", "money"]):
                extracted["events"].append(f"Finance topic discussed: {content[:80]}")
            if any(w in content for w in ["job", "career", "scholarship", "course"]):
                extracted["events"].append(f"Career topic discussed: {content[:80]}")

    # Save to memory stores
    saved_count = 0
    for event in extracted.get("events", [])[:3]:
        await save_episodic_memory(user_id, "life_event", event)
        await save_semantic_memory(user_id, event)
        saved_count += 1

    for pref in extracted.get("preferences", [])[:2]:
        await save_episodic_memory(user_id, "preference", pref)
        saved_count += 1

    for concern in extracted.get("concerns", [])[:2]:
        await save_episodic_memory(user_id, "concern", concern)
        await save_semantic_memory(user_id, concern)
        saved_count += 1

    logger.info(f"Memory extraction complete: {saved_count} items saved for user {user_id}")
    return {"extracted": saved_count, "details": extracted}


async def get_user_memory_context(
    user_id: str,
    query: str,
) -> str:
    """Build memory context string for the orchestrator to pass to sub-agents."""
    episodic = await get_episodic_memories(user_id, limit=3)
    semantic = await get_relevant_memories(user_id, query, top_k=3)

    parts = []
    if episodic:
        parts.append("Recent context about this user:")
        for mem in episodic:
            parts.append(f"- [{mem.get('event_type', 'info')}] {mem.get('content', '')}")

    if semantic:
        parts.append("\nRelevant past topics:")
        for mem in semantic:
            parts.append(f"- {mem.get('learning', '')} (relevance: {mem.get('similarity', 0):.2f})")

    return "\n".join(parts) if parts else "No prior context available for this user."
