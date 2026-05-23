"""
cache.py — Response caching by query hash.
Uses Firestore collection "response_cache" with graceful degradation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger("shakti.cache")

# ── In-memory fallback cache ──
_local_cache: dict[str, dict[str, Any]] = {}

# ── Firestore client (lazy init) ──
_firestore_client = None
_firestore_available = False


def _get_firestore():
    """Lazy-init Firestore client."""
    global _firestore_client, _firestore_available
    if _firestore_client is not None:
        return _firestore_client

    try:
        from google.cloud import firestore  # type: ignore
        _firestore_client = firestore.Client()
        _firestore_available = True
        logger.info("Firestore cache connected")
        return _firestore_client
    except Exception as e:
        logger.warning(f"Firestore unavailable for cache — using local: {e}")
        _firestore_available = False
        return None


def _make_cache_key(query: str, age_band: str, pillar: str) -> str:
    """Generate deterministic cache key from query + age_band + pillar."""
    raw = f"{query.strip().lower()}|{age_band}|{pillar}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


async def get_cached_response(
    query: str,
    age_band: str,
    pillar: str,
    max_age_seconds: int = 3600,
) -> Optional[dict[str, Any]]:
    """
    Retrieve cached response for a query.
    Returns None if not found or expired.
    """
    key = _make_cache_key(query, age_band, pillar)

    # Try Firestore first
    db = _get_firestore()
    if db:
        try:
            doc = db.collection("response_cache").document(key).get()
            if doc.exists:
                data = doc.to_dict()
                age = time.time() - data.get("timestamp", 0)
                if age < max_age_seconds:
                    logger.info(f"Cache HIT (Firestore): {key[:8]}...")
                    return data.get("response")
                else:
                    logger.info(f"Cache EXPIRED (Firestore): {key[:8]}...")
        except Exception as e:
            logger.warning(f"Firestore cache read error: {e}")

    # Fallback to local
    if key in _local_cache:
        data = _local_cache[key]
        age = time.time() - data.get("timestamp", 0)
        if age < max_age_seconds:
            logger.info(f"Cache HIT (local): {key[:8]}...")
            return data.get("response")
        else:
            del _local_cache[key]

    return None


async def set_cached_response(
    query: str,
    age_band: str,
    pillar: str,
    response: dict[str, Any],
) -> None:
    """Store response in cache."""
    key = _make_cache_key(query, age_band, pillar)
    cache_entry = {
        "query": query,
        "age_band": age_band,
        "pillar": pillar,
        "response": response,
        "timestamp": time.time(),
    }

    # Try Firestore
    db = _get_firestore()
    if db:
        try:
            db.collection("response_cache").document(key).set(cache_entry)
            logger.info(f"Cache SET (Firestore): {key[:8]}...")
        except Exception as e:
            logger.warning(f"Firestore cache write error: {e}")

    # Always store locally too
    _local_cache[key] = cache_entry
    logger.info(f"Cache SET (local): {key[:8]}...")


async def invalidate_cache(query: str, age_band: str, pillar: str) -> None:
    """Remove a specific cache entry."""
    key = _make_cache_key(query, age_band, pillar)

    db = _get_firestore()
    if db:
        try:
            db.collection("response_cache").document(key).delete()
        except Exception:
            pass

    _local_cache.pop(key, None)
    logger.info(f"Cache INVALIDATED: {key[:8]}...")
