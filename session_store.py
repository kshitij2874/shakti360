"""
session_store.py — Firestore-backed conversation + approval state.

Cloud Run runs multiple instances with no shared memory, so module-level
dicts lose state when a follow-up request lands on a different instance.
This store persists state in Firestore (with an in-memory write-through
cache for speed and a graceful local-only fallback when Firestore is down).

State older than STALE_AFTER_SECONDS is treated as empty so we never resume
an ancient conversation.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("shakti.session_store")

STALE_AFTER_SECONDS = 2 * 3600  # 2 hours

_CONV_COLLECTION = "conversation_state"
_APPROVAL_COLLECTION = "pending_approvals"

# In-memory fallback / write-through cache (per instance)
_local_conv: dict[str, dict[str, Any]] = {}
_local_approvals: dict[str, dict[str, Any]] = {}

_firestore_client = None
_firestore_checked = False


def _get_firestore():
    global _firestore_client, _firestore_checked
    if _firestore_checked:
        return _firestore_client
    _firestore_checked = True
    try:
        from google.cloud import firestore  # type: ignore
        _firestore_client = firestore.Client()
        logger.info("Firestore connected for session_store")
    except Exception as e:
        logger.warning(f"Firestore unavailable for session_store — using local only: {e}")
        _firestore_client = None
    return _firestore_client


def _fresh(payload: Optional[dict]) -> dict[str, Any]:
    """Return the stored state if not stale, else empty dict."""
    if not payload:
        return {}
    updated = payload.get("_updated_at", 0)
    if time.time() - updated > STALE_AFTER_SECONDS:
        return {}
    return {k: v for k, v in payload.items() if k != "_updated_at"}


# ═══════════════════════════════════════════════════════════
# CONVERSATION STATE
# ═══════════════════════════════════════════════════════════

async def get_session_state(session_id: str) -> dict[str, Any]:
    if not session_id:
        return {}
    db = _get_firestore()
    if db:
        try:
            doc = db.collection(_CONV_COLLECTION).document(session_id).get()
            if doc.exists:
                state = _fresh(doc.to_dict())
                _local_conv[session_id] = state  # refresh cache
                return state
            return {}
        except Exception as e:
            logger.warning(f"Firestore conv read failed, using local: {e}")
    return _fresh(_local_conv.get(session_id))


async def set_session_state(session_id: str, state: dict[str, Any]) -> None:
    if not session_id:
        return
    payload = {**state, "_updated_at": time.time()}
    _local_conv[session_id] = state
    db = _get_firestore()
    if db:
        try:
            db.collection(_CONV_COLLECTION).document(session_id).set(payload)
        except Exception as e:
            logger.warning(f"Firestore conv write failed (kept local): {e}")


async def clear_session_state(session_id: str) -> None:
    if not session_id:
        return
    _local_conv.pop(session_id, None)
    db = _get_firestore()
    if db:
        try:
            db.collection(_CONV_COLLECTION).document(session_id).delete()
        except Exception as e:
            logger.warning(f"Firestore conv delete failed: {e}")


# ═══════════════════════════════════════════════════════════
# PENDING TOOL APPROVALS
# ═══════════════════════════════════════════════════════════

async def set_pending_approval(session_id: str, data: dict[str, Any]) -> None:
    if not session_id:
        return
    payload = {**data, "_updated_at": time.time()}
    _local_approvals[session_id] = data
    db = _get_firestore()
    if db:
        try:
            db.collection(_APPROVAL_COLLECTION).document(session_id).set(payload)
        except Exception as e:
            logger.warning(f"Firestore approval write failed (kept local): {e}")


async def pop_pending_approval(session_id: str) -> Optional[dict[str, Any]]:
    """Fetch and remove a pending approval. Returns None if absent/stale."""
    if not session_id:
        return None
    result: Optional[dict[str, Any]] = None
    db = _get_firestore()
    if db:
        try:
            ref = db.collection(_APPROVAL_COLLECTION).document(session_id)
            doc = ref.get()
            if doc.exists:
                fresh = _fresh(doc.to_dict())
                result = fresh or None
                ref.delete()
        except Exception as e:
            logger.warning(f"Firestore approval pop failed, using local: {e}")
    if result is None:
        local = _local_approvals.pop(session_id, None)
        result = _fresh(local) or None
    else:
        _local_approvals.pop(session_id, None)
    return result
