"""auth.py — Firebase Auth middleware for ShaktiAgent."""
import logging
from fastapi import Header, HTTPException

import firebase_admin
from firebase_admin import auth, credentials

logger = logging.getLogger("shakti.auth")

_initialized = False


def _ensure_init():
    """Lazy initialize Firebase Admin once."""
    global _initialized
    if _initialized:
        return
    try:
        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        _initialized = True
        logger.info("Firebase Admin initialized")
    except ValueError:
        _initialized = True
    except Exception as e:
        logger.warning(f"Firebase Admin init failed: {e}")


async def verify_token(authorization: str = Header(None)) -> dict:
    """Verify Firebase ID token, return decoded user info."""
    _ensure_init()

    if not authorization:
        raise HTTPException(401, "Missing Authorization header")

    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(401, "Empty token")

    try:
        decoded = auth.verify_id_token(token)
        return {
            "uid": decoded["uid"],
            "email": decoded.get("email"),
            "name": decoded.get("name"),
            "picture": decoded.get("picture"),
        }
    except Exception as e:
        logger.error(f"Token verify failed: {e}")
        raise HTTPException(401, f"Invalid token: {e}")
