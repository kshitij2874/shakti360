from fastapi import Header, HTTPException, Depends
from firebase_admin import auth, credentials, initialize_app
from firebase_admin._apps import _apps as firebase_apps
import logging

logger = logging.getLogger("shakti.auth")

# Initialize Firebase Admin (uses default service account on Cloud Run)
if not firebase_apps:
    try:
        initialize_app()
    except Exception as e:
        logger.warning(f"Firebase Admin init: {e}")

async def verify_token(authorization: str = Header(None)) -> dict:
    """Verify Firebase ID token, return decoded user info."""
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    
    token = authorization.replace("Bearer ", "").strip()
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
