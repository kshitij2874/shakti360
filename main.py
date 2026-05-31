"""
main.py — FastAPI app for ShaktiAgent.
Endpoints: /chat, /approve, /reject, /sessions/{user_id}, /metrics, /health,
           /onboarding/state, /onboarding/answer, /greeting
Serves index.html frontend.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from auth import verify_token

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("shakti.main")

# ── Lazy imports (avoid import-time GCP failures) ──
from observability import metrics
from tools import execute_tool
from onboarding import (
    OnboardingAnswer,
    get_onboarding_state,
    save_onboarding_answer,
    get_current_question,
    get_visible_questions,
    reset_onboarding,
    ONBOARDING_FLOW,
)
from greeting import generate_greeting


# ── Pending tool calls store (Firestore-backed for human oversight gate) ──
from session_store import set_pending_approval, pop_pending_approval


async def _extract_and_save_life_context(
    *,
    session_id: str,
    user_id: str,
    query: str,
    pillar: str,
    answer: str,
    clarifying_qa: list,
) -> None:
    """Background task: distil a life-context phrase from the answered session and
    write it back to the same session doc so future /greeting calls can use it."""
    try:
        from greeting import extract_life_context
        snapshot = {
            "query": query,
            "pillar": pillar,
            "answer": answer,
            "clarifying_qa": clarifying_qa,
        }
        life_context = await extract_life_context(snapshot)
        if not life_context:
            return
        try:
            from google.cloud import firestore as gcloud_firestore  # type: ignore
            db = gcloud_firestore.Client()
            db.collection("sessions").document(session_id).set(
                {"life_context": life_context},
                merge=True,
            )
            logger.info(f"life_context saved for session {session_id}: {life_context[:80]}")
        except Exception as e:
            logger.warning(f"Failed to persist life_context: {e}")
    except Exception as e:
        logger.warning(f"life_context extraction failed: {e}")

# ── Session history store ──
_session_history: dict[str, list[dict[str, Any]]] = {}


# ═══════════════════════════════════════════════════════
# LIFESPAN — Ingest RAG docs on startup
# ═══════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ingest RAG documents."""
    logger.info("═" * 50)
    logger.info("  ShaktiAgent — Starting up")
    logger.info("═" * 50)

    try:
        from rag import ingest_documents
        count = await ingest_documents()
        logger.info(f"RAG ready: {count} chunks indexed")
    except Exception as e:
        logger.warning(f"RAG ingestion skipped: {e}")

    # Restore cumulative metrics so deploys/restarts don't wipe the dashboard
    try:
        metrics.load_persisted()
    except Exception as e:
        logger.warning(f"Metrics restore skipped: {e}")

    # Init Vertex AI if possible
    try:
        import vertexai
        project = os.getenv("PROJECT_ID", "deployfest-kv-2026")
        location = os.getenv("VERTEX_LOCATION", "us-central1")
        vertexai.init(project=project, location=location)
        logger.info(f"Vertex AI initialized: {project}/{location}")
    except Exception as e:
        logger.warning(f"Vertex AI init skipped: {e}")

    yield
    logger.info("ShaktiAgent — Shutting down")


# ═══════════════════════════════════════════════════════
# APP INIT
# ═══════════════════════════════════════════════════════

app = FastAPI(
    title="ShaktiAgent",
    description="Multi-Agent AI Life Companion for Women",
    version="1.0.0",
    lifespan=lifespan,
)

# Restrict origins via ALLOWED_ORIGINS env (comma-separated). Default "*".
# Auth is via Authorization header (not cookies), so credentials aren't needed —
# keeping allow_credentials False lets a wildcard origin stay valid + safe.
_allowed = os.getenv("ALLOWED_ORIGINS", "*").strip()
_origins = ["*"] if _allowed == "*" else [o.strip() for o in _allowed.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Lightweight per-IP rate limiter (per-instance; basic abuse protection) ──
import collections
_RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_rate_buckets: dict[str, collections.deque] = collections.defaultdict(collections.deque)


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    # Only throttle the expensive write endpoints; let GETs and static through.
    if request.method == "POST" and request.url.path not in ("/health",):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        bucket = _rate_buckets[client_ip]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please slow down a moment."},
            )
        bucket.append(now)
    return await call_next(request)


# ═══════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    age_band: str = Field(..., pattern=r"^(11-24|25-40|41\+)$")
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None


class ApproveRequest(BaseModel):
    session_id: str


class RejectRequest(BaseModel):
    session_id: str
    reason: Optional[str] = None


# ═══════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.post("/chat")
async def chat(request: ChatRequest, user: dict = Depends(verify_token)):
    """
    Main chat endpoint.
    Classifies intent → routes to sub-agent → validates → returns response.
    """
    # Secure user_id from token
    request.user_id = user["uid"]
    session_id = request.session_id or str(uuid.uuid4())[:12]

    try:
        from orchestrator import process_query
        from memory import extract_and_save_memories, ShortTermMemory

        result = await process_query(
            user_id=request.user_id,
            age_band=request.age_band,
            query=request.query,
            session_id=session_id,
        )

        # Store pending tool calls for human oversight gate
        tools_to_call = result.get("tools_to_call", [])
        if tools_to_call:
            await set_pending_approval(session_id, {
                "tools": tools_to_call,
                "user_id": request.user_id,
                "timestamp": time.time(),
            })

        # Store session history (in-memory)
        if request.user_id not in _session_history:
            _session_history[request.user_id] = []
        _session_history[request.user_id].append({
            "session_id": session_id,
            "timestamp": time.time(),
            "query": request.query,
            "age_band": request.age_band,
            "pillar": result.get("pillar", ""),
            "latency_ms": result.get("latency_ms", 0),
            "citations_count": len(result.get("citations", [])),
            "approved": None,
            "trace_url": result.get("trace_url"),
            "response_preview": result.get("response", "")[:200],
        })
        # Keep only last 50 per user
        _session_history[request.user_id] = _session_history[request.user_id][-50:]

        # Only persist sessions + extract memories on final ANSWER turns,
        # not on intermediate clarifying-question exchanges.
        is_final_answer = result.get("type", "answer") == "answer"

        # Persist session to Firestore for greeting/context recall
        if is_final_answer:
            try:
                from google.cloud import firestore as gcloud_firestore  # type: ignore
                db = gcloud_firestore.Client()
                answer_text = result.get("response", "")
                session_payload = {
                    "user_id": request.user_id,
                    "session_id": session_id,
                    "query": request.query,
                    "age_band": request.age_band,
                    "pillar": result.get("pillar", ""),
                    "citations": result.get("citations", []),
                    "clarifying_qa": result.get("clarifying_qa", []),
                    "answer": answer_text[:1500],
                    "response_preview": answer_text[:300],
                    "created_at": gcloud_firestore.SERVER_TIMESTAMP,
                }
                db.collection("sessions").document(session_id).set(session_payload, merge=True)
            except Exception as e:
                logger.warning(f"Failed to persist session to Firestore: {e}")

            # Async life-context extraction (non-blocking) — patches the same
            # session doc once Gemini returns. Greeting endpoint reads it later.
            asyncio.create_task(_extract_and_save_life_context(
                session_id=session_id,
                user_id=request.user_id,
                query=request.query,
                pillar=result.get("pillar", ""),
                answer=result.get("response", ""),
                clarifying_qa=result.get("clarifying_qa", []),
            ))

        # Async memory extraction (non-blocking) — only on final answer
        if is_final_answer:
            asyncio.create_task(
                extract_and_save_memories(
                    user_id=request.user_id,
                    conversation=[
                        {"role": "user", "content": request.query},
                        {"role": "assistant", "content": result.get("response", "")},
                    ],
                )
            )

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/approve")
async def approve(request: ApproveRequest, user: dict = Depends(verify_token)):
    """Approve and fire queued tool calls for a session."""
    pending = await pop_pending_approval(request.session_id)
    if not pending:
        raise HTTPException(status_code=404, detail="No pending tools for this session.")

    results = []
    for tool_spec in pending["tools"]:
        tool_name = tool_spec["tool"]
        params = tool_spec.get("params", {})
        logger.info(f"Executing approved tool: {tool_name}")
        result = await execute_tool(tool_name, params)
        results.append({"tool": tool_name, "result": result})

    metrics.record_approval()

    # Update session history
    for user_sessions in _session_history.values():
        for session in user_sessions:
            if session["session_id"] == request.session_id:
                session["approved"] = True

    return JSONResponse(content={"session_id": request.session_id, "tool_results": results})


@app.post("/reject")
async def reject(request: RejectRequest, user: dict = Depends(verify_token)):
    """Reject queued tool calls. Logs the rejection."""
    pending = await pop_pending_approval(request.session_id)
    if not pending:
        raise HTTPException(status_code=404, detail="No pending tools for this session.")

    metrics.record_rejection()

    # Update session history
    for user_sessions in _session_history.values():
        for session in user_sessions:
            if session["session_id"] == request.session_id:
                session["approved"] = False

    logger.info(f"Tools REJECTED for session {request.session_id}: {request.reason or 'no reason'}")
    return JSONResponse(content={
        "session_id": request.session_id,
        "status": "rejected",
        "reason": request.reason,
    })


@app.get("/sessions/{user_id}")
async def get_sessions(user_id: str, user: dict = Depends(verify_token)):
    """Return last 20 sessions for a user — reads from Firestore first, falls back to in-memory."""
    real_user_id = user["uid"]

    # Try Firestore (persists across container restarts)
    try:
        from google.cloud import firestore as gcloud_firestore  # type: ignore
        db = gcloud_firestore.Client()
        docs = (
            db.collection("sessions")
            .where("user_id", "==", real_user_id)
            .order_by("created_at", direction=gcloud_firestore.Query.DESCENDING)
            .limit(20)
            .stream()
        )
        sessions = []
        for doc in docs:
            d = doc.to_dict() or {}
            sessions.append({
                "session_id": d.get("session_id", doc.id),
                "timestamp": d.get("created_at").timestamp() if hasattr(d.get("created_at"), "timestamp") else 0,
                "query": d.get("query", ""),
                "age_band": d.get("age_band", ""),
                "pillar": d.get("pillar", ""),
                "latency_ms": 0,
                "citations_count": len(d.get("citations", [])),
                "approved": None,
                "trace_url": d.get("trace_url"),
                "response_preview": d.get("response_preview") or d.get("answer", "")[:200],
            })
        if sessions:
            return JSONResponse(content={"user_id": real_user_id, "sessions": sessions})
    except Exception as e:
        logger.warning(f"Firestore sessions read failed, falling back to in-memory: {e}")

    # Fallback: in-memory history (resets on restart)
    sessions = _session_history.get(real_user_id, [])
    return JSONResponse(content={"user_id": real_user_id, "sessions": sessions[-20:]})


@app.get("/metrics")
async def get_metrics():
    """Return aggregated metrics for the dashboard."""
    return JSONResponse(content=metrics.get_metrics())


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from rag import _vectors_loaded
    return JSONResponse(content={
        "status": "ok",
        "agents": 4,
        "rag": "active" if _vectors_loaded else "loading",
        "memory": "active",
        "observability": "langfuse" if os.getenv("LANGFUSE_PUBLIC_KEY") else "console",
        "web_search": "configured" if os.getenv("TAVILY_API_KEY") else "not configured",
        "web_search_mode": os.getenv("WEB_SEARCH_MODE", "fallback"),
        "llm": "deepseek" if os.getenv("DEEPSEEK_API_KEY") else "gemini",
        "project": os.getenv("PROJECT_ID", "deployfest-kv-2026"),
    })


# ═══════════════════════════════════════════════════════
# ONBOARDING ENDPOINTS
# ═══════════════════════════════════════════════════════

@app.get("/onboarding/state")
async def onboarding_state(user: dict = Depends(verify_token)):
    """Get current onboarding state for the authenticated user."""
    state = await get_onboarding_state(user["uid"])

    if state["complete"]:
        prof = state["profile"] or {}
        return JSONResponse(content={
            "complete": True,
            "profile": {
                "preferred_name": prof.get("preferred_name"),
                "age_band": prof.get("age_band"),
                "language": prof.get("language"),
                "user_mode": prof.get("user_mode"),
                "subject_role": prof.get("subject_role"),
                "asker_role": prof.get("asker_role"),
            },
        })

    next_q = get_current_question(state["answers"])
    visible = get_visible_questions(state["answers"])
    answered = len(state["answers"])
    total = max(len(visible), answered + (1 if next_q else 0))

    return JSONResponse(content={
        "complete": False,
        "current_question": next_q,
        "answered_count": answered,
        "total_steps": total,
        "answers_so_far": state["answers"],
    })


@app.post("/onboarding/answer")
async def onboarding_answer(answer: OnboardingAnswer, user: dict = Depends(verify_token)):
    """Save an onboarding answer and return the next question (or completion)."""
    result = await save_onboarding_answer(user["uid"], answer)
    return JSONResponse(content=result)


@app.post("/onboarding/reset")
async def onboarding_reset(user: dict = Depends(verify_token)):
    """Clear onboarding so the user can re-run the flow (e.g. switch from other to self)."""
    result = await reset_onboarding(user["uid"])
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════
# GREETING ENDPOINT
# ═══════════════════════════════════════════════════════

@app.get("/greeting")
async def greeting(user: dict = Depends(verify_token)):
    """Return a personalized greeting for the returning user."""
    result = await generate_greeting(user["uid"])
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════
# ATTACHMENT UPLOAD
# ═══════════════════════════════════════════════════════

# Caps to keep things safe
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
_ALLOWED_MIME = {
    "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif",
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


@app.post("/attachments/upload")
async def upload_attachment(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user: dict = Depends(verify_token),
):
    """Accept a user file, store in GCS (if available), return a friendly ack.

    The file content is intentionally NOT parsed or sent to any LLM.
    Kalpana just acknowledges receipt so the user feels heard and can continue.
    """
    user_id = user["uid"]
    safe_name = (file.filename or "attachment").replace("/", "_").replace("\\", "_")[:120]

    # Enforce size + mime
    content = await file.read()
    size = len(content)
    if size > _MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large. Please keep it under 10 MB.")
    mime = file.content_type or "application/octet-stream"
    if mime not in _ALLOWED_MIME:
        raise HTTPException(415, f"Unsupported file type: {mime}. Please share an image, PDF, or Word doc.")

    # Try GCS upload (best-effort)
    gs_uri = None
    bucket_name = os.getenv("ATTACHMENTS_BUCKET") or f"{os.getenv('PROJECT_ID', 'shakti360')}-attachments"
    try:
        from google.cloud import storage  # type: ignore
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        # Don't error if bucket missing — just skip
        if bucket.exists():
            blob_path = f"user_attachments/{user_id}/{session_id}/{int(time.time())}_{safe_name}"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(content, content_type=mime)
            gs_uri = f"gs://{bucket_name}/{blob_path}"
            logger.info(f"Attachment uploaded: {gs_uri} ({size} bytes)")
        else:
            logger.warning(f"Attachment bucket '{bucket_name}' does not exist; skipping GCS upload.")
    except Exception as e:
        logger.warning(f"GCS upload failed (continuing without persistence): {e}")

    # Record metadata on the session doc (non-blocking, best-effort)
    try:
        from google.cloud import firestore as gcloud_firestore  # type: ignore
        db = gcloud_firestore.Client()
        db.collection("sessions").document(session_id).set({
            "user_id": user_id,
            "attachments": gcloud_firestore.ArrayUnion([{
                "filename": safe_name,
                "mime": mime,
                "size": size,
                "gs_uri": gs_uri,
                "uploaded_at": time.time(),
            }]),
        }, merge=True)
    except Exception as e:
        logger.warning(f"Session attachment metadata write failed: {e}")

    # If we're in a clarifying flow, advance it AND return the next question
    # in the same response so the frontend doesn't need an extra /chat call.
    advance_result = None
    try:
        from orchestrator import (
            _get_session_state, _set_session_state, _generate_full_answer
        )
        state = await _get_session_state(session_id)
        if state and state.get("phase") == "clarifying":
            qs = state.get("clarifying_questions", [])
            idx = state.get("asked_index", 0)
            qa_pairs = list(state.get("qa_pairs", []))
            if idx < len(qs):
                qa_pairs.append({
                    "q": qs[idx],
                    "a": f"[User shared a document: {safe_name}]",
                })
            new_idx = idx + 1

            if new_idx < len(qs):
                # More clarifying questions remain
                next_q = qs[new_idx]
                await _set_session_state(session_id, {
                    **state, "qa_pairs": qa_pairs, "asked_index": new_idx,
                })
                from clarifier import is_yes_no as _iyn, is_doc_request as _idr
                advance_result = {
                    "type": "clarifying_question",
                    "response": next_q,
                    "question": next_q,
                    "is_yes_no": _iyn(next_q),
                    "is_doc_request": _idr(next_q),
                    "question_index": new_idx,
                    "questions_total": len(qs),
                    "pillar": state.get("pillar"),
                    "session_id": session_id,
                }
            else:
                # All clarifying questions answered — generate the real answer now
                await _set_session_state(session_id, {
                    **state, "qa_pairs": qa_pairs, "phase": "answering",
                })
                advance_result = await _generate_full_answer(
                    user_id=user_id,
                    age_band=state.get("age_band") or "25-40",
                    query=state.get("original_query", ""),
                    pillar=state.get("pillar"),
                    qa_pairs=qa_pairs,
                    session_id=session_id,
                    start_time=time.time(),
                )
    except Exception as e:
        logger.warning(f"Could not advance clarifying state after upload: {e}")

    return JSONResponse(content={
        "ok": True,
        "filename": safe_name,
        "size": size,
        "mime": mime,
        "gs_uri": gs_uri,
        "ack": (
            f"Got it \u2014 I've received {safe_name}. Thanks for trusting me with this."
        ),
        "next": advance_result,
    })


# ═══════════════════════════════════════════════════════
# DIRECT TOOL EXECUTION (for next-step action cards)
# ═══════════════════════════════════════════════════════

class ToolExecuteRequest(BaseModel):
    tool: str
    params: dict[str, Any] = {}

@app.post("/tool/execute")
async def tool_execute(req: ToolExecuteRequest, user: dict = Depends(verify_token)):
    """Execute a tool directly and return its result. Used by next-step action cards."""
    result = await execute_tool(req.tool, req.params)
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════
# SESSION RESET (for "New chat" button)
# ═══════════════════════════════════════════════════════

class SessionResetRequest(BaseModel):
    session_id: str

@app.post("/session/reset")
async def session_reset(req: SessionResetRequest, user: dict = Depends(verify_token)):
    """Clear the orchestrator session state for a given session_id.
    Called when user clicks 'New chat' to ensure the backend doesn't
    carry over follow-up context from the previous conversation.
    """
    from orchestrator import _clear_session_state
    await _clear_session_state(req.session_id)
    logger.info(f"Session state cleared for {req.session_id}")
    return JSONResponse(content={"ok": True, "session_id": req.session_id})


# ═══════════════════════════════════════════════════════
# FRONTEND SERVING
# ═══════════════════════════════════════════════════════

@app.get("/")
async def serve_frontend():
    """Serve the single-page frontend."""
    index_path = Path(__file__).parent / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse(content={"message": "ShaktiAgent API is running. Frontend not found."})


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
