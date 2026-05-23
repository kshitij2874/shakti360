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
from fastapi import FastAPI, HTTPException, Depends
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


# ── Pending tool calls store (for human oversight gate) ──
_pending_approvals: dict[str, dict[str, Any]] = {}


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            _pending_approvals[session_id] = {
                "tools": tools_to_call,
                "user_id": request.user_id,
                "timestamp": time.time(),
            }

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
    pending = _pending_approvals.pop(request.session_id, None)
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
    pending = _pending_approvals.pop(request.session_id, None)
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
    """Return last 20 sessions for a user (secured by token UID)."""
    real_user_id = user["uid"]
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
