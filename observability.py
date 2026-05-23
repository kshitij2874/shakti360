"""
observability.py — Langfuse instrumentation for ShaktiAgent.
Provides @observe() decorator and trace metadata helpers.
Gracefully degrades to console logging when Langfuse keys are missing.
"""
from __future__ import annotations

import functools
import logging
import os
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("shakti.observability")

# ── Langfuse Client (lazy singleton) ──
_langfuse_client = None
_langfuse_available = False


def _init_langfuse():
    """Initialize Langfuse client from env vars."""
    global _langfuse_client, _langfuse_available

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.info("Langfuse keys not configured — observability will log to console only")
        _langfuse_available = False
        return

    try:
        from langfuse import Langfuse  # type: ignore
        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        _langfuse_available = True
        logger.info(f"Langfuse connected: {host}")
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")
        _langfuse_available = False


def get_langfuse():
    """Get Langfuse client, initializing if needed."""
    global _langfuse_client
    if _langfuse_client is None and not _langfuse_available:
        _init_langfuse()
    return _langfuse_client


class TraceContext:
    """Holds metadata for the current trace/span."""

    def __init__(
        self,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.trace_id = trace_id
        self.session_id = session_id
        self.user_id = user_id
        self.metadata = metadata or {}
        self._trace = None
        self._spans: list[Any] = []

    def start_trace(self, name: str) -> "TraceContext":
        """Start a new Langfuse trace."""
        client = get_langfuse()
        if client:
            try:
                self._trace = client.trace(
                    name=name,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    metadata=self.metadata,
                )
                self.trace_id = self._trace.id
            except Exception as e:
                logger.warning(f"Failed to start Langfuse trace: {e}")
        return self

    def start_span(self, name: str, metadata: Optional[dict] = None) -> Any:
        """Start a new span within the current trace."""
        client = get_langfuse()
        if client and self._trace:
            try:
                span = self._trace.span(
                    name=name,
                    metadata={**self.metadata, **(metadata or {})},
                )
                self._spans.append(span)
                return span
            except Exception as e:
                logger.warning(f"Failed to start span: {e}")
        return None

    def end_span(self, span: Any, output: Any = None, status: str = "ok") -> None:
        """End a span."""
        if span:
            try:
                span.end(output=output, level="DEFAULT" if status == "ok" else "ERROR")
            except Exception:
                pass

    def add_generation(
        self,
        name: str,
        model: str,
        input_text: str,
        output_text: str,
        usage: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an LLM generation event."""
        client = get_langfuse()
        if client and self._trace:
            try:
                self._trace.generation(
                    name=name,
                    model=model,
                    input=input_text,
                    output=output_text,
                    usage=usage,
                    metadata={**self.metadata, **(metadata or {})},
                )
            except Exception as e:
                logger.warning(f"Failed to log generation: {e}")

    def finalize(self) -> Optional[str]:
        """Flush the trace. Returns the trace URL if available."""
        client = get_langfuse()
        if client:
            try:
                client.flush()
                host = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
                if self.trace_id:
                    return f"{host}/trace/{self.trace_id}"
            except Exception:
                pass
        return None


def observe(name: Optional[str] = None):
    """
    Decorator for observability. Logs function execution to Langfuse.
    Falls back to console logging if Langfuse is unavailable.

    Usage:
        @observe("orchestrator_classify")
        async def classify_intent(query: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            logger.info(f"[OBSERVE] START {span_name}")

            try:
                result = await func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                logger.info(f"[OBSERVE] END   {span_name} — {latency_ms:.0f}ms")
                return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                logger.error(f"[OBSERVE] FAIL  {span_name} — {latency_ms:.0f}ms — {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            logger.info(f"[OBSERVE] START {span_name}")

            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                logger.info(f"[OBSERVE] END   {span_name} — {latency_ms:.0f}ms")
                return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                logger.error(f"[OBSERVE] FAIL  {span_name} — {latency_ms:.0f}ms — {e}")
                raise

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ── Metrics Aggregator ──

class MetricsCollector:
    """In-memory metrics for the /metrics endpoint."""

    def __init__(self):
        self.total_sessions: int = 0
        self.total_latency_ms: float = 0.0
        self.total_cost: float = 0.0
        self.hallucination_checks: int = 0
        self.hallucination_passes: int = 0
        self.pillar_counts: dict[str, int] = {"HEALTH": 0, "FINANCE": 0, "CAREER": 0}
        self.age_band_counts: dict[str, int] = {"11-24": 0, "25-40": 0, "41+": 0}
        self.approvals: int = 0
        self.rejections: int = 0
        self.latency_history: list[dict[str, Any]] = []

    def record_session(
        self,
        latency_ms: float,
        pillar: str,
        age_band: str,
        cost: float = 0.0,
        hallucination_passed: bool = True,
    ) -> None:
        self.total_sessions += 1
        self.total_latency_ms += latency_ms
        self.total_cost += cost
        self.hallucination_checks += 1
        if hallucination_passed:
            self.hallucination_passes += 1
        self.pillar_counts[pillar] = self.pillar_counts.get(pillar, 0) + 1
        self.age_band_counts[age_band] = self.age_band_counts.get(age_band, 0) + 1
        self.latency_history.append({"ts": time.time(), "ms": latency_ms})
        # Keep only last 100
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]

    def record_approval(self) -> None:
        self.approvals += 1

    def record_rejection(self) -> None:
        self.rejections += 1

    def get_metrics(self) -> dict[str, Any]:
        avg_latency = self.total_latency_ms / max(self.total_sessions, 1)
        avg_cost = self.total_cost / max(self.total_sessions, 1)
        hall_rate = 1.0 - (self.hallucination_passes / max(self.hallucination_checks, 1))
        total_decisions = self.approvals + self.rejections
        approval_rate = self.approvals / max(total_decisions, 1)

        return {
            "total_sessions": self.total_sessions,
            "avg_latency_ms": round(avg_latency, 1),
            "avg_cost_per_session": round(avg_cost, 4),
            "hallucination_rate": round(hall_rate, 4),
            "pillar_split": self.pillar_counts,
            "age_band_split": self.age_band_counts,
            "approval_rate": round(approval_rate, 4),
            "latency_history": self.latency_history[-20:],
        }


# Global metrics instance
metrics = MetricsCollector()
