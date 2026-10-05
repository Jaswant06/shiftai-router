"""OpenAI-compatible local server, so existing apps can use ShiftAI unchanged.

    shiftai serve                      # http://127.0.0.1:8800/v1

Point any OpenAI-compatible client at http://127.0.0.1:8800/v1 and pick a model:

    shiftai          route each request at the default quality target
    shiftai-95       route with a 95% quality target (any number from 50 to 100)
    qwen3.5:4b       skip routing and use that installed model directly

A `quality` field in the request body also sets the target. Every response
says which model answered: the `model` field, the X-ShiftAI-Model header, and
an extra `shiftai` object with the reason. Streaming (`"stream": true`) is
supported. Tool calls and images are not.

Routing reads the latest user message; the chosen model receives the whole
conversation.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

from .ollama import ChatResult, ModelNotInstalled, OllamaClient, OllamaUnavailable
from .router import Decision, Router

try:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("The server needs FastAPI: pip install 'shiftai-router[server]'") from exc

ROUTED_NAME = "shiftai"
_ROUTED = re.compile(rf"^{ROUTED_NAME}(?:-(\d{{2,3}}))?$")
DEFAULT_MAX_TOKENS = 1024


@dataclass
class ServerStats:
    """Running totals for the stats endpoint, based on the router's own estimates."""

    requests: int = 0
    routed: int = 0
    by_model: Counter = field(default_factory=Counter)
    est_seconds_saved: float = 0.0
    est_joules_saved: float | None = None  # stays None when the machine has no power telemetry
    overhead_ms_total: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, model: str, decision: Decision | None, largest: str | None) -> None:
        with self.lock:
            self.requests += 1
            self.by_model[model] += 1
            if decision is None or largest is None:
                return
            self.routed += 1
            self.overhead_ms_total += decision.overhead_ms
            self.est_seconds_saved += decision.est_latency_s[largest] - decision.est_latency_s[model]
            big, chosen = decision.est_energy_j[largest], decision.est_energy_j[model]
            if big is not None and chosen is not None:
                self.est_joules_saved = (self.est_joules_saved or 0.0) + big - chosen

    def as_dict(self) -> dict:
        with self.lock:
            return {
                "requests": self.requests,
                "routed_requests": self.routed,
                "requests_by_model": dict(self.by_model),
                "estimated_seconds_saved_vs_largest": round(self.est_seconds_saved, 2),
                "estimated_joules_saved_vs_largest": (
                    round(self.est_joules_saved, 1) if self.est_joules_saved is not None else "not measured"
                ),
                "mean_router_overhead_ms": round(self.overhead_ms_total / self.routed, 1) if self.routed else None,
            }


def message_text(message: dict) -> str:
    """Plain text of an OpenAI message whose content is a string or a list of parts."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")


def to_ollama_messages(messages: list[dict]) -> list[dict]:
    return [{"role": m.get("role", "user"), "content": message_text(m)} for m in messages]


def routing_prompt(messages: list[dict]) -> str:
    """The text the router judges: the latest user message."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message_text(message)
    return message_text(messages[-1]) if messages else ""


def requested_quality(model: str, body: dict, default: float) -> float | None:
    """The quality target for a routed request, or None for a direct model name."""
    match = _ROUTED.match(model)
    if not match:
        return None
    quality = body.get("quality") or (float(match.group(1)) if match.group(1) else default)
    return max(50.0, min(100.0, float(quality)))


def decision_summary(decision: Decision) -> dict:
    return {
        "model": decision.model,
        "reason": decision.reason,
        "quality_target": decision.quality_target,
        "predicted_quality": {m: round(p, 3) for m, p in decision.predicted.items()},
        "estimated_seconds": {m: round(s, 2) for m, s in decision.est_latency_s.items()},
        "optimizing": decision.optimizing,
        "prompt_familiarity": decision.novelty,
        "warning": decision.warning,
        "router_overhead_ms": round(decision.overhead_ms, 1),
    }


def create_app(
    router: Router | None = None,
    client: OllamaClient | None = None,
    default_quality: float = 90,
) -> FastAPI:
    """Build the app. The router loads lazily so the server starts even before setup."""
    app = FastAPI(title="ShiftAI", description="OpenAI-compatible router for local models")
    state = {"router": router, "client": client}

    @app.exception_handler(ModelNotInstalled)
    def model_missing(_, exc: ModelNotInstalled):
        return JSONResponse({"error": {"message": str(exc), "type": "model_not_found"}}, status_code=404)

    @app.exception_handler(OllamaUnavailable)
    def ollama_down(_, exc: OllamaUnavailable):
        return JSONResponse({"error": {"message": str(exc), "type": "ollama_unavailable"}}, status_code=503)
    stats = ServerStats()

    def get_router() -> Router:
        if state["router"] is None:
            try:
                state["router"] = Router(client=state["client"])
            except (FileNotFoundError, RuntimeError) as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["router"]

    def get_client() -> OllamaClient:
        if state["client"] is None:
            state["client"] = state["router"].client if state["router"] else OllamaClient()
        return state["client"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "router_loaded": state["router"] is not None}

    @app.get("/v1/models")
    def list_models() -> dict:
        now = int(time.time())
        names = [ROUTED_NAME] + [f"{ROUTED_NAME}-{q}" for q in (80, 90, 95, 99)]
        try:
            names += [m["name"] for m in get_client().tags()]
        except Exception:  # Ollama down: still advertise the routed names
            pass
        return {"object": "list", "data": [{"id": n, "object": "model", "created": now, "owned_by": "shiftai"} for n in names]}

    @app.get("/shiftai/stats")
    def get_stats() -> dict:
        return stats.as_dict()

    # Handlers are plain functions: FastAPI runs them in worker threads, so a
    # long generation never blocks other requests.
    @app.post("/shiftai/decide")
    def decide(body: dict = Body(...)) -> dict:
        """Show the routing decision for a prompt without generating an answer."""
        prompt = body.get("prompt") or routing_prompt(body.get("messages", []))
        decision = get_router().decide(prompt, float(body.get("quality", default_quality)))
        return decision_summary(decision)

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict = Body(...)):
        messages = body.get("messages") or []
        if not messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        requested = body.get("model") or ROUTED_NAME
        quality = requested_quality(requested, body, default_quality)

        decision, largest = None, None
        if quality is not None:
            router_ = get_router()
            decision = router_.decide(routing_prompt(messages), quality)
            model, largest = decision.model, router_.largest
        else:
            model = requested

        options = {
            "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS),
            "temperature": float(body.get("temperature", 0.0)),
            "seed": body.get("seed"),
        }
        ollama_messages = to_ollama_messages(messages)
        stats.record(model, decision, largest)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        headers = {"X-ShiftAI-Model": model}
        if decision is not None:
            headers["X-ShiftAI-Reason"] = decision.reason
        extra = {"shiftai": decision_summary(decision)} if decision else {}

        if body.get("stream"):
            def chunk(delta: dict, finish: str | None = None, **more) -> str:
                payload = {
                    "id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    **more,
                }
                return f"data: {json.dumps(payload)}\n\n"

            def events():
                yield chunk({"role": "assistant", "content": ""})
                finish = "stop"
                for piece in get_client().chat_stream(model, ollama_messages, **options):
                    if isinstance(piece, ChatResult):
                        finish = "length" if piece.truncated else "stop"
                    else:
                        yield chunk({"content": piece})
                yield chunk({}, finish, **extra)
                yield "data: [DONE]\n\n"

            return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

        result = get_client().chat(model, ollama_messages, **options)
        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "length" if result.truncated else "stop",
            }],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.prompt_tokens + result.output_tokens,
            },
            **extra,
        }
        return JSONResponse(payload, headers=headers)

    return app
