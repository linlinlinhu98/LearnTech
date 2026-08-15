"""Local-only HTTP server for terminal testing (no AgentScope runtime).

The Bailian platform runs ``main.py`` (AgentScope ReActAgent + runtime engine,
AgentScope v1.0.11). That runtime's ``agentscope.agent.ReActAgent`` does not
exist in the locally installed AgentScope v2.x, so ``main.py`` cannot be
imported here.

This file mirrors the same endpoints using only FastAPI + the AgentScope-free
``TutorDispatcher`` / ``tutor_core``, so the full HTTP chain can be curl-tested
locally. ``/process/sync`` falls back to ``TutorDispatcher`` instead of a
ReActAgent — same five sub-agent tools, same plan→execute→synthesize flow.

Run:
    python local_server.py            # http://127.0.0.1:8080
"""

from __future__ import annotations

import os
import socket
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ---- Lumen imports (dual-mode: package vs direct execution) ----
try:
    from .knowledge_store import (
        LessonChunk,
        embedding_provider,
        knowledge_store,
        seed_demo_data,
    )
    from .dispatcher import TutorDispatcher
    from .tutor_core import build_tool_registry
except ImportError:  # Direct execution / local tests
    from knowledge_store import (
        LessonChunk,
        embedding_provider,
        knowledge_store,
        seed_demo_data,
    )
    from dispatcher import TutorDispatcher
    from tutor_core import build_tool_registry


# ---- flat YAML config (no pyyaml) ----
def _read_flat_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yml")
    result = {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if value.lower() == "true":
                    result[key] = True
                elif value.lower() == "false":
                    result[key] = False
                elif value.isdigit():
                    result[key] = int(value)
                else:
                    result[key] = value
    except Exception:
        pass
    return result


_config = _read_flat_config()


def _get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


LOCAL_IP = _get_local_ip()


def _ok(data=None, message="success"):
    return {"code": 200, "message": message, "data": data, "host": LOCAL_IP}


app = FastAPI(title="Lumen-Bailian (local)")


# ---- seed demo course data once at startup ----
try:
    seed_demo_data()
except Exception:
    pass


# ============================================================
# Standard endpoints
# ============================================================
@app.get("/")
def read_root():
    return {"name": "Lumen-Bailian", "status": "running"}


@app.get("/health")
def health_check():
    return "OK"


@app.post("/createSession")
def create_session():
    return {"uniqueCode": str(uuid.uuid4()), **_ok()}


# ============================================================
# Knowledge management
# ============================================================
@app.post("/api/v1/knowledge/stats")
def knowledge_stats():
    return _ok(data={
        "total_chunks": len(knowledge_store),
        "embedding_model": embedding_provider.model_id,
        "embedding_dim": embedding_provider.dim,
    })


@app.post("/api/v1/knowledge/search")
async def knowledge_search(request: Request):
    body = await request.json()
    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "query is required"})
    try:
        query_emb = embedding_provider.embed([query])[0]
        results = knowledge_store.search(query_emb, top_k=body.get("top_k", 5))
    except Exception as exc:
        return _ok(
            data={"query": query, "results": [], "error": str(exc)},
            message="search completed with errors",
        )
    return _ok(data={
        "query": query,
        "results": [
            {
                "lesson_id": chunk.lesson_id,
                "lesson_title": chunk.lesson_title,
                "text_preview": chunk.text[:200],
                "similarity_score": round(score, 4),
            }
            for chunk, score in results
        ],
    })


@app.post("/api/v1/knowledge/ingest")
async def knowledge_ingest(request: Request):
    body = await request.json()
    raw_chunks = body.get("chunks", [])
    if not raw_chunks:
        return JSONResponse(status_code=400, content={"error": "chunks is required"})

    chunks, texts = [], []
    for c in raw_chunks:
        chunks.append(LessonChunk(
            id=str(uuid.uuid4())[:8],
            lesson_id=c.get("lesson_id", ""),
            lesson_title=c.get("lesson_title", "Untitled"),
            text=c.get("text", ""),
            course_title=body.get("course_title", ""),
        ))
        texts.append(c.get("text", ""))

    try:
        embeddings = embedding_provider.embed(texts)
    except Exception:
        embeddings = embedding_provider._noop_embed(texts)

    knowledge_store.add_chunks(chunks, embeddings)
    return _ok(data={
        "chunks_ingested": len(chunks),
        "total_chunks": len(knowledge_store),
    })


# ============================================================
# Multi-agent tutor (Module 3)
# ============================================================
def _make_dispatcher() -> TutorDispatcher:
    return TutorDispatcher(
        tools=build_tool_registry(),
        max_iterations=int(_config.get("MAX_DISPATCH_ITERATIONS", 5)),
        max_llm_calls=int(_config.get("MAX_DISPATCH_LLM_CALLS", 8)),
    )


@app.post("/api/v1/tutor/ask")
async def tutor_ask(request: Request):
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "question is required"})
    result = await _make_dispatcher().run(
        question=question,
        user_id=str(body.get("user_id") or ""),
        course_id=str(body.get("course_id") or ""),
        history=body.get("history") or [],
    )
    return _ok(data=result)


@app.post("/process/sync")
async def process_sync(request: Request):
    """Bailian-compatible chat entry; locally falls back to TutorDispatcher."""
    body = await request.json()
    messages = body.get("messages") or body.get("msgs") or []
    if not isinstance(messages, list):
        messages = []

    question = ""
    for m in reversed(messages):
        if isinstance(m, dict) and str(m.get("role")) == "user":
            question = str(m.get("content") or "").strip()
            if question:
                break
    if not question:
        question = str(body.get("content") or "").strip()

    if not question:
        return JSONResponse(status_code=400, content={"error": "no user message found"})

    history = [m for m in messages if isinstance(m, dict) and m.get("content")][-6:]
    result = await _make_dispatcher().run(
        question=question,
        user_id=str(body.get("user_id") or ""),
        history=history,
    )
    return _ok(data=result)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("START_HOST") or _config.get("START_HOST", "127.0.0.1")
    port = int(os.getenv("PORT") or _config.get("PORT", 8080))
    uvicorn.run(app, host=host, port=port)
