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
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ---- Lumen imports (dual-mode: package vs direct execution) ----
try:
    from . import authoring as _authoring_mod
    from . import intake as _intake_mod
    from .knowledge_store import (
        LessonChunk,
        embedding_provider,
        knowledge_store,
        seed_demo_data,
    )
    from .dispatcher import TutorDispatcher
    from . import tutor_core as _tutor_core_mod
    from .tutor_core import build_tool_registry
    from .file_parser import parse_file, SUPPORTED
    from .llm_utils import LlmBudget
except ImportError:  # Direct execution / local tests
    import authoring as _authoring_mod
    import intake as _intake_mod
    from knowledge_store import (
        LessonChunk,
        embedding_provider,
        knowledge_store,
        seed_demo_data,
    )
    from dispatcher import TutorDispatcher
    import tutor_core as _tutor_core_mod
    from tutor_core import build_tool_registry
    from file_parser import parse_file, SUPPORTED
    from llm_utils import LlmBudget


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    course_id = str(body.get("course_id") or "")
    try:
        query_emb = embedding_provider.embed([query])[0]
        results = knowledge_store.search(
            query_emb,
            top_k=body.get("top_k", 5),
            course_id=course_id,
        )
    except Exception as exc:
        return _ok(
            data={"query": query, "results": [], "error": str(exc)},
            message="search completed with errors",
        )
    return _ok(data={
        "query": query,
        "course_id": course_id,
        "results": [
            {
                "lesson_id": chunk.lesson_id,
                "lesson_title": chunk.lesson_title,
                "course_id": chunk.course_id,
                "text_preview": chunk.text[:200],
                "similarity_score": round(score, 4),
            }
            for chunk, score in results
        ],
    })


# ============================================================
# Module 1: Guided Intake (Learning Brief)
# ============================================================
_intake_mgr = _intake_mod.get_intake_manager()


@app.post("/api/v1/intake/start")
async def intake_start(request: Request):
    """Start a new intake session for a user."""
    body = await request.json()
    user_id = str(body.get("user_id") or str(uuid.uuid4())[:8])
    session = _intake_mgr.start(user_id)
    first_question = _intake_mod.IntakeManager._default_next_question(session)
    return _ok(data={
        "session_id": session.user_id,
        "state": session.state,
        "round": session.round,
        "max_rounds": 6,
        "brief": dict(session.brief),
        "next_question": first_question,
    })


@app.post("/api/v1/intake/respond")
async def intake_respond(request: Request):
    """Feed a user message into the active intake session and get the next question or confirmed brief."""
    body = await request.json()
    user_id = str(body.get("user_id") or "")
    message = str(body.get("message") or "")
    if not user_id:
        return JSONResponse(status_code=400, content={"error": "user_id is required"})
    result = _intake_mgr.respond(user_id, message)
    return _ok(data=result)


@app.post("/api/v1/intake/status")
async def intake_status(request: Request):
    """Get the current state of an intake session."""
    body = await request.json()
    user_id = str(body.get("user_id") or "")
    session = _intake_mgr.get(user_id)
    if not session:
        return _ok(data={"state": "NOT_FOUND"})
    return _ok(data={
        "session_id": session.user_id,
        "state": session.state,
        "round": session.round,
        "brief": dict(session.brief),
    })


@app.post("/api/v1/intake/cancel")
async def intake_cancel(request: Request):
    """Cancel the active intake session."""
    body = await request.json()
    user_id = str(body.get("user_id") or "")
    _intake_mgr.cancel(user_id)
    return _ok(message="intake session cancelled")


# ============================================================
# Module 2: Course Management
# ============================================================

@app.post("/api/v1/courses/generate")
async def generate_course(request: Request):
    """Run the 6-stage authoring pipeline to generate a course from a LearningBrief.

    Body: {"brief": {"goal": "...", "current_level": "...", ...}}
    """
    body = await request.json()
    brief = body.get("brief") or {}
    if not brief.get("goal"):
        return JSONResponse(status_code=400, content={"error": "brief.goal is required"})

    pipeline = _authoring_mod.AuthoringPipeline()
    result = await pipeline.run(brief)

    # Ingest chunks into knowledge store
    chunks = _authoring_mod.ingest_course(
        result.get("raw_content", {}),
        course_title=result.get("raw_content", {}).get("course_title", ""),
    )
    course_id = result.get("course_id", "")
    course_title = result.get("raw_content", {}).get("course_title", "")

    chunk_objs, texts = [], []
    for c in chunks:
        chunk_objs.append(LessonChunk(
            id=c["id"],
            lesson_id=c["lesson_id"],
            lesson_title=c["lesson_title"],
            section_id=c.get("section_id", ""),
            text=c["text"],
            course_id=course_id,
            course_title=course_title,
        ))
        texts.append(c["text"])

    try:
        embeddings = embedding_provider.embed(texts)
    except Exception:
        embeddings = embedding_provider._noop_embed(texts)

    knowledge_store.add_chunks(chunk_objs, embeddings)

    return _ok(data={
        "course_id": course_id,
        "course_title": course_title,
        "outline": result.get("outline", {}),
        "final_verdict": result.get("final_verdict", {}),
        "llm_calls_used": result.get("llm_calls_used", 0),
        "chunks_ingested": len(chunk_objs),
        "pipeline_stage": result.get("pipeline_stage", {}),
    })


@app.post("/api/v1/courses/list")
def list_courses():
    """List all courses stored in the knowledge base."""
    return _ok(data={"courses": knowledge_store.list_courses()})


@app.post("/api/v1/knowledge/ingest")
async def knowledge_ingest(request: Request):
    body = await request.json()
    raw_chunks = body.get("chunks", [])
    if not raw_chunks:
        return JSONResponse(status_code=400, content={"error": "chunks is required"})

    course_id = str(body.get("course_id") or "")
    course_title = str(body.get("course_title") or "")

    chunks, texts = [], []
    for c in raw_chunks:
        chunks.append(LessonChunk(
            id=str(uuid.uuid4())[:8],
            lesson_id=c.get("lesson_id", ""),
            lesson_title=c.get("lesson_title", "Untitled"),
            text=c.get("text", ""),
            course_id=course_id,
            course_title=course_title,
        ))
        texts.append(c.get("text", ""))

    try:
        embeddings = embedding_provider.embed(texts)
    except Exception:
        embeddings = embedding_provider._noop_embed(texts)

    knowledge_store.add_chunks(chunks, embeddings)
    return _ok(data={
        "course_id": course_id,
        "chunks_ingested": len(chunks),
        "total_chunks": len(knowledge_store),
    })


@app.post("/api/v1/knowledge/upload")
async def knowledge_upload(request: Request):
    """Parse and ingest a uploaded file (.md .pdf .docx .pptx).

    Uses FastAPI's UploadFile for multipart/form-data, or accepts raw
    file bytes + filename in JSON body as fallback.
    """
    import io as _io

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        # FastAPI multipart
        form = await request.form()
        file = form.get("file")
        if file is None:
            return JSONResponse(status_code=400, content={"error": "file part is required"})
        filename = getattr(file, "filename", "unknown")
        file_bytes = await file.read()
    else:
        # JSON body: { "filename": "...", "content": "<base64>" }
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
        filename = str(body.get("filename") or "")
        b64 = str(body.get("content") or "")
        if not b64:
            return JSONResponse(status_code=400, content={"error": "content (base64) is required"})
        try:
            import base64 as _base64
            file_bytes = _base64.b64decode(b64)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"base64 decode error: {e}"})

    course_id = str(body.get("course_id") or "") if "multipart" not in content_type else ""
    course_title = str(body.get("course_title") or "") if "multipart" in content_type else ""
    user_id = str(body.get("user_id") or "") if "multipart" in content_type else ""

    try:
        text = parse_file(file_bytes, filename)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    if len(text) < 50:
        return JSONResponse(status_code=422, content={"error": "文件内容过少，无法导入"})

    # Chunk the text (simple 300-char chunks with 50-char overlap)
    chunk_size = 300
    overlap = 50
    chunks, texts = [], []
    i = 0
    while i < len(text):
        chunk_text = text[i : i + chunk_size].strip()
        if chunk_text:
            cid = str(uuid.uuid4())[:8]
            chunks.append(LessonChunk(
                id=cid,
                lesson_id=f"uploaded_{cid}",
                lesson_title=f"{Path(filename).stem} (片段{len(chunks)+1})",
                section_id="",
                text=chunk_text,
                course_id=course_id,
                course_title=course_title,
            ))
            texts.append(chunk_text)
        i += chunk_size - overlap

    if texts:
        try:
            embeddings = embedding_provider.embed(texts)
        except Exception:
            embeddings = embedding_provider._noop_embed(texts)
        knowledge_store.add_chunks(chunks, embeddings)

    return _ok(data={
        "filename": filename,
        "text_length": len(text),
        "chunks_ingested": len(chunks),
        "total_chunks": len(knowledge_store),
        "preview": text[:200],
    })


@app.post("/api/v1/knowledge/courses")
def list_courses():
    """List all courses stored in the knowledge base."""
    return _ok(data={"courses": knowledge_store.list_courses()})


# ============================================================
# P0-2: Review Schedule
# ============================================================
@app.post("/api/v1/review/schedule")
async def review_schedule(request: Request):
    """FSRS-based review scheduling. Exam date optional."""
    body = await request.json()
    result = await _tutor_core_mod.schedule_review(
        exam_date=str(body.get("exam_date") or ""),
        course_id=str(body.get("course_id") or ""),
        user_id=str(body.get("user_id") or ""),
        top_k=int(body.get("top_k") or 5),
    )
    import json as _json
    data = _json.loads(result)
    return _ok(data=data)


# ============================================================
# P1-1: Mastery Report
# ============================================================
@app.post("/api/v1/mastery/report")
async def mastery_report(request: Request):
    """Generate mastery report for a course."""
    body = await request.json()
    result = await _tutor_core_mod.mastery_report(
        course_id=str(body.get("course_id") or ""),
        user_id=str(body.get("user_id") or ""),
    )
    import json as _json
    data = _json.loads(result)
    return _ok(data=data)


# ============================================================
# P1-2: Adaptive Mock Exam
# ============================================================
@app.post("/api/v1/mock/start")
async def mock_start(request: Request):
    """Start a new mock exam session."""
    body = await request.json()
    result = await _tutor_core_mod.mock_exam(
        action="start",
        course_id=str(body.get("course_id") or ""),
        user_id=str(body.get("user_id") or ""),
    )
    import json as _json
    data = _json.loads(result)
    return _ok(data=data)


@app.post("/api/v1/mock/answer")
async def mock_answer(request: Request):
    """Process an answer: grade it with LLM, record result, return next question."""
    body = await request.json()
    budget = LlmBudget(max_calls=8)
    result = await _tutor_core_mod.mock_exam(
        action="answer",
        course_id=str(body.get("course_id") or ""),
        user_id=str(body.get("user_id") or ""),
        budget=budget,
        correct=bool(body.get("correct")) if "correct" in body else None,
        answer_text=str(body.get("answer_text") or "").strip(),
    )
    import json as _json
    data = _json.loads(result)
    return _ok(data=data)


@app.post("/api/v1/mock/record")
async def mock_record(request: Request):
    """Record a student's answer to a quiz question and update FSRS state."""
    body = await request.json()
    chunk_id = str(body.get("chunk_id") or "")
    user_id = str(body.get("user_id") or "")
    correct = bool(body.get("correct", False))
    if not chunk_id or not user_id:
        return JSONResponse(status_code=400, content={"error": "chunk_id and user_id are required"})
    result = await _tutor_core_mod.record_answer(
        chunk_id=chunk_id,
        user_id=user_id,
        correct=correct,
    )
    import json as _json
    data = _json.loads(result)
    return _ok(data=data)


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
