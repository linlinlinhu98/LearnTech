"""Lumen-Bailian — AI-Powered Personalized Learning Path Planner.

Built on the AgentScope Bailian reference template (v1.0.11 / OpenTrek).
Adds RAG-powered tutoring with course search, quiz generation,
and concept explanation tools.

Dual-mode design:
  - Platform: runs as an AgentScope AgentApp with ReActAgent + runtime engine.
  - Local : falls back to FastAPI + TutorDispatcher (no AgentScope runtime needed).
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
import uuid
from logging.handlers import RotatingFileHandler
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from opentelemetry import trace as ot_trace

import httpx  # only used in model-client builder; available in both envs

# ---- Lumen internal imports (always safe — no AgentScope dependency) ----
try:
    from . import agent_tools as _agent_tools_mod
    from . import knowledge_store as _ks_mod
    from . import prompts as _prompts_mod
    from . import tutor_core as _tc_mod
    from . import dispatcher as _disp_mod
    _TOP_LEVEL_PKG = True
except ImportError:
    import agent_tools as _agent_tools_mod
    import knowledge_store as _ks_mod
    import prompts as _prompts_mod
    import tutor_core as _tc_mod
    import dispatcher as _disp_mod
    _TOP_LEVEL_PKG = False

# -----------------------------------------------------------------------
# AgentScope framework — imported lazily only when running on the platform.
# All members used here (ReActAgent, AgentApp, etc.) are platform v1.0.11
# only.  Locally (v1.1.6+) these imports would fail, but we never invoke
# the platform code path locally — local testing goes through local_server.
# -----------------------------------------------------------------------

_AS = None  # deferred AgentScope namespace

def _get_agentscope() -> Any:
    """Lazily import and cache the full AgentScope + runtime namespace."""
    global _AS
    if _AS is not None:
        return _AS

    from agentscope.agent import ReActAgent
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.model import OpenAIChatModel
    from agentscope.tool import Toolkit
    from agentscope_runtime.adapters.agentscope.memory import AgentScopeSessionHistoryMemory
    from agentscope_runtime.engine import AgentApp, LocalDeployManager
    from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
    from agentscope_runtime.engine.tracing import TraceType, TracingUtil, trace
    from agentscope_runtime.engine.services.agent_state import InMemoryStateService
    from agentscope_runtime.engine.services.session_history import InMemorySessionHistoryService

    _AS = dict(
        ReActAgent=ReActAgent,
        OpenAIChatFormatter=OpenAIChatFormatter,
        OpenAIChatModel=OpenAIChatModel,
        Toolkit=Toolkit,
        AgentScopeSessionHistoryMemory=AgentScopeSessionHistoryMemory,
        AgentApp=AgentApp,
        LocalDeployManager=LocalDeployManager,
        AgentRequest=AgentRequest,
        InMemoryStateService=InMemoryStateService,
        InMemorySessionHistoryService=InMemorySessionHistoryService,
        TraceType=TraceType,
        TracingUtil=TracingUtil,
        trace=trace,
    )
    return _AS


def _get_stream_func():
    """Return stream_printing_messages from agentscope.pipeline (platform only)."""
    from agentscope.pipeline import stream_printing_messages
    return stream_printing_messages


# ============================================================
# Config reader
# ============================================================
def read_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    cfg = {}
    with open(config_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            cfg[key] = value
    return cfg


config = read_config()

# ============================================================
# Logging setup
# ============================================================
_LOG_DIR = os.getenv("APP_LOG_DIR", "/home/admin/app_logs")
try:
    os.makedirs(_LOG_DIR, exist_ok=True)
except OSError:
    _LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(_LOG_DIR, exist_ok=True)

_logger = logging.getLogger("lumen_bailian")
_log_lvl = (os.getenv("LOG_LEVEL") or config.get("LOG_LEVEL", "INFO")).upper()
_logger.setLevel(_log_lvl)

if not _logger.handlers:
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    _fh = RotatingFileHandler(
        os.path.join(_LOG_DIR, os.getenv("LOG_FILE_NAME") or config.get("LOG_FILE_NAME", "lumen_bailian.log")),
        maxBytes=50 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    _fh.setFormatter(_fmt)
    _logger.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    _logger.addHandler(_sh)

# ============================================================
# Helpers
# ============================================================
def _local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


LOCAL_IP = _local_ip()


def success_response(data=None, message="success"):
    return {"code": 200, "message": message, "data": data, "host": LOCAL_IP}


# ============================================================
# Platform-only AgentApp (only constructed on the platform)
# ============================================================
def _create_agent_app():
    AS = _get_agentscope()

    agent_app = AS["AgentApp"](
        app_name=config.get("APP_NAME", "Lumen-Bailian"),
        app_description="AI-Powered Personalized Learning Path Planner with RAG.",
    )

    @agent_app.init
    async def init_func(self):
        self.state_service = AS["InMemoryStateService"]()
        self.session_service = AS["InMemorySessionHistoryService"]()
        await self.state_service.start()
        await self.session_service.start()
        try:
            _ks_mod.seed_demo_data()
            _logger.info("Knowledge store seeded: %s chunks", len(_ks_mod.knowledge_store))
        except Exception as exc:
            _logger.warning("Seeding failed (non-fatal): %s", exc)

    @agent_app.shutdown
    async def shutdown_func(self):
        await self.state_service.stop()
        await self.session_service.stop()

    @agent_app.endpoint("/")
    @AS["trace"](trace_type=AS["TraceType"].LLM, trace_name="root", is_root_span=True)
    def read_root():
        return {"name": config.get("APP_NAME", "Lumen-Bailian"), "status": "running"}

    @agent_app.endpoint("/health")
    @AS["trace"](trace_type=AS["TraceType"].OTHER, trace_name="health", is_root_span=True)
    def health_check():
        return "OK"

    @agent_app.endpoint("/createSession")
    @AS["trace"](trace_type=AS["TraceType"].OTHER, trace_name="create_session", is_root_span=True)
    def create_session(request: Request, body: dict):
        return {"uniqueCode": str(uuid.uuid4()), **success_response()}

    @agent_app.endpoint("/clearSession")
    @AS["trace"](trace_type=AS["TraceType"].OTHER, trace_name="clear_session", is_root_span=True)
    def clear_session(request: Request, body: dict):
        return success_response(message="session_id cleared")

    @agent_app.endpoint("/abortSession")
    @AS["trace"](trace_type=AS["TraceType"].OTHER, trace_name="abort_session", is_root_span=True)
    def abort_session(request: Request, body: dict):
        return success_response(message="session_id aborted")

    # /api/v1/knowledge/*
    @agent_app.endpoint("/api/v1/knowledge/stats")
    @AS["trace"](trace_type=AS["TraceType"].OTHER, trace_name="knowledge_stats", is_root_span=True)
    def knowledge_stats(request: Request):
        return success_response(data={
            "total_chunks": len(_ks_mod.knowledge_store),
            "embedding_model": _ks_mod.embedding_provider.model_id,
            "embedding_dim": _ks_mod.embedding_provider.dim,
        })

    @agent_app.endpoint("/api/v1/knowledge/search", methods=["POST"])
    @AS["trace"](trace_type=AS["TraceType"].OTHER, trace_name="knowledge_search", is_root_span=True)
    async def knowledge_search(request: Request, body: dict):
        query = (body.get("query") or "").strip()
        if not query:
            return JSONResponse(status_code=400, content={"error": "query is required"})
        try:
            emb = _ks_mod.embedding_provider.embed([query])[0]
            results = _ks_mod.knowledge_store.search(emb, top_k=body.get("top_k", 5))
        except Exception as exc:
            return success_response(
                data={"query": query, "results": [], "error": str(exc)},
                message="search completed with errors",
            )
        return success_response(data={
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

    @agent_app.endpoint("/api/v1/knowledge/ingest", methods=["POST"])
    @AS["trace"](trace_type=AS["TraceType"].OTHER, trace_name="knowledge_ingest", is_root_span=True)
    async def knowledge_ingest(request: Request, body: dict):
        raw_chunks = body.get("chunks", [])
        if not raw_chunks:
            return JSONResponse(status_code=400, content={"error": "chunks is required"})
        chunks, texts = [], []
        for c in raw_chunks:
            chunks.append(_ks_mod.LessonChunk(
                id=str(uuid.uuid4())[:8],
                lesson_id=c.get("lesson_id", ""),
                lesson_title=c.get("lesson_title", "Untitled"),
                text=c.get("text", ""),
                course_title=body.get("course_title", ""),
            ))
            texts.append(c.get("text", ""))
        try:
            embeddings = _ks_mod.embedding_provider.embed(texts)
        except Exception:
            embeddings = _ks_mod.embedding_provider._noop_embed(texts)
        _ks_mod.knowledge_store.add_chunks(chunks, embeddings)
        return success_response(data={
            "chunks_ingested": len(chunks),
            "total_chunks": len(_ks_mod.knowledge_store),
        })

    # /api/v1/tutor/ask
    @agent_app.endpoint("/api/v1/tutor/ask", methods=["POST"])
    @AS["trace"](trace_type=AS["TraceType"].LLM, trace_name="tutor_ask", is_root_span=True)
    async def tutor_ask(request: Request, body: dict):
        question = (body.get("question") or "").strip()
        if not question:
            return JSONResponse(status_code=400, content={"error": "question is required"})
        dispatcher = _disp_mod.TutorDispatcher(
            tools=_tc_mod.build_tool_registry(),
            max_iterations=int(config.get("MAX_DISPATCH_ITERATIONS", 5)),
            max_llm_calls=int(config.get("MAX_DISPATCH_LLM_CALLS", 8)),
        )
        result = await dispatcher.run(
            question=question,
            user_id=str(body.get("user_id") or ""),
            course_id=str(body.get("course_id") or ""),
            history=body.get("history") or [],
        )
        return success_response(data=result)

    # /debug/config
    @agent_app.endpoint("/debug/config")
    def debug_config(request: Request):
        raw_key = os.getenv("DASHSCOPE_API_KEY") or ""
        masked = (raw_key[:8] + "****" + raw_key[-4:]) if len(raw_key) > 12 else ("<empty>" if not raw_key else "<too short>")
        return success_response(data={
            "env": {
                "DASHSCOPE_MODEL_CODE": os.getenv("DASHSCOPE_MODEL_CODE") or "<not set>",
                "DASHSCOPE_API_KEY_masked": masked,
                "DASHSCOPE_API_URL": os.getenv("DASHSCOPE_API_URL") or "<not set>",
                "MODEL_PROVIDER": os.getenv("MODEL_PROVIDER") or "<not set>",
                "BAILIAN_APP_ENV": os.getenv("BAILIAN_APP_ENV") or "<not set>",
            },
            "config_yml": {
                "MODEL_PROVIDER": config.get("MODEL_PROVIDER"),
                "DASHSCOPE_MODEL_CODE": config.get("DASHSCOPE_MODEL_CODE"),
                "DASHSCOPE_API_URL": config.get("DASHSCOPE_API_URL"),
            },
            "effective": {
                "provider": os.getenv("MODEL_PROVIDER") or config.get("MODEL_PROVIDER") or "third_party",
                "model": os.getenv("DASHSCOPE_MODEL_CODE") or config.get("DASHSCOPE_MODEL_CODE", "qwen-plus"),
                "url": os.getenv("DASHSCOPE_API_URL") or config.get("DASHSCOPE_API_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "has_api_key": bool(os.getenv("DASHSCOPE_API_KEY")),
            },
        })

    # ---------------------------------------------------------------
    # ReActAgent chat handler — the core AgentScope query function
    # ---------------------------------------------------------------
    @agent_app.query(framework="agentscope")
    @AS["trace"](trace_type=AS["TraceType"].LLM, trace_name="llm_func", is_root_span=True)
    async def query_func(self, msgs, request: "AgentRequest" = None, response=None, **kwargs):
        AS = _get_agentscope()
        session_id = request.session_id
        user_id = request.user_id
        access_source = (request.model_extra or {}).get("access_source") or "api"

        current_span = ot_trace.get_current_span()
        if current_span and current_span.is_recording():
            current_span.set_attribute("bailian.app.session_id", session_id or "")
            current_span.set_attribute("bailian.app.metric.source", access_source)
            current_span.set_attribute("bailian.app.agent.ip", LOCAL_IP)
        AS["TracingUtil"].set_common_attributes({"bailian.app.session_id": session_id or ""})
        AS["TracingUtil"].set_common_attributes({"bailian.app.metric.source": access_source})
        AS["TracingUtil"].set_common_attributes({"bailian.app.agent.ip": LOCAL_IP})

        state = await self.state_service.export_state(session_id=session_id, user_id=user_id)

        # ---- Lumen toolkit (5 sub-agent tools) ----
        toolkit = AS["Toolkit"]()
        toolkit.register_tool_function(_agent_tools_mod.tool_retrieve)
        toolkit.register_tool_function(_agent_tools_mod.tool_web_search)
        toolkit.register_tool_function(_agent_tools_mod.tool_run_code)
        toolkit.register_tool_function(_agent_tools_mod.tool_generate_quiz)
        toolkit.register_tool_function(_agent_tools_mod.tool_explain_concept)

        # ---- Model: third_party (DashScope compatible) ----
        model_provider = (
            os.getenv("MODEL_PROVIDER") or config.get("MODEL_PROVIDER") or "third_party"
        ).lower()

        if model_provider == "ai_studio":
            vpc_url = os.getenv("DASHSCOPE_API_URL") or config.get("VPC_OPENAI_API_URL") or ""
            vpc_key = os.getenv("DASHSCOPE_API_KEY") or config.get("VPC_OPENAI_API_KEY") or ""
            vpc_model = os.getenv("DASHSCOPE_MODEL_CODE") or config.get("VPC_OPENAI_MODEL") or ""
            for suffix in ("/chat/completions", "/completions", "/responses", "/embeddings"):
                if vpc_url.endswith(suffix):
                    vpc_url = vpc_url[: -len(suffix)]
                    break
            vpc_url = vpc_url.rstrip("/")
            proxy = os.getenv("VPC_HTTP_PROXY") or config.get("VPC_HTTP_PROXY") or None
            if (os.getenv("BAILIAN_APP_ENV") or "").lower() == "prod":
                proxy = None
            kwargs2: dict[str, Any] = {"base_url": vpc_url}
            if proxy:
                try:
                    kwargs2["http_client"] = httpx.AsyncClient(proxy=proxy, timeout=httpx.Timeout(60.0, connect=10.0))
                except TypeError:
                    kwargs2["http_client"] = httpx.AsyncClient(proxies=proxy, timeout=httpx.Timeout(60.0, connect=10.0))
            model_obj = AS["OpenAIChatModel"](model_name=vpc_model, api_key=vpc_key, stream=True, client_kwargs=kwargs2)
            formatter_obj = AS["OpenAIChatFormatter"]()
        else:
            model_name = os.getenv("DASHSCOPE_MODEL_CODE") or config.get("DASHSCOPE_MODEL_CODE", "qwen-plus")
            api_key = os.getenv("DASHSCOPE_API_KEY") or config.get("DASHSCOPE_API_KEY") or ""
            api_url = (
                os.getenv("DASHSCOPE_API_URL")
                or config.get("DASHSCOPE_API_URL")
                or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            for suffix in ("/chat/completions", "/completions", "/responses", "/embeddings"):
                if api_url.endswith(suffix):
                    api_url = api_url[: -len(suffix)]
                    break
            api_url = api_url.rstrip("/")
            model_obj = AS["OpenAIChatModel"](
                model_name=model_name,
                api_key=api_key,
                stream=True,
                client_kwargs={"base_url": api_url},
            )
            formatter_obj = AS["OpenAIChatFormatter"]()

        agent = AS["ReActAgent"](
            name="Lumen",
            model=model_obj,
            sys_prompt=_prompts_mod.MAIN_AGENT_PROMPT,
            toolkit=toolkit,
            memory=AS["AgentScopeSessionHistoryMemory"](
                service=self.session_service,
                session_id=session_id,
                user_id=user_id,
            ),
            formatter=formatter_obj,
        )

        if state:
            agent.load_state_dict(state)

        stream_printing_messages = _get_stream_func()
        async for msg, last in stream_printing_messages(
            agents=[agent],
            coroutine_task=agent(msgs),
        ):
            yield msg, last

        await self.state_service.save_state(
            user_id=user_id,
            session_id=session_id,
            state=agent.state_dict(),
        )

    return agent_app


# ============================================================
# Main entry point (platform only)
# ============================================================
async def main():
    AS = _get_agentscope()
    deployer = AS["LocalDeployManager"](
        host=config.get("START_HOST", "0.0.0.0"),
        port=int(config.get("PORT", 8080)),
    )
    app = _create_agent_app()
    await app.deploy(deployer)
    print("Lumen-Bailian Agent started. Press Ctrl+C to stop...")
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping service...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nService stopped")
