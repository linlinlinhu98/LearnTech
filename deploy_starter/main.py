"""Lumen-Bailian — AI-Powered Personalized Learning Path Planner.

Built on the AgentScope Bailian reference template.
Adds RAG-powered tutoring with course search, quiz generation,
and concept explanation tools.
"""

import asyncio
import logging
import os
import socket
import time
import uuid
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse

from fastapi import Request
from fastapi.responses import JSONResponse
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.model import DashScopeChatModel
from agentscope.pipeline import stream_printing_messages
import aiohttp
import httpx
from agentscope.tool import Toolkit, ToolResponse
from agentscope.tool._response import TextBlock
from agentscope_runtime.adapters.agentscope.memory import AgentScopeSessionHistoryMemory
from agentscope_runtime.engine import AgentApp, LocalDeployManager
from agentscope_runtime.engine.schemas.agent_schemas import AgentRequest
from agentscope_runtime.engine.services.agent_state import InMemoryStateService
from agentscope_runtime.engine.services.session_history import (
    InMemorySessionHistoryService,
)
from agentscope_runtime.engine.tracing import TraceType, trace, TracingUtil
from opentelemetry import trace as ot_trace
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
import agentscope

# ---- Lumen tool imports (dual-mode: package vs platform direct execution) ----
try:
    from .agent_tools import (
        tool_explain_concept,
        tool_generate_quiz,
        tool_retrieve,
        tool_run_code,
        tool_web_search,
    )
    from .knowledge_store import knowledge_store, embedding_provider, seed_demo_data, LessonChunk
    from .prompts import MAIN_AGENT_PROMPT
    from .tutor_core import build_tool_registry
    from .dispatcher import TutorDispatcher
except ImportError:
    from agent_tools import (
        tool_explain_concept,
        tool_generate_quiz,
        tool_retrieve,
        tool_run_code,
        tool_web_search,
    )
    from knowledge_store import knowledge_store, embedding_provider, seed_demo_data, LessonChunk
    from prompts import MAIN_AGENT_PROMPT
    from tutor_core import build_tool_registry
    from dispatcher import TutorDispatcher

# ============================================================
# Tracing setup (from reference template)
# ============================================================
if os.getenv("TRACE_ENABLE_REPORT", "").lower() in ("true", "1", "yes"):
    agentscope._config.trace_enabled = True

    from opentelemetry.sdk.trace import TracerProvider as _SdkTracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPSpanGrpcExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPSpanHttpExporter,
    )

    if not isinstance(ot_trace.get_tracer_provider(), _SdkTracerProvider):
        _resource = Resource(attributes={
            SERVICE_NAME: os.getenv("SERVICE_NAME", "agentscope_runtime"),
            SERVICE_VERSION: os.getenv("SERVICE_VERSION", "1.0.0"),
            "source": "agentscope_runtime-source",
        })
        _provider = _SdkTracerProvider(resource=_resource)

        _trace_endpoint = os.getenv("TRACE_ENDPOINT", "")
        if _trace_endpoint:
            _exporter = OTLPSpanGrpcExporter(
                endpoint=_trace_endpoint,
                insecure=True,
                headers=f"Authentication={os.getenv('TRACE_AUTHENTICATION', '')}",
            )
            _provider.add_span_processor(BatchSpanProcessor(_exporter))

        if os.getenv("TRACE_ENABLE_DEBUG", "").lower() in ("true", "1", "yes"):
            _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        ot_trace.set_tracer_provider(_provider)

    import agentscope_runtime.engine.tracing.wrapper as _rt_wrapper
    _rt_wrapper._otel_tracer = ot_trace.get_tracer("agentscope_runtime")


# ============================================================
# Config reader (from reference template)
# ============================================================
def read_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
                    elif value.isdigit():
                        value = int(value)
                    config[key] = value
    return config


config = read_config()

# ============================================================
# Logging setup (from reference template)
# ============================================================
LOG_DIR = os.getenv("APP_LOG_DIR", "/home/admin/app_logs")
os.makedirs(LOG_DIR, exist_ok=True)

_log_level = (os.getenv("LOG_LEVEL") or config.get("LOG_LEVEL", "INFO")).upper()

logger = logging.getLogger("agent_app")
logger.setLevel(_log_level)

if not logger.handlers:
    _formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    log_name = os.getenv("LOG_FILE_NAME") or config.get("LOG_FILE_NAME", "app.log")
    _file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_name),
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    _file_handler.setFormatter(_formatter)
    logger.addHandler(_file_handler)

    _stream_handler = logging.StreamHandler()
    _stream_handler.setFormatter(_formatter)
    logger.addHandler(_stream_handler)


# ============================================================
# Helpers (from reference template)
# ============================================================
def get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


LOCAL_IP = get_local_ip()


def success_response(data=None, message="success"):
    return {"code": 200, "message": message, "data": data, "host": LOCAL_IP}


def error_response(message="error", code=500, data=None):
    return {"code": code, "message": message, "data": data, "host": LOCAL_IP}


agent_app = AgentApp(
    app_name=config.get("APP_NAME"),
    app_description="AI-Powered Personalized Learning Path Planner with RAG.",
)


# ============================================================
# Lifecycle (from reference template, with Lumen seeding)
# ============================================================
@agent_app.init
async def init_func(self):
    self.state_service = InMemoryStateService()
    self.session_service = InMemorySessionHistoryService()
    await self.state_service.start()
    await self.session_service.start()
    # Lumen: seed demo course data so knowledge store is never empty
    try:
        seed_demo_data()
        logger.info("Lumen knowledge store seeded: %s chunks", len(knowledge_store))
    except Exception as exc:
        logger.warning("Demo data seeding failed (non-fatal): %s", exc)


@agent_app.shutdown
async def shutdown_func(self):
    await self.state_service.stop()
    await self.session_service.stop()


# ============================================================
# Standard Bailian endpoints (from reference template)
# ============================================================
@agent_app.endpoint("/")
@trace(trace_type=TraceType.LLM, trace_name="llm_func", is_root_span=True)
def read_root():
    return {"name": config.get("APP_NAME", "Lumen-Bailian"), "status": "running"}


@agent_app.endpoint("/health")
@trace(trace_type=TraceType.LLM, trace_name="llm_func", is_root_span=True)
def health_check():
    return "OK"


@agent_app.endpoint("/createSession")
@trace(trace_type=TraceType.OTHER, trace_name="create_session", is_root_span=True)
def create_session(request: Request, body: dict):
    unique_code = str(uuid.uuid4())
    return {"uniqueCode": unique_code, **success_response()}


def set_custome_trace(body: dict):
    session_id = body.get("session_id")
    access_source = body.get("access_source", "api")
    user_id = body.get("user_id")
    current_span = ot_trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute("bailian.app.session_id", session_id or "")
        current_span.set_attribute("bailian.app.metric.source", access_source or "")
        current_span.set_attribute("bailian.app.agent.ip", LOCAL_IP or "")
    TracingUtil.set_common_attributes({"bailian.app.session_id": session_id or ""})
    TracingUtil.set_common_attributes({"bailian.app.metric.source": access_source or ""})
    TracingUtil.set_common_attributes({"bailian.app.agent.ip": LOCAL_IP or ""})


@agent_app.endpoint("/clearSession")
@trace(trace_type=TraceType.OTHER, trace_name="clear_session", is_root_span=True)
def clear_session(request: Request, body: dict):
    set_custome_trace(body)
    return success_response(message="session_id cleared")


@agent_app.endpoint("/abortSession")
@trace(trace_type=TraceType.OTHER, trace_name="abort_session", is_root_span=True)
def abort_session(request: Request, body: dict):
    set_custome_trace(body)
    return success_response(message="session_id aborted")


# ============================================================
# /process/sync (from reference template)
# ============================================================
SYNC_DEFAULT_TIMEOUT = int(os.getenv("SYNC_TIMEOUT_SECONDS", "600"))
SYNC_MAX_TIMEOUT = SYNC_DEFAULT_TIMEOUT * 5


def _parse_sync_timeout(headers) -> int:
    raw = headers.get("x-sync-timeout-seconds") or headers.get("X-Sync-Timeout-Seconds")
    if not raw:
        return SYNC_DEFAULT_TIMEOUT
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return SYNC_DEFAULT_TIMEOUT
    if v <= 0:
        return SYNC_DEFAULT_TIMEOUT
    return min(v, SYNC_MAX_TIMEOUT)


@agent_app.endpoint("/process/sync", methods=["POST"])
@trace(trace_type=TraceType.LLM, trace_name="process_sync", is_root_span=True)
async def process_sync(request: Request, body: dict):
    set_custome_trace(body)
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not ready", "message": "Runner not initialized"},
        )

    timeout = _parse_sync_timeout(request.headers)
    started_at = time.monotonic()
    final_response = None
    session_id = None

    try:
        async with asyncio.timeout(timeout):
            async for ev in runner.stream_query(body):
                if getattr(ev, "object", None) == "response":
                    final_response = ev
                    session_id = getattr(ev, "session_id", None) or session_id
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "error": "Timeout",
                "message": f"exceeded {timeout}s",
                "session_id": session_id,
            },
            headers={"X-Session-Id": session_id or ""},
        )
    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={"error": "Service not ready", "message": str(e)},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "message": str(e)},
        )

    if final_response is None:
        return JSONResponse(
            status_code=500,
            content={"error": "No response frame produced"},
        )

    latency_ms = int((time.monotonic() - started_at) * 1000)
    payload = final_response.model_dump(mode="json")
    payload["host"] = LOCAL_IP
    return JSONResponse(
        content=payload,
        headers={
            "X-Session-Id": str(payload.get("session_id") or ""),
            "X-Sync-Latency-Ms": str(latency_ms),
        },
    )


# ============================================================
# Lumen Knowledge Management Endpoints
# ============================================================

@agent_app.endpoint("/api/v1/knowledge/search", methods=["POST"])
@trace(trace_type=TraceType.OTHER, trace_name="knowledge_search", is_root_span=True)
async def knowledge_search(request: Request, body: dict):
    query = (body.get("query") or "").strip()
    if not query:
        return JSONResponse(status_code=400, content={"error": "query is required"})
    try:
        query_emb = embedding_provider.embed([query])[0]
        results = knowledge_store.search(query_emb, top_k=body.get("top_k", 5))
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
@trace(trace_type=TraceType.OTHER, trace_name="knowledge_ingest", is_root_span=True)
async def knowledge_ingest(request: Request, body: dict):
    raw_chunks = body.get("chunks", [])
    if not raw_chunks:
        return JSONResponse(status_code=400, content={"error": "chunks is required"})

    chunks = []
    texts = []
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
    return success_response(data={
        "chunks_ingested": len(chunks),
        "total_chunks": len(knowledge_store),
    })


@agent_app.endpoint("/api/v1/knowledge/stats")
@trace(trace_type=TraceType.OTHER, trace_name="knowledge_stats", is_root_span=True)
def knowledge_stats(request: Request):
    return success_response(data={
        "total_chunks": len(knowledge_store),
        "embedding_model": embedding_provider.model_id,
        "embedding_dim": embedding_provider.dim,
    })


# ============================================================
# Lumen Multi-Agent Tutor Endpoint (Module 3)
# ============================================================

@agent_app.endpoint("/api/v1/tutor/ask", methods=["POST"])
@trace(trace_type=TraceType.LLM, trace_name="tutor_ask", is_root_span=True)
async def tutor_ask(request: Request, body: dict):
    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse(status_code=400, content={"error": "question is required"})

    user_id = str(body.get("user_id") or "")
    course_id = str(body.get("course_id") or "")
    history = body.get("history") or []
    if not isinstance(history, list):
        history = []

    dispatcher = TutorDispatcher(
        tools=build_tool_registry(),
        max_iterations=int(config.get("MAX_DISPATCH_ITERATIONS", 5)),
        max_llm_calls=int(config.get("MAX_DISPATCH_LLM_CALLS", 8)),
    )
    result = await dispatcher.run(
        question=question,
        user_id=user_id,
        course_id=course_id,
        history=history,
    )
    return success_response(data=result)


# ============================================================
# Diagnostic endpoint - check platform-injected configuration
# ============================================================

@agent_app.endpoint("/debug/config")
def debug_config(request: Request):
    """Show current model configuration (keys masked) for debugging."""
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


# ============================================================
# Agent query - Lumen RAG tutor (model provider from reference)
# ============================================================

@agent_app.query(framework="agentscope")
@trace(trace_type=TraceType.LLM, trace_name="llm_func", is_root_span=True)
async def query_func(
    self,
    msgs,
    request: AgentRequest = None,
    response=None,
    **kwargs,
):
    session_id = request.session_id
    logger.info("[query_func] session_id=%s, host=%s", session_id, LOCAL_IP)
    access_source = request.model_extra.get("access_source") or 'api'
    user_id = request.user_id

    current_span = ot_trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute("bailian.app.session_id", session_id or "")
        current_span.set_attribute("bailian.app.metric.source", access_source or "")
        current_span.set_attribute("bailian.app.agent.ip", LOCAL_IP or "")
    TracingUtil.set_common_attributes({"bailian.app.session_id": session_id or ""})
    TracingUtil.set_common_attributes({"bailian.app.metric.source": access_source or ""})
    TracingUtil.set_common_attributes({"bailian.app.agent.ip": LOCAL_IP or ""})

    state = await self.state_service.export_state(
        session_id=session_id,
        user_id=user_id,
    )

    # ---- Lumen Toolkit ----
    toolkit = Toolkit()
    toolkit.register_tool_function(tool_retrieve)
    toolkit.register_tool_function(tool_web_search)
    toolkit.register_tool_function(tool_run_code)
    toolkit.register_tool_function(tool_generate_quiz)
    toolkit.register_tool_function(tool_explain_concept)

    # ---- Model provider switch (from reference template) ----
    model_provider = (
        os.getenv("MODEL_PROVIDER")
        or config.get("MODEL_PROVIDER")
        or "third_party"
    ).lower()

    if model_provider == "ai_studio":
        vpc_api_url = os.getenv("DASHSCOPE_API_URL") or config.get("VPC_OPENAI_API_URL")
        vpc_api_key = os.getenv("DASHSCOPE_API_KEY") or config.get("VPC_OPENAI_API_KEY")
        vpc_model = os.getenv("DASHSCOPE_MODEL_CODE") or config.get("VPC_OPENAI_MODEL", "")

        for _suffix in ("/chat/completions", "/completions", "/responses", "/embeddings"):
            if vpc_api_url and vpc_api_url.endswith(_suffix):
                vpc_api_url = vpc_api_url[: -len(_suffix)]
                break
        if vpc_api_url:
            vpc_api_url = vpc_api_url.rstrip("/")
        vpc_http_proxy = os.getenv("VPC_HTTP_PROXY") or config.get("VPC_HTTP_PROXY")

        bailian_app_env = (os.getenv("BAILIAN_APP_ENV") or config.get("BAILIAN_APP_ENV") or "").lower()
        if bailian_app_env == "prod":
            vpc_http_proxy = None

        openai_client_kwargs = {"base_url": vpc_api_url}
        if vpc_http_proxy:
            try:
                _http_client = httpx.AsyncClient(
                    proxy=vpc_http_proxy,
                    timeout=httpx.Timeout(60.0, connect=10.0),
                )
            except TypeError:
                _http_client = httpx.AsyncClient(
                    proxies=vpc_http_proxy,
                    timeout=httpx.Timeout(60.0, connect=10.0),
                )
            openai_client_kwargs["http_client"] = _http_client

        logger.info(
            "[OpenAIChatModel/ai_studio] base_url=%s, model=%r, proxy=%s",
            vpc_api_url, vpc_model, vpc_http_proxy or "<none>",
        )

        model_obj = OpenAIChatModel(
            model_name=vpc_model,
            api_key=vpc_api_key,
            stream=True,
            client_kwargs=openai_client_kwargs,
        )
        formatter_obj = OpenAIChatFormatter()
    else:
        model_name = os.getenv("DASHSCOPE_MODEL_CODE") or config.get("DASHSCOPE_MODEL_CODE", "qwen-plus")
        api_key = os.getenv("DASHSCOPE_API_KEY") or config.get("DASHSCOPE_API_KEY")
        api_url = (
            os.getenv("DASHSCOPE_API_URL")
            or config.get("DASHSCOPE_API_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        for _suffix in ("/chat/completions", "/completions", "/responses", "/embeddings"):
            if api_url.endswith(_suffix):
                api_url = api_url[: -len(_suffix)]
                break
        api_url = api_url.rstrip("/")

        logger.info(
            "[OpenAIChatModel/third_party] base_url=%s, model=%s, key=%s",
            api_url, model_name, (api_key[:6] + "***") if api_key else "<empty>",
        )

        model_obj = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=True,
            client_kwargs={"base_url": api_url},
        )
        formatter_obj = OpenAIChatFormatter()

    # ---- Lumen ReActAgent ----
    agent = ReActAgent(
        name="Lumen",
        model=model_obj,
        sys_prompt=MAIN_AGENT_PROMPT,
        toolkit=toolkit,
        memory=AgentScopeSessionHistoryMemory(
            service=self.session_service,
            session_id=session_id,
            user_id=user_id,
        ),
        formatter=formatter_obj,
    )

    if state:
        agent.load_state_dict(state)

    async for msg, last in stream_printing_messages(
        agents=[agent],
        coroutine_task=agent(msgs),
    ):
        yield msg, last

    state = agent.state_dict()

    await self.state_service.save_state(
        user_id=user_id,
        session_id=session_id,
        state=state,
    )


@trace(trace_type=TraceType.OTHER, trace_name="testObservability", is_root_span=True)
def testObservability():
    print("testObservability")


# ============================================================
# Main entry point (from reference template)
# ============================================================
async def main():
    deployer = LocalDeployManager(
        host=config.get("START_HOST", "127.0.0.1"),
        port=config.get("PORT", 8080),
    )
    testObservability()

    await agent_app.deploy(deployer)

    print("Lumen-Bailian Agent started, press Ctrl+C to stop...")
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        print("\nStopping service...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nService stopped")
