"""Agentscope-free core logic for the five tutor sub-agents.

The dispatcher (dispatcher.py) calls these functions directly so the whole
tutoring turn stays under a single LLM budget. Each LLM-consuming function
accepts an optional ``budget`` (LlmBudget) and charges it before calling the
model. This module has no AgentScope dependency so it can be unit-tested
locally.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

try:
    from .code_runner import run_python
    from .knowledge_store import embedding_provider, knowledge_store
    from .llm_utils import LlmBudget, chat_completion, get_config
except ImportError:  # Direct execution / local tests
    from code_runner import run_python
    from knowledge_store import embedding_provider, knowledge_store
    from llm_utils import LlmBudget, chat_completion, get_config


def _charge(budget: LlmBudget | None) -> bool:
    """Charge one LLM call against the budget; False when exhausted."""
    if budget is None:
        return True
    if not budget.can_charge(1):
        return False
    budget.charge(1)
    return True


# ---------------------------------------------------------------------------
# 1. Retriever (0 LLM calls — embedding + cosine search only)
# ---------------------------------------------------------------------------
async def retrieve_course(
    query: str,
    top_k: int = 5,
    course_id: str = "",
    budget: LlmBudget | None = None,
) -> str:
    """Search the course knowledge base; returns structured JSON results."""
    query = (query or "").strip()
    if not query:
        return json.dumps({"results": [], "note": "检索词为空。"}, ensure_ascii=False)

    try:
        query_emb = embedding_provider.embed([query])[0]
        results = knowledge_store.search(query_emb, top_k=top_k, course_id=course_id or "")
    except Exception as exc:
        return json.dumps(
            {"results": [], "error": f"检索失败: {exc}"},
            ensure_ascii=False,
        )

    if not results:
        return json.dumps(
            {"results": [], "note": "未找到相关课程资料，请换关键词重试。"},
            ensure_ascii=False,
        )

    items = []
    for i, (chunk, score) in enumerate(results, start=1):
        items.append({
            "ref": i,
            "lesson_id": chunk.lesson_id,
            "lesson_title": chunk.lesson_title,
            "course_title": chunk.course_title,
            "similarity": round(score, 4),
            "text": chunk.text[:500],
        })
    return json.dumps({"results": items}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 2. Web Searcher
#
# 已尝试的方案及平台限制：
#
# 方案 A — DashScope enable_search=True（联网搜索，模型自带）
#   payload["extra_body"] = {"enable_search": True}
#   状态：❌ 平台 AgentScope Java 运行时对所有 LLM 调用施加 10 秒硬超时，
#         联网搜索无法在期限内完成，始终报错
#         "Did not observe any item...within 10000ms in 'flatMap'"
#
# 方案 B — Tavily（免费 key，https://app.tavily.com）
#   POST https://api.tavily.com/search  {api_key, query, search_depth, max_results}
#   状态：❌ 百炼平台未配置 TAVILY_API_KEY，返回 "TVILY_API_KEY not configured"
#
# 方案 C — DuckDuckGo Instant Answer（免 key，https://api.duckduckgo.com）
#   GET https://api.duckduckgo.com/?q={query}&format=json
#   状态：❌ 国内网络环境 ConnectTimeout，无法访问
#
# 方案 D — 直接 httpx 调 DashScope（绕过 AgentScope）
#   状态：❌ 平台统一路由，所有 LLM 请求均经过 Java 运行时超时拦截，
#         enable_search=True 仍触发 10s 超时
#
# 方案 E（当前采用）— 模型预训练知识
#   直接让 LLM 回答，不带 enable_search，不发往 AgentScope
#   状态：✅ 正常工作，绕过 10s 超时
#   缺点：依赖模型训练数据（Qwen-plus 截至 2024 年底），无法获取实时信息
#
# 如需启用上述任一方案，只需取消对应 payload 的注释，并注释掉方案 E。
# ---------------------------------------------------------------------------

async def search_web(query: str, budget: LlmBudget | None = None) -> str:
    """Answer using the model's built-in knowledge (no external search).

    The deployed model (Qwen-plus) has knowledge up to late 2024 and can answer
    most learning-related questions without external search.
    Completely bypasses the 10s Java-runtime timeout.
    """
    query = (query or "").strip()
    if not query:
        return "[联网搜索：缺少查询词]"

    if budget is None or _charge(budget):
        return await _model_knowledge_search(query)
    return "[联网搜索：LLM 调用预算已耗尽，请减少问题数量]"


async def _model_knowledge_search(query: str) -> str:
    """Use the LLM's pre-trained knowledge to answer without any external call."""
    model = get_config("DASHSCOPE_MODEL_CODE", "qwen-plus")
    api_key = get_config("DASHSCOPE_API_KEY")
    api_url = get_config(
        "DASHSCOPE_API_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    url = f"{api_url.rstrip('/')}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"你是一位知识渊博的AI助教。请针对「{query}」给出准确、"
                    "条理清晰的中文回答，结合你的知识库尽量全面。如果问题涉及"
                    "最新事件或你不确定的信息，请明确告知。回答需要有结构，"
                    "适当分点，适当举例，便于学习者理解。"
                ),
            }
        ],
        "temperature": 0.3,
        "max_tokens": 1536,
    }

    # === 方案 A（已废弃）：DashScope 联网搜索 ===
    # 平台 Java 运行时 10s 超时，开通后将 extra_body 取消注释即可启用
    # payload["extra_body"] = {"enable_search": True}

    # === 方案 B（已废弃）：Tavily 真联网搜索 ===
    # 百炼平台未配置 key，如需启用：
    # 1. 在 https://app.tavily.com 注册免费 key
    # 2. 在平台环境变量中添加 TAVILY_API_KEY
    # 3. 取消下方 _tavily_search 的注释，并将 search_web 改为调用它

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0)
        ) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        if not resp.is_success:
            return f"[联网搜索失败: HTTP {resp.status_code}，模型知识回答也失败]"
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if content and content.strip():
            return content.strip()
    except Exception:
        pass

    return (
        "[联网搜索暂时不可用，且模型回答也失败。"
        "建议：1) 尝试检索课程内容 /retrieve；2) 换关键词重试]"
    )


# === 方案 B 备用（取消注释即可启用）===
# async def _tavily_search(query: str) -> str | None:
#     """Search via Tavily API. Returns None if key is missing or call fails."""
#     tavily_key = get_config("TAVILY_API_KEY")
#     if not tavily_key:
#         return None
#     url = get_config("TAVILY_API_URL", "https://api.tavily.com/search")
#     payload = {
#         "api_key": tavily_key,
#         "query": query,
#         "search_depth": "basic",
#         "max_results": 5,
#         "include_answer": True,
#     }
#     try:
#         async with httpx.AsyncClient(timeout=30.0) as client:
#             resp = await client.post(url, json=payload)
#         if not resp.is_success:
#             return None
#         data = resp.json()
#         answer = (data.get("answer") or "").strip()
#         results = data.get("results") or []
#         if not answer and not results:
#             return None
#         lines = []
#         if answer:
#             lines.append(answer)
#         if results:
#             lines.append("")
#             lines.append("来源：")
#             for i, r in enumerate(results[:5], 1):
#                 title = (r.get("title") or "").strip()
#                 link = (r.get("url") or "").strip()
#                 content = (r.get("content") or "").strip()
#                 lines.append(f"{i}. {title} ({link})")
#                 if content:
#                     lines.append(f"   {content[:300]}")
#         return "\n".join(lines)
#     except Exception:
#         return None


# ---------------------------------------------------------------------------
# 3. Code Runner (0 LLM calls — sandboxed subprocess)
# ---------------------------------------------------------------------------
async def run_code(code: str, budget: LlmBudget | None = None) -> str:
    """Execute learner code in the restricted sandbox; returns structured JSON."""
    return json.dumps(run_python(code), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 4. Quiz Generator (1 LLM call)
# ---------------------------------------------------------------------------
async def generate_quiz(
    topic: str,
    difficulty: str = "medium",
    budget: LlmBudget | None = None,
) -> str:
    """Generate practice questions for a topic."""
    topic = (topic or "").strip()
    if not topic:
        return "[生成测验：缺少主题]"

    if not _charge(budget):
        return "[LLM 调用预算已耗尽，无法生成测验]"

    prompt = (
        f"你是一位专业的出题老师。请针对「{topic}」生成 3 道{difficulty}难度的练习题。\n\n"
        "要求：\n"
        "1. 题目类型多样化（选择题、判断题、简答题各一）\n"
        "2. 每道题都要有答案解析\n"
        "3. 难度适配：easy=基础概念，medium=理解应用，hard=综合分析\n"
        "4. 用中文出题\n\n"
        "请以 JSON 数组格式输出，每道题包含 type, question, options(选择题), "
        "answer, explanation 字段。"
    )
    try:
        return await chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
    except Exception as exc:
        return f"[生成测验时出错：{exc}]"


# ---------------------------------------------------------------------------
# 5. Concept Explainer (1 LLM call)
# ---------------------------------------------------------------------------
async def explain_concept(
    concept: str,
    level: str = "beginner",
    budget: LlmBudget | None = None,
) -> str:
    """Explain a concept with layered explanation."""
    concept = (concept or "").strip()
    if not concept:
        return "[讲解概念：缺少概念名]"

    if not _charge(budget):
        return "[LLM 调用预算已耗尽，无法讲解概念]"

    prompt = (
        f"请向一位{level}水平的学生讲解「{concept}」这个概念。\n\n"
        "请按以下结构分层讲解：\n"
        "1. **一句话总结**：用最简单的一句话概括这个概念\n"
        "2. **生活类比**：用一个生活中的例子或类比帮助理解\n"
        "3. **深入细节**：讲解关键原理、要点和注意事项\n"
        "4. **关键要点**：列出 3-5 个需要记住的核心要点\n\n"
        "用中文回答，语气友好、耐心。"
    )
    try:
        return await chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=2048,
        )
    except Exception as exc:
        return f"[解释概念时出错：{exc}]"


# ---------------------------------------------------------------------------
# Tool registry for the dispatcher
# ---------------------------------------------------------------------------
def build_tool_registry() -> dict[str, Any]:
    """Map sub-agent names to their async core functions (all accept budget)."""
    return {
        "retrieve": retrieve_course,
        "web_search": search_web,
        "run_code": run_code,
        "generate_quiz": generate_quiz,
        "explain_concept": explain_concept,
    }
