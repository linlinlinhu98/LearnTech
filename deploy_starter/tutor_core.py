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
# 2. Web Searcher (0 LLM calls — live Tavily search API)
# ---------------------------------------------------------------------------
async def search_web(query: str, budget: LlmBudget | None = None) -> str:
    """Search the live web via Tavily; returns answer + cited sources.

    DashScope enable_search is NOT used — it routes through the platform's
    Java runtime which has a hard 10-second HTTP timeout, causing
    "Did not observe any item..." errors when search takes longer.
    Tavily calls httpx directly (outside the Java sandbox), no such limit.

    budget is accepted for interface consistency (Tavily is free, 0 LLM cost).
    """
    query = (query or "").strip()
    if not query:
        return "[联网搜索：缺少查询词]"

    tavily_key = get_config("TAVILY_API_KEY")
    if not tavily_key:
        return (
            "[联网搜索暂不可用：未配置 TAVILY_API_KEY。"
            " 请在平台环境变量中添加 Tavily API key（免费 https://app.tavily.com），"
            " 或尝试检索课程内容 /retrieve]"
        )

    url = get_config("TAVILY_API_URL", "https://api.tavily.com/search")
    payload = {
        "api_key": tavily_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": True,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
        if not resp.is_success:
            return f"[联网搜索失败: HTTP {resp.status_code}]"
        data = resp.json()
    except httpx.TimeoutException:
        return "[联网搜索超时，请稍后重试]"
    except Exception as exc:
        return f"[联网搜索失败: {exc}]"

    answer = (data.get("answer") or "").strip()
    results = data.get("results") or []
    if not answer and not results:
        return "[联网搜索：未找到相关结果]"

    lines: list[str] = []
    if answer:
        lines.append(answer)
    if results:
        lines.append("")
        lines.append("来源：")
        for i, r in enumerate(results[:5], start=1):
            title = (r.get("title") or "").strip()
            link = (r.get("url") or "").strip()
            content = (r.get("content") or "").strip()
            lines.append(f"{i}. {title} ({link})")
            if content:
                lines.append(f"   {content[:300]}")
    return "\n".join(lines)


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
