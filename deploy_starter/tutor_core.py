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
# 3. Session State (in-memory, per-user tracking for review & mastery)
# ---------------------------------------------------------------------------
import time

# Key: f"{user_id}:{chunk_id}" → {"S": float, "last_review": float, "retrievals": int, "correct": int, "incorrect": int}
_review_state: dict[str, dict] = {}

# Key: f"{user_id}" → {"exam_date": float|None, "focus_course_id": str}
_session_meta: dict[str, dict] = {}

# Key: f"{user_id}" → set of chunk_ids that are "weak" (连续答错)
_weak_chunks: dict[str, set[str]] = {}

# Key: f"{user_id}" → list of (chunk_id, difficulty) for current mock exam
_mock_queue: dict[str, list] = {}

# Key: f"{user_id}" → {"state": str, "results": list, "current_index": int, "llm_used": int}
# state: "idle" | "active" | "finished"
_mock_session: dict[str, dict] = {}

# Track consecutive correct answers per weak chunk: f"{user_id}:{chunk_id}" → streak count
_mock_streak: dict[str, int] = {}


def _review_key(user_id: str, chunk_id: str) -> str:
    return f"{user_id}:{chunk_id}"


def _get_review_state(user_id: str, chunk_id: str) -> dict:
    return _review_state.get(_review_key(user_id, chunk_id), {
        "S": 1.0,
        "last_review": time.time(),
        "retrievals": 0,
        "correct": 0,
        "incorrect": 0,
    })


def _save_review_state(user_id: str, chunk_id: str, state: dict) -> None:
    _review_state[_review_key(user_id, chunk_id)] = state


def _set_session_exam(user_id: str, exam_date: str | None) -> None:
    if user_id not in _session_meta:
        _session_meta[user_id] = {}
    if exam_date:
        try:
            from datetime import datetime
            _session_meta[user_id]["exam_date"] = datetime.fromisoformat(exam_date.replace("Z", "+00:00")).timestamp()
        except Exception:
            _session_meta[user_id]["exam_date"] = None
    else:
        _session_meta[user_id]["exam_date"] = None


def _get_exam_urgency(user_id: str) -> float:
    """Return urgency coefficient: <=7d=3.0, 8-30d=2.0, >30d=1.0"""
    meta = _session_meta.get(user_id, {})
    exam_ts = meta.get("exam_date")
    if not exam_ts:
        return 1.0
    now = time.time()
    days_left = (exam_ts - now) / 86400
    if days_left <= 0:
        return 5.0  # exam day or past
    if days_left <= 7:
        return 3.0
    if days_left <= 30:
        return 2.0
    return 1.0


def _mark_weak(user_id: str, chunk_id: str) -> None:
    if user_id not in _weak_chunks:
        _weak_chunks[user_id] = set()
    _weak_chunks[user_id].add(chunk_id)


def _is_weak(user_id: str, chunk_id: str) -> bool:
    return chunk_id in _weak_chunks.get(user_id, set())


def _clear_weak(user_id: str, chunk_id: str) -> None:
    _weak_chunks.get(user_id, set()).discard(chunk_id)


def _fsrs_score(user_id: str, chunk_id: str) -> float:
    """Return (1-R) * urgency. Higher = more urgent to review."""
    state = _get_review_state(user_id, chunk_id)
    S = max(0.1, state["S"])
    t = (time.time() - state["last_review"]) / 86400  # days
    R = max(0.0, min(1.0, 1.0 - (1.0 / S) * (1.0 - __import__("math").exp(-t / S))))
    # Simplified: R ≈ exp(-t/S)
    import math
    R = math.exp(-t / S) if S > 0 else 0.0
    urgency = _get_exam_urgency(user_id)
    return (1.0 - R) * urgency


# ---------------------------------------------------------------------------
# P0-1: Text Material Ingestion (0 LLM calls)
# ---------------------------------------------------------------------------
async def ingest_text(
    text: str,
    course_id: str = "",
    course_title: str = "",
    user_id: str = "",
    budget: LlmBudget | None = None,
) -> str:
    """Split pasted courseware text into chunks and store in the knowledge base.

    Args:
        text: raw courseware text (200-10000 chars)
        course_id: target course identifier (auto-generated if empty)
        course_title: human-readable title for the course
        user_id: for ACL (owner of this course)
        budget: accepted for interface consistency (0 LLM cost)

    Returns a JSON string with the ingestion result.
    """
    text = (text or "").strip()
    if not text:
        return json.dumps({"success": False, "error": "文本为空"}, ensure_ascii=False)

    if len(text) > 10000:
        return json.dumps({
            "success": False,
            "error": "内容较长，请分多次粘贴，每次不超过 10000 字",
        }, ensure_ascii=False)

    if len(text) < 200:
        return json.dumps({
            "success": False,
            "error": "内容过短（<200字），不触发导入。请粘贴至少200字的学习材料。",
        }, ensure_ascii=False)

    # Auto-generate course_id if not provided
    if not course_id:
        import uuid
        course_id = f"imported_{uuid.uuid4().hex[:8]}"

    if not course_title:
        course_title = "导入课程"

    # Simple chunking: split by double newlines or single newlines for long paragraphs
    # Target 200-400 chars per chunk
    chunks_data: list[dict] = []
    paragraphs = text.split("\n\n")
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # If single paragraph is very long, split by sentences (approx 200 char per unit)
        if len(para) > 400 and "。" in para:
            sentences = para.split("。")
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(current_chunk) + len(sent) + 1 <= 400:
                    current_chunk += ("。" if current_chunk else "") + sent
                else:
                    if current_chunk:
                        chunks_data.append({"text": current_chunk + "。"})
                        chunk_idx += 1
                    current_chunk = sent
        else:
            if len(current_chunk) + len(para) + 2 <= 400:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                if current_chunk:
                    chunks_data.append({"text": current_chunk})
                    chunk_idx += 1
                current_chunk = para

    if current_chunk:
        chunks_data.append({"text": current_chunk})
        chunk_idx += 1

    # Build LessonChunk objects
    try:
        from knowledge_store import LessonChunk, embedding_provider, knowledge_store
    except ImportError:
        from knowledge_store import LessonChunk, embedding_provider, knowledge_store

    chunks, texts = [], []
    for i, cd in enumerate(chunks_data):
        chunk_text = cd["text"]
        section_id = f"S{i+1:03d}"
        lesson_id = f"L001"  # imported text goes into a single lesson
        chunks.append(LessonChunk(
            id=f"{lesson_id}-{section_id}",
            lesson_id=lesson_id,
            lesson_title=f"第{i+1}节",
            section_id=section_id,
            text=chunk_text,
            course_id=course_id,
            course_title=course_title,
        ))
        texts.append(chunk_text)

    # Embed
    try:
        embeddings = embedding_provider.embed(texts)
    except Exception:
        embeddings = embedding_provider._noop_embed(texts)

    knowledge_store.add_chunks(chunks, embeddings)

    # Record retrieval counts for mastery (all start at 0)
    for chunk in chunks:
        key = _review_key(user_id, chunk.id)
        if key not in _review_state:
            _review_state[key] = {
                "S": 1.0,
                "last_review": time.time(),
                "retrievals": 0,
                "correct": 0,
                "incorrect": 0,
            }

    return json.dumps({
        "success": True,
        "course_id": course_id,
        "course_title": course_title,
        "chunks_ingested": len(chunks),
        "note": f"已整理 {len(chunks)} 个知识块，存入课程「{course_title}」。现在可以问我任何相关问题。",
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# P0-2: Review Schedule (0 LLM calls — pure calculation)
# ---------------------------------------------------------------------------
async def schedule_review(
    exam_date: str = "",
    course_id: str = "",
    user_id: str = "",
    top_k: int = 5,
    budget: LlmBudget | None = None,
) -> str:
    """Calculate FSRS-based review priorities for a course.

    R = exp(-t/S), urgency = (1-R) * exam_urgency_coefficient.
    Returns top_k chunks ranked by urgency descending.
    """
    _set_session_exam(user_id, exam_date or None)

    try:
        from knowledge_store import knowledge_store
    except ImportError:
        from knowledge_store import knowledge_store

    # Collect chunks for this course
    all_chunks = []
    for chunk, _ in knowledge_store._chunks:
        if course_id and chunk.course_id and chunk.course_id != course_id:
            continue
        all_chunks.append(chunk)

    if not all_chunks:
        return json.dumps({
            "results": [],
            "note": "知识库为空：请先导入学习材料或生成课程，我才能计算复习计划。",
        }, ensure_ascii=False)

    # Score each chunk
    scored = []
    for chunk in all_chunks:
        score = _fsrs_score(user_id, chunk.id)
        state = _get_review_state(user_id, chunk.id)
        import math
        R = math.exp(-((time.time() - state["last_review"]) / 86400) / max(0.1, state["S"]))
        retrievability = max(0.0, min(1.0, R))
        scored.append({
            "chunk_id": chunk.id,
            "lesson_id": chunk.lesson_id,
            "lesson_title": chunk.lesson_title,
            "section_id": chunk.section_id,
            "course_title": chunk.course_title,
            "retrievability": round(retrievability * 100, 1),
            "urgency_score": round(score, 3),
            "S": round(state["S"], 2),
            "correct": state["correct"],
            "incorrect": state["incorrect"],
            "is_weak": _is_weak(user_id, chunk.id),
        })

    # Sort by urgency descending
    scored.sort(key=lambda x: x["urgency_score"], reverse=True)

    # Check if exam is today
    urgency_now = _get_exam_urgency(user_id)
    if urgency_now >= 5.0:
        return json.dumps({
            "results": [],
            "exam_today": True,
            "note": "考试当天！建议直接开始模考来检验掌握情况。回复「开始模考」。",
        }, ensure_ascii=False)

    top = scored[:top_k]
    return json.dumps({
        "results": top,
        "exam_urgency": urgency_now,
        "exam_date": exam_date,
        "total_chunks": len(all_chunks),
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# P1-2: Record Quiz Answer (0 LLM calls) — update FSRS state after student answers
# ---------------------------------------------------------------------------
async def record_answer(
    chunk_id: str = "",
    user_id: str = "",
    correct: bool = False,
    budget: LlmBudget | None = None,
) -> str:
    """Record a quiz result and update FSRS stability S for the chunk.

    Called by the dispatcher after generate_quiz when the student answers.
    This updates S in the review state:
      correct=True  → S += 0.2 (stability improves)
      correct=False → S -= 0.15 (stability drops)
    Also manages weak-chunk tracking for mock exam priority.
    """
    if not chunk_id or not user_id:
        return json.dumps({"success": False, "error": "chunk_id 和 user_id 必填"}, ensure_ascii=False)

    state = _get_review_state(user_id, chunk_id)
    if correct:
        state["correct"] = state.get("correct", 0) + 1
        state["S"] = state.get("S", 1.0) + 0.2
        # Clear weak if streak >= 2
        streak_key = f"{user_id}:{chunk_id}"
        _mock_streak[streak_key] = _mock_streak.get(streak_key, 0) + 1
        if _mock_streak.get(streak_key, 0) >= 2:
            _clear_weak(user_id, chunk_id)
    else:
        state["incorrect"] = state.get("incorrect", 0) + 1
        state["S"] = max(0.1, state.get("S", 1.0) - 0.15)
        # Mark as weak for mock exam priority
        _mark_weak(user_id, chunk_id)
        _mock_streak.pop(f"{user_id}:{chunk_id}", None)

    state["last_review"] = time.time()
    _save_review_state(user_id, chunk_id, state)

    return json.dumps({
        "success": True,
        "chunk_id": chunk_id,
        "correct": correct,
        "new_S": round(state["S"], 2),
        "is_weak": _is_weak(user_id, chunk_id),
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# P1-1: Mastery Report (0 LLM calls)
# ---------------------------------------------------------------------------
async def mastery_report(
    course_id: str = "",
    user_id: str = "",
    budget: LlmBudget | None = None,
) -> str:
    """Generate a text mastery report for a course based on retrieval and quiz records."""
    try:
        from knowledge_store import knowledge_store
    except ImportError:
        from knowledge_store import knowledge_store

    all_chunks = []
    for chunk, _ in knowledge_store._chunks:
        if course_id and chunk.course_id and chunk.course_id != course_id:
            continue
        all_chunks.append(chunk)

    if not all_chunks:
        return json.dumps({
            "report": "知识库为空：请先导入材料或生成课程，再查询掌握度。",
        }, ensure_ascii=False)

    # Group by lesson
    by_lesson: dict[str, list] = {}
    for chunk in all_chunks:
        lid = chunk.lesson_id or "unknown"
        if lid not in by_lesson:
            by_lesson[lid] = {
                "lesson_title": chunk.lesson_title or lid,
                "chunks": [],
            }
        state = _get_review_state(user_id, chunk.id)
        retrievability = 0.0
        try:
            import math
            t = (time.time() - state["last_review"]) / 86400
            retrievability = math.exp(-t / max(0.1, state["S"]))
        except Exception:
            pass

        mastery = min(100, max(0,
            state["retrievals"] * 10 +
            state["correct"] * 15 +
            5 if (state["correct"] > 0 and state["incorrect"] == 0) else 0 -
            state["incorrect"] * 10
        ))
        by_lesson[lid]["chunks"].append({
            "chunk_id": chunk.id,
            "section_id": chunk.section_id,
            "mastery_score": max(0, mastery),
            "retrievals": state["retrievals"],
            "correct": state["correct"],
            "incorrect": state["incorrect"],
            "retrievability": round(retrievability * 100, 1),
            "is_weak": _is_weak(user_id, chunk.id),
        })

    # Compute per-lesson mastery
    lesson_summaries = []
    for lid, data in by_lesson.items():
        chunks = data["chunks"]
        avg_mastery = sum(c["mastery_score"] for c in chunks) / len(chunks) if chunks else 0
        total_correct = sum(c["correct"] for c in chunks)
        total_incorrect = sum(c["incorrect"] for c in chunks)
        total_retrievals = sum(c["retrievals"] for c in chunks)
        never_interacted = sum(1 for c in chunks if c["retrievals"] == 0 and c["correct"] == 0 and c["incorrect"] == 0)

        if avg_mastery >= 75:
            level = "high"
        elif avg_mastery >= 45:
            level = "mid"
        else:
            level = "low"

        lesson_summaries.append({
            "lesson_id": lid,
            "lesson_title": data["lesson_title"],
            "mastery_score": round(avg_mastery, 1),
            "level": level,
            "total_correct": total_correct,
            "total_incorrect": total_incorrect,
            "total_retrievals": total_retrievals,
            "never_interacted": never_interacted,
            "chunks": chunks,
        })

    lesson_summaries.sort(key=lambda x: x["mastery_score"])

    # Overall
    overall = sum(l["mastery_score"] for l in lesson_summaries) / len(lesson_summaries) if lesson_summaries else 0
    weak_lessons = [l for l in lesson_summaries if l["mastery_score"] < 45]
    never_learning = [l for l in lesson_summaries if l["never_interacted"] == len(l["chunks"])]

    # Build text report
    lines = [f"课程掌握度报告："]
    lines.append(f"总体掌握度：{overall:.0f}%")
    lines.append("")
    for l in lesson_summaries:
        lines.append(f"{l['lesson_title']} — {l['mastery_score']:.0f}% "
                     f"(检索{l['total_retrievals']}次/答对{l['total_correct']}题/答错{l['total_incorrect']}题)")
    lines.append("")
    if weak_lessons:
        lines.append("薄弱章节：")
        for l in weak_lessons:
            lines.append(f"  · {l['lesson_title']} ({l['mastery_score']:.0f}%)")
        lines.append("")
    if never_learning:
        lines.append("从未互动章节（需重点学习）：")
        for l in never_learning:
            lines.append(f"  · {l['lesson_title']}")
        lines.append("")
    lines.append("建议：回复「讲讲[章节名]」开始学习，或「出几道[章节名]的题」检验掌握。")

    return json.dumps({
        "report": "\n".join(lines),
        "lesson_summaries": lesson_summaries,
        "overall_mastery": round(overall, 1),
        "weak_lessons": [l["lesson_id"] for l in weak_lessons],
        "never_learning": [l["lesson_id"] for l in never_learning],
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# P1-2: Adaptive Mock Exam (1 LLM call per question)
# ---------------------------------------------------------------------------
async def mock_exam(
    action: str = "start",
    course_id: str = "",
    user_id: str = "",
    budget: LlmBudget | None = None,
    **kwargs,
) -> str:
    """Adaptive mock exam tool.

    Actions:
      start  — initialize exam session, pick topics (0 LLM calls)
      answer — record student's answer and return next question or summary (1 LLM call)
    """
    if action == "start":
        return _mock_start(course_id, user_id)

    if action == "answer":
        correct = kwargs.get("correct")
        answer_text = kwargs.get("answer_text", "")
        return await _mock_answer(course_id, user_id, budget, correct=correct, answer_text=answer_text)

    return json.dumps({"error": f"未知 action: {action}"}, ensure_ascii=False)


def _mock_start(course_id: str, user_id: str) -> str:
    """Pick topics and initialize exam session. 0 LLM calls."""
    try:
        from knowledge_store import knowledge_store
    except ImportError:
        from knowledge_store import knowledge_store

    all_chunks = []
    for chunk, _ in knowledge_store._chunks:
        if course_id and chunk.course_id and chunk.course_id != course_id:
            continue
        all_chunks.append(chunk)

    if not all_chunks:
        return json.dumps({
            "error": "知识库为空，请先导入材料或生成课程。",
        }, ensure_ascii=False)

    # Pick up to 3 chunks from different lessons for variety
    lesson_groups: dict[str, list] = {}
    for chunk in all_chunks:
        lid = chunk.lesson_id or "unknown"
        if lid not in lesson_groups:
            lesson_groups[lid] = []
        lesson_groups[lid].append(chunk)

    # Select up to 3 lessons
    selected: list[dict] = []
    for lid, chunks in list(lesson_groups.items())[:3]:
        import random
        ch = random.choice(chunks)
        selected.append({
            "chunk_id": ch.id,
            "lesson_id": lid,
            "lesson_title": ch.lesson_title,
            "section_id": ch.section_id,
            "course_title": ch.course_title,
        })

    # Prioritize weak chunks
    if user_id:
        weak_ids = _weak_chunks.get(user_id, set())
        prioritized = []
        for s in selected:
            if s["chunk_id"] in weak_ids:
                prioritized.append(s)
        prioritized.extend([s for s in selected if s["chunk_id"] not in weak_ids])
        selected = prioritized[:3]

    _mock_session[user_id] = {
        "state": "active",
        "topics": selected,
        "results": [],  # list of {chunk_id, correct, llm_used}
        "current_index": 0,
        "llm_used": 0,
    }

    return json.dumps({
        "started": True,
        "topics": [
            {"lesson_title": s["lesson_title"], "course_title": s["course_title"]}
            for s in selected
        ],
        "total": len(selected),
        "note": f"模考开始，共 {len(selected)} 道题。答对会提高下次同类题难度，答错会自动降低难度并讲解。",
    }, ensure_ascii=False)


async def _grade_answer(
    answer_text: str,
    question: str,
    correct_answer: str,
    question_type: str,
    budget: LlmBudget | None,
    chunk_text: str = "",
) -> dict:
    """Use LLM to grade a learner's text answer. Returns {correct, feedback}."""
    if question_type == "single_choice" and correct_answer:
        # Multiple choice: exact match
        first_char = answer_text.strip()[0].upper()
        correct = first_char == correct_answer.upper()[0]
        return {"correct": correct, "feedback": ""}

    # For fill_blank and short_answer, use LLM to judge
    system_msg = (
        "你是一位严格的编程助教。请根据以下规则判断学生回答并给出点评。\n"
        "【判断标准】\n"
        "填空题：学生填的内容是否与标准答案语义一致或等效（大小写/术语差异可接受） → correct=true\n"
        "简答题：学生是否理解核心概念，解释基本正确（允许合理误差，不要求完美） → correct=true\n"
        "学生回答为空、只写'不知道'、明显敷衍 → correct=false\n"
        "学生回答与正确答案主题无关 → correct=false\n"
        "【输出格式】必须严格遵循以下JSON格式，不要输出任何其他内容：\n"
        '{"correct": true或false, "feedback": "点评内容"}'\n"
        "【点评内容要求】\n"
        "correct=true时：先肯定优点（1句），再复述本题目对应的关键知识点（2-3句），总字数80-120字。\n"
        "correct=false时：先指出错误或问题（1句），然后结合课程内容详细讲解本题涉及的核心知识点（3-5句），最后给出本题的标准答案（2-3句），总字数150-250字。\n"
        "不要省略任何部分，不要使用省略号。"
    )
    chunk_section = f"【课程内容】\n{chunk_text}\n\n" if chunk_text else ""
    user_msg = (
        f"{chunk_section}"
        f"【题目】{question}\n"
        f"【题型】{question_type}\n"
        f"【参考标准答案】{correct_answer or '（无固定标准答案，请根据课程内容判断）'}\n"
        f"【学生回答】{answer_text}\n"
        "请严格按照上述格式输出JSON。"
    )

    try:
        from llm_utils import chat_completion_json
        result = await chat_completion_json(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.1,
            max_tokens=2048,
        )
        if isinstance(result, dict):
            return {
                "correct": bool(result.get("correct")),
                "feedback": str(result.get("feedback", "")),
            }
    except Exception:
        pass
    # Fallback: mark wrong if we can't grade
    return {"correct": False, "feedback": "（自动评判不可用，请自行核对答案）"}


async def _mock_answer(
    course_id: str,
    user_id: str,
    budget: LlmBudget | None = None,
    correct: bool | None = None,
    answer_text: str = "",
) -> str:
    """Record answer (grade via LLM if text provided), adjust difficulty, generate next question."""
    session = _mock_session.get(user_id, {})
    if session.get("state") != "active":
        return json.dumps({
            "error": "没有进行中的模考，请先说「开始模考」。",
        }, ensure_ascii=False)

    topics = session.get("topics", [])
    idx = session.get("current_index", 0)
    results = session.get("results", [])

    # If text answer provided, use LLM to grade it (consumes 1 budget call)
    if answer_text and correct is None:
        if not _charge(budget):
            return json.dumps({"finished": True, "error": "LLM 预算已耗尽，无法评判回答。"}, ensure_ascii=False)
        # Look up chunk text for the previous question
        prev_chunk_id = session.get("last_question", {}).get("chunk_id", "")
        chunk_text = ""
        for ck, _ in knowledge_store._chunks:
            if ck.id == prev_chunk_id:
                chunk_text = ck.text[:600]
                break
        grading_result = await _grade_answer(
            answer_text=answer_text,
            question=session.get("last_question", {}).get("question", ""),
            correct_answer=session.get("last_question", {}).get("correct_answer", ""),
            question_type=session.get("last_question", {}).get("question_type", "short_answer"),
            budget=budget,
            chunk_text=chunk_text,
        )
        correct = grading_result["correct"]
        grading_feedback = grading_result.get("feedback", "")
    else:
        grading_feedback = ""

    # Record previous answer
    if idx > 0 and correct is not None:
        prev_topic = topics[idx - 1]
        prev_chunk_id = prev_topic["chunk_id"]
        results.append({"chunk_id": prev_chunk_id, "correct": correct})
        session["results"] = results
        streak_key = f"{user_id}:{prev_chunk_id}"
        if correct:
            _mock_streak[streak_key] = _mock_streak.get(streak_key, 0) + 1
            if _mock_streak.get(streak_key, 0) >= 2:
                _weak_chunks.get(user_id, set()).discard(prev_chunk_id)
        else:
            _mock_streak[streak_key] = 0
            _weak_chunks.setdefault(user_id, set()).add(prev_chunk_id)
        await record_answer(prev_chunk_id, user_id, correct)

    # Check if exam is done
    if session["current_index"] >= len(topics):
        return _mock_summary(user_id, session)

    topic = topics[session["current_index"]]
    chunk_id = topic["chunk_id"]
    session["current_index"] += 1

    # Get chunk text for context
    chunk_text = ""
    try:
        from knowledge_store import knowledge_store as _ks_instance
    except ImportError:
        from knowledge_store import knowledge_store as _ks_instance
    for ck, _ in _ks_instance._chunks:
        if ck.id == chunk_id:
            chunk_text = ck.text[:500]
            break

    if not _charge(budget):
        return json.dumps({
            "finished": True,
            "error": "LLM 预算已耗尽，无法继续出题。",
        }, ensure_ascii=False)

    # Determine difficulty based on streak
    streak_key = f"{user_id}:{chunk_id}"
    streak = _mock_streak.get(streak_key, 0)
    if streak >= 3:
        difficulty = "hard"
    elif streak >= 1:
        difficulty = "medium"
    else:
        difficulty = "easy"

    # Generate question via LLM — vary question type by index for variety
    question_types = ["single_choice", "fill_blank", "short_answer", "single_choice", "fill_blank"]
    qtype = question_types[session.get("current_index", 0) % len(question_types)]

    if qtype == "single_choice":
        system_msg = (
            "你是一位Python数据分析助教。根据【课程内容】出1道单选题，题目中必须包含课程内容里出现的具体代码或术语。\n"
            "严格遵循以下JSON格式（不要输出任何其他内容）：\n"
            '{"question":"在Python中，list和dict的主要区别是？\\nA. list有序dict无序\\nB. list用[]访问dict用[]访问\\nC. list可修改dict不可修改\\nD. list元素不重复dict元素可重复","options":["A. list有序dict无序","B. list用[]访问dict用[]访问","C. list可修改dict不可修改","D. list元素不重复dict元素可重复"],"answer":"A"}'
        )
        user_msg = (
            f"【课程内容】（必须基于这段内容出题，禁止脱离这段内容编造）：\n{chunk_text}\n\n"
            f"难度：{difficulty}\n出1道与上述课程内容直接相关的单选题。"
        )
    elif qtype == "fill_blank":
        system_msg = (
            "你是一位Python数据分析助教。根据【课程内容】出1道填空题，"
            "答案必须是课程内容里出现过的具体代码、函数名或术语。\n"
            "严格遵循以下JSON格式（不要输出任何其他内容）：\n"
            '{"question":"在Python中，创建空列表的语句是___","options":[],"answer":"[]"}'
        )
        user_msg = (
            f"【课程内容】（必须基于这段内容出题，禁止脱离这段内容编造）：\n{chunk_text}\n\n"
            f"难度：{difficulty}\n出1道与上述课程内容直接相关的填空题。"
        )
    else:
        system_msg = (
            "你是一位Python数据分析助教。根据【课程内容】出1道简答题，"
            "题目必须涉及课程内容中出现的具体代码示例或概念对比。\n"
            "严格遵循以下JSON格式（不要输出任何其他内容）：\n"
            '{"question":"请写出列表推导式[表达式 for 变量 in 可迭代对象]的完整语法，并举一个实际例子","options":[],"answer":""}'
        )
        user_msg = (
            f"【课程内容】（必须基于这段内容出题，禁止脱离这段内容编造）：\n{chunk_text}\n\n"
            f"难度：{difficulty}\n出1道与上述课程内容直接相关的简答题。"
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        from llm_utils import chat_completion_json
        llm_result = await chat_completion_json(messages)
        if isinstance(llm_result, dict) and not llm_result.get("error"):
            question_data = llm_result
        else:
            question_data = {
                "question": f"请解释 {topic['lesson_title']} 中的核心概念，并说明其在数据分析中的应用。",
                "options": [],
                "answer": "",
            }
    except Exception:
        question_data = {
            "question": f"关于「{topic['lesson_title']}」，请简要说明其核心知识点。",
            "options": [],
            "answer": "",
        }

    session["llm_used"] = session.get("llm_used", 0) + 1
    last_q = {
        "question": question_data.get("question", f"请解释 {topic['lesson_title']}。"),
        "options": question_data.get("options", []),
        "correct_answer": question_data.get("answer", ""),
        "question_type": qtype,
        "lesson_title": topic["lesson_title"],
        "chunk_id": chunk_id,
    }
    session["last_question"] = last_q

    return json.dumps({
        "question": last_q["question"],
        "options": last_q["options"],
        "correct_answer": last_q["correct_answer"],
        "question_type": qtype,
        "progress": f"{session['current_index']}/{len(topics)}",
        "lesson_title": topic["lesson_title"],
        "chunk_id": chunk_id,
        "note": "请作答，系统会自动记录。",
        "grading_feedback": grading_feedback if idx > 0 else "",
        "grading_correct": correct if idx > 0 else None,
    }, ensure_ascii=False)


def _mock_summary(user_id: str, session: dict) -> str:
    """Generate mock exam summary. 0 LLM calls."""
    results = session.get("results", [])
    topics = session.get("topics", [])

    session["state"] = "finished"

    if not results:
        # No results recorded yet — just mark finished
        return json.dumps({
            "finished": True,
            "total": len(topics),
            "correct": 0,
            "incorrect": 0,
            "note": "模考已结束，未记录答题结果。",
        }, ensure_ascii=False)

    correct = sum(1 for r in results if r.get("correct"))
    incorrect = len(results) - correct

    # Identify weak topics
    weak_ids = set(r["chunk_id"] for r in results if not r.get("correct"))

    # Update global weak tracking
    for chunk_id in weak_ids:
        _mark_weak(user_id, chunk_id)

    # Clear streaks for correct ones
    for r in results:
        if r.get("correct"):
            _mock_streak.pop(f"{user_id}:{r['chunk_id']}", None)

    lines = []
    lines.append(f"模考小结（共 {len(topics)} 题）：")
    lines.append(f"答对 {correct} / 答错 {incorrect}")
    if weak_ids:
        weak_titles = [t["lesson_title"] for t in topics if t["chunk_id"] in weak_ids]
        lines.append("")
        lines.append(f"薄弱章节：{', '.join(weak_titles)}")
        lines.append("下次对话中我会优先问你这些章节的题。")
    else:
        lines.append("表现良好！所有章节都已掌握。")

    return json.dumps({
        "finished": True,
        "total": len(topics),
        "correct": correct,
        "incorrect": incorrect,
        "weak_chunk_ids": list(weak_ids),
        "report": "\n".join(lines),
    }, ensure_ascii=False)


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
        # P0-1: 对话式材料导入
        "ingest_text": ingest_text,
        # P0-2: 复习调度
        "schedule_review": schedule_review,
        # P1-1: 掌握度报告
        "mastery_report": mastery_report,
        # P1-2: 自适应模考
        "mock_exam": mock_exam,
        # P1-2: 答题记录（答题后更新FSRS状态）
        "record_answer": record_answer,
    }
