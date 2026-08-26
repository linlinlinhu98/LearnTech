"""6-stage course authoring pipeline: Researcher → Outliner → Critic → Reviser → LessonDrafter → FinalCritic.

Each stage calls the LLM once; stages are串行 (critic→reviser can loop up to MAX_REVISE_ROUNDS).
No AgentScope dependency — pure async/await, unit-testable offline.
"""

from __future__ import annotations

import uuid
from typing import Any

try:
    from .llm_utils import LlmBudget, chat_completion_json
    from .prompts import (
        RESEARCHER_PROMPT,
        OUTLINER_PROMPT,
        CRITIC_PROMPT,
        REVISER_PROMPT,
        LESSON_DRAFTER_PROMPT,
        FINAL_CRITIC_PROMPT,
    )
except ImportError:  # Direct execution / local tests
    from llm_utils import LlmBudget, chat_completion_json
    from prompts import (
        RESEARCHER_PROMPT,
        OUTLINER_PROMPT,
        CRITIC_PROMPT,
        REVISER_PROMPT,
        LESSON_DRAFTER_PROMPT,
        FINAL_CRITIC_PROMPT,
    )

MAX_REVISE_ROUNDS = 2  # Critic→Reviser loop cap


# ---------------------------------------------------------------------------
# Stage runners (each calls the LLM once)
# ---------------------------------------------------------------------------

async def _researcher(brief: dict, budget: LlmBudget) -> dict:
    """Stage 1 — Researcher: generate core concepts + learning order."""
    user_content = _brief_summary(brief)
    messages = [
        {"role": "system", "content": RESEARCHER_PROMPT},
        {"role": "user", "content": user_content},
    ]
    budget.charge(1)
    result = await chat_completion_json(messages)
    if isinstance(result, dict) and "core_concepts" in result:
        return result
    return {"core_concepts": [], "recommended_order": [], "learning_strategy": ""}


async def _outliner(research_note: dict, budget: LlmBudget) -> dict:
    """Stage 2 — Outliner: build hierarchical course outline."""
    messages = [
        {"role": "system", "content": OUTLINER_PROMPT},
        {"role": "user", "content": f"研究笔记：\n{_json_pretty(research_note)}"},
    ]
    budget.charge(1)
    result = await chat_completion_json(messages)
    if isinstance(result, dict) and "lessons" in result:
        return result
    return {"course_title": "", "estimated_hours": 0.0, "lessons": []}


async def _critic(outline: dict, budget: LlmBudget) -> dict:
    """Stage 3 — Critic: review outline and return critique."""
    messages = [
        {"role": "system", "content": CRITIC_PROMPT},
        {"role": "user", "content": f"课程大纲：\n{_json_pretty(outline)}"},
    ]
    budget.charge(1)
    result = await chat_completion_json(messages)
    if isinstance(result, dict) and "passed" in result:
        return result
    return {"passed": True, "score": 100, "problems": [], "suggestions": []}


async def _reviser(outline: dict, critique: dict, budget: LlmBudget) -> dict:
    """Stage 4 — Reviser: revise outline based on critique."""
    messages = [
        {"role": "system", "content": REVISER_PROMPT},
        {"role": "user", "content": (
            f"原大纲：\n{_json_pretty(outline)}\n\n"
            f"批评意见：\n{_json_pretty(critique)}"
        )},
    ]
    budget.charge(1)
    result = await chat_completion_json(messages)
    if isinstance(result, dict) and "final_outline" in result:
        return result["final_outline"]
    return outline


async def _lesson_drafter(outline: dict, budget: LlmBudget) -> dict:
    """Stage 5 — Lesson Drafter: write full section content for each leaf node."""
    course_id = f"course_{uuid.uuid4().hex[:8]}"
    messages = [
        {"role": "system", "content": LESSON_DRAFTER_PROMPT},
        {"role": "user", "content": f"课程大纲：\n{_json_pretty(outline)}"},
    ]
    budget.charge(1)
    result = await chat_completion_json(messages)
    if isinstance(result, dict) and "lessons" in result:
        result["course_id"] = course_id
        return result
    return {"course_id": course_id, "course_title": outline.get("course_title", ""), "lessons": []}


async def _final_critic(content: dict, budget: LlmBudget) -> dict:
    """Stage 6 — Final Critic: read full content and return pass/fail + issues."""
    messages = [
        {"role": "system", "content": FINAL_CRITIC_PROMPT},
        {"role": "user", "content": f"完整课程：\n{_json_pretty(content)}"},
    ]
    budget.charge(1)
    result = await chat_completion_json(messages)
    if isinstance(result, dict) and "passed" in result:
        return result
    return {"passed": True, "score": 100, "issues": [], "summary": ""}


# ---------------------------------------------------------------------------
# Public pipeline runner
# ---------------------------------------------------------------------------

class AuthoringPipeline:
    """6-stage pipeline that takes a LearningBrief and produces CourseRawContent."""

    def __init__(self, max_llm_calls: int = 30) -> None:
        # Stage 1-6 each cost 1 call; critic→reviser loop adds 2 more per iteration.
        # Budget of 30 covers the full pipeline comfortably.
        self.budget = LlmBudget(max_llm_calls)

    async def run(self, brief: dict) -> dict[str, Any]:
        """Execute the full 6-stage pipeline and return the final course content."""
        stage = {"current": "researcher", "done": []}

        # Stage 1
        research_note = await _researcher(brief, self.budget)
        stage["done"].append("researcher")
        stage["current"] = "outliner"

        # Stage 2
        outline = await _outliner(research_note, self.budget)
        stage["done"].append("outliner")
        stage["current"] = "critic"

        # Stage 3 → 4 (Critic → Reviser loop, up to MAX_REVISE_ROUNDS)
        revise_count = 0
        while True:
            critique = await _critic(outline, self.budget)
            stage["done"].append(f"critic_r{revise_count}")
            if critique.get("passed", True) and critique.get("score", 100) >= 70:
                break
            if revise_count >= MAX_REVISE_ROUNDS:
                stage["done"].append("critic_max_rounds_reached")
                break
            revised_outline = await _reviser(outline, critique, self.budget)
            outline = revised_outline
            stage["done"].append(f"reviser_r{revise_count}")
            revise_count += 1
        stage["current"] = "lesson_drafter"

        # Stage 5
        raw_content = await _lesson_drafter(outline, self.budget)
        stage["done"].append("lesson_drafter")
        stage["current"] = "final_critic"

        # Stage 6 (Final Critic — no re-draft loop per spec; just record result)
        final_verdict = await _final_critic(raw_content, self.budget)
        stage["done"].append("final_critic")
        stage["current"] = "done"

        return {
            "course_id": raw_content.get("course_id", ""),
            "course_title": raw_content.get("course_title", ""),
            "outline": outline,
            "research_note": research_note,
            "raw_content": raw_content,
            "final_verdict": final_verdict,
            "pipeline_stage": stage,
            "llm_calls_used": self.budget.used,
            "llm_calls_budget": self.budget.max_calls,
        }


# ---------------------------------------------------------------------------
# Ingest course into knowledge store
# ---------------------------------------------------------------------------

def ingest_course(raw_content: dict, course_title: str = "") -> list[dict]:
    """Convert CourseRawContent into LessonChunks ready for the knowledge store.

    Returns the list of chunk dicts (not persisted — caller passes them to
    knowledge_store.add_chunks via the /api/v1/knowledge/ingest endpoint).

    Chunk structure:
      id          = f"{lesson_id}-{section_id}"
      lesson_id  = lesson_id from raw_content
      lesson_title = lesson_title
      section_id = section_id
      text       = section content
      course_id  = course_id from raw_content
      course_title = course_title override or from raw_content
    """
    course_id = raw_content.get("course_id", "")
    title = course_title or raw_content.get("course_title", "")

    chunks = []
    for lesson in raw_content.get("lessons", []):
        lesson_id = lesson.get("lesson_id", "")
        lesson_title = lesson.get("lesson_title", "")
        for section in lesson.get("sections", []):
            chunk_id = f"{lesson_id}-{section.get('section_id', '')}"
            chunks.append({
                "id": chunk_id,
                "lesson_id": lesson_id,
                "lesson_title": lesson_title,
                "section_id": section.get("section_id", ""),
                "text": section.get("content", ""),
                "course_id": course_id,
                "course_title": title,
            })
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brief_summary(brief: dict) -> str:
    parts = []
    for key in ("goal", "current_level", "target_level", "available_time",
                "preferred_style", "constraints", "success_criteria"):
        val = brief.get(key, "")
        if val:
            parts.append(f"{key}: {val}")
    return "\n".join(parts) or "(no brief fields)"


def _json_pretty(data: Any) -> str:
    import json
    return json.dumps(data, ensure_ascii=False, indent=2)
