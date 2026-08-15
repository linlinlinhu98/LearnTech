"""AgentScope tool wrappers for Lumen's five tutor sub-agents.

Each wrapper delegates to the Agentscope-free core logic in tutor_core.py and
wraps the result in an AgentScope ``ToolResponse`` for the coordinator agent.
All LLM calls inside the core functions use httpx directly — no openai SDK.
"""

from __future__ import annotations

import json
from typing import Any

from agentscope.tool import ToolResponse
from agentscope.tool._response import TextBlock

try:
    from .tutor_core import (
        explain_concept,
        generate_quiz,
        retrieve_course,
        run_code,
        search_web,
    )
except ImportError:  # Platform direct execution: python deploy_starter/main.py
    from tutor_core import (
        explain_concept,
        generate_quiz,
        retrieve_course,
        run_code,
        search_web,
    )


def _tool_response(payload: dict[str, Any] | list[Any] | str) -> ToolResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return ToolResponse(content=[TextBlock(type="text", text=text)])


async def tool_retrieve(query: str, top_k: int = 5, course_id: str = "") -> ToolResponse:
    """Search the course knowledge base for relevant materials.

    Use this when a student asks a question related to course content.
    Retrieves the most semantically similar lesson chunks.

    Args:
        query (str): The search query — student's question or topic.
        top_k (int): Number of chunks to retrieve, default 5.
        course_id (str): Optional course scope filter.

    Returns:
        ToolResponse: Structured JSON of retrieved lesson chunks with scores.
    """
    result = await retrieve_course(query, top_k=top_k, course_id=course_id)
    return _tool_response(result)


async def tool_web_search(query: str) -> ToolResponse:
    """Supplement course content with model background knowledge.

    Use this when course materials cannot cover the question, or the student
    needs broader context beyond the course.

    Args:
        query (str): The question or topic to look up.

    Returns:
        ToolResponse: A concise information summary.
    """
    result = await search_web(query)
    return _tool_response(result)


async def tool_run_code(code: str) -> ToolResponse:
    """Run Python code in a restricted sandbox and return its output.

    Use this for calculations, verifying code, or demonstrating a code example.
    Only safe standard-library modules are available.

    Args:
        code (str): Complete, runnable Python code.

    Returns:
        ToolResponse: JSON with stdout, stderr, and exit_code.
    """
    result = await run_code(code)
    return _tool_response(result)


async def tool_generate_quiz(topic: str, difficulty: str = "medium") -> ToolResponse:
    """Generate practice quiz questions for a given topic.

    Use this when the student asks for exercises, a quiz, or practice questions.

    Args:
        topic (str): The topic to generate questions about.
        difficulty (str): Difficulty level — easy, medium, or hard.

    Returns:
        ToolResponse: Generated quiz questions with answers and explanations.
    """
    result = await generate_quiz(topic, difficulty=difficulty)
    return _tool_response(result)


async def tool_explain_concept(concept: str, level: str = "beginner") -> ToolResponse:
    """Explain a concept in depth with a layered explanation.

    Use this when the student asks "what is...", "explain...", or cannot
    understand a concept.

    Args:
        concept (str): The concept or topic to explain.
        level (str): Student level — beginner, intermediate, or advanced.

    Returns:
        ToolResponse: A layered explanation of the concept.
    """
    result = await explain_concept(concept, level=level)
    return _tool_response(result)
