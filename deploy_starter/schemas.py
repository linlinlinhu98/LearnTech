"""JSON shape contracts for Lumen-Bailian tutor planner and course authoring."""

from __future__ import annotations

# ------------------------------------------------------------------
# Module 3 — Tutor Dispatcher
# ------------------------------------------------------------------
TUTOR_PLAN_SHAPE = {
    "action": "execute | finalize",
    "reason": "string",
    "steps": [
        {
            "sub_agent": "retrieve | web_search | run_code | generate_quiz | explain_concept",
            "query": "string (optional, for retrieve/web_search)",
            "code": "string (optional, for run_code)",
            "topic": "string (optional, for generate_quiz)",
            "difficulty": "string (optional, for generate_quiz, default: medium)",
            "concept": "string (optional, for explain_concept)",
            "level": "string (optional, for explain_concept, default: beginner)",
            "top_k": "int (optional, for retrieve, default: 5)",
        }
    ],
}

# ------------------------------------------------------------------
# Module 1 — Course Authoring Pipeline
# ------------------------------------------------------------------
LEARNING_BRIEF_SHAPE = {
    "goal": "string — what the learner wants to achieve",
    "current_level": "string — beginner | intermediate | advanced",
    "target_level": "string — beginner | intermediate | advanced",
    "available_time": "string — e.g. 5 hours/week",
    "preferred_style": "string (optional) — visual | reading | practice | mixed",
    "constraints": "string (optional)",
    "success_criteria": "string (optional) — how to measure completion",
}

COURSE_OUTLINE_SHAPE = {
    "course_title": "string",
    "lessons": [
        {
            "lesson_id": "string",
            "lesson_title": "string",
            "objectives": ["string"],
            "key_points": ["string"],
        }
    ],
    "estimated_hours": "float",
}

CRITIQUE_SHAPE = {
    "passed": "bool",
    "score": "int 0-100",
    "problems": ["string"],
    "suggestions": ["string"],
}

COURSE_RAW_CONTENT_SHAPE = {
    "course_id": "string",
    "course_title": "string",
    "lessons": [
        {
            "lesson_id": "string",
            "lesson_title": "string",
            "chunks": [
                {
                    "chunk_id": "string",
                    "text": "string",
                }
            ],
        }
    ],
}

# ------------------------------------------------------------------
# Module 2 — Knowledge Store / RAG
# ------------------------------------------------------------------
LESSON_CHUNK_SHAPE = {
    "id": "string",
    "lesson_id": "string",
    "lesson_title": "string",
    "text": "string",
    "course_id": "string (optional, default empty = public)",
    "course_title": "string (optional)",
}
