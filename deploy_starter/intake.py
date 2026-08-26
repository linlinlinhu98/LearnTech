"""Guided learning goal definition — intake state machine.

Architecture:
  IntakeManager (process-wide registry)
    └── IntakeSession per user_id
          ├── state: COLLECTING | CONFIRMED | CANCELLED
          ├── round: 0..MAX_ROUNDS
          └── brief: partial LearningBrief dict

Each turn the coordinator calls IntakeManager.respond(user_id, message).
If the brief is complete or round >= MAX_ROUNDS, state → CONFIRMED
and the confirmed brief is returned. The caller then triggers authoring.

No AgentScope dependency — pure async/await, unit-testable offline.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

MAX_ROUNDS = 6
IDLE_TIMEOUT_SECONDS = 600  # 10 minutes


class IntakeSession:
    """Holds the in-progress state for one user's intake conversation."""

    __slots__ = ("user_id", "round", "state", "brief", "created_at")

    def __init__(self, user_id: str) -> None:
        self.user_id: str = user_id
        self.round: int = 0
        self.state: str = "COLLECTING"
        self.brief: dict[str, Any] = {
            "goal": "",
            "current_level": "",
            "target_level": "",
            "available_time": "",
            "preferred_style": "",
            "constraints": "",
            "success_criteria": "",
        }
        self.created_at: float = time.time()

    @property
    def is_complete(self) -> bool:
        """Brief is considered complete when goal + current_level + available_time are filled."""
        return bool(
            self.brief.get("goal")
            and self.brief.get("current_level")
            and self.brief.get("available_time")
        )

    @property
    def is_expired(self) -> bool:
        """Expired if idle more than IDLE_TIMEOUT_SECONDS."""
        return (time.time() - self.created_at) > IDLE_TIMEOUT_SECONDS

    def apply_answer(self, answer: str) -> None:
        """Merge a user's answer into the appropriate brief field.

        The answer may fill multiple fields if the user provided several at once.
        We fill in order of priority: goal -> current_level -> target_level ->
        available_time -> preferred_style -> constraints -> success_criteria.
        """
        answer = answer.strip()
        if not answer:
            return

        fields = [
            ("goal", self.brief.get("goal", "")),
            ("current_level", self.brief.get("current_level", "")),
            ("target_level", self.brief.get("target_level", "")),
            ("available_time", self.brief.get("available_time", "")),
            ("preferred_style", self.brief.get("preferred_style", "")),
            ("constraints", self.brief.get("constraints", "")),
            ("success_criteria", self.brief.get("success_criteria", "")),
        ]
        # Find first empty field and fill it
        for i, (key, val) in enumerate(fields):
            if not val:
                self.brief[key] = answer
                return

        # If all filled, append to success_criteria
        if self.brief["success_criteria"]:
            self.brief["success_criteria"] += "；" + answer
        else:
            self.brief["success_criteria"] = answer

    def advance(self) -> None:
        self.round += 1
        if self.is_complete or self.round >= MAX_ROUNDS:
            self.state = "CONFIRMED"

    def cancel(self) -> None:
        self.state = "CANCELLED"


class IntakeManager:
    """Process-wide registry of active intake sessions.

    In a production deployment with multiple workers, replace this in-memory
    dict with a Redis dict keyed by user_id.
    """

    __slots__ = ("_sessions",)

    def __init__(self) -> None:
        self._sessions: dict[str, IntakeSession] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, user_id: str) -> IntakeSession:
        """Start a new intake session for user_id, replacing any existing one."""
        session = IntakeSession(user_id)
        self._sessions[user_id] = session
        return session

    def get(self, user_id: str) -> IntakeSession | None:
        return self._sessions.get(user_id)

    def cancel(self, user_id: str) -> None:
        session = self._sessions.get(user_id)
        if session:
            session.cancel()

    def respond(
        self,
        user_id: str,
        message: str,
        llm_generate_question: bool = True,
    ) -> dict[str, Any]:
        """Main entry point: feed a user message and get the next response.

        Args:
            user_id: the user identifier.
            message: the user's latest message (may fill one or more brief fields).
            llm_generate_question: if True, use LLM to generate the next question;
                                      otherwise do priority-based field fill.

        Returns a dict with keys:
            - session_id, state, round, is_complete
            - brief (current partial or confirmed brief)
            - next_question (empty string when complete)
            - suggestion (one-line tip shown in UI while waiting)
        """
        # Auto-start or resume
        session = self._sessions.get(user_id)
        if session is None:
            session = self.start(user_id)

        # Handle system commands
        lc_msg = message.lower().strip()
        if lc_msg in ("取消", "quit", "exit", "stop"):
            session.cancel()
            return self._build_response(session, next_question="")

        # Check expiry
        if session.state in ("CONFIRMED", "CANCELLED"):
            # Restart if they send a new goal
            session = self.start(user_id)

        # Apply answer to brief
        session.apply_answer(message)
        session.advance()

        if session.state == "CONFIRMED":
            return self._build_response(
                session,
                next_question="",
            )

        # Generate next question
        next_question = self._default_next_question(session)
        return self._build_response(session, next_question=next_question)

    # ------------------------------------------------------------------
    # Response builder
    # ------------------------------------------------------------------
    @staticmethod
    def _build_response(
        session: IntakeSession,
        next_question: str,
    ) -> dict[str, Any]:
        suggestion = ""
        if session.state == "COLLECTING" and not session.brief.get("current_level"):
            suggestion = "尽量具体描述你目前的基础，比如：'学过 Python 基础，了解变量和循环'"
        elif session.state == "COLLECTING" and not session.brief.get("available_time"):
            suggestion = "例如：'每周 5 小时，持续 3 个月'"

        return {
            "session_id": session.user_id,
            "state": session.state,
            "round": session.round,
            "max_rounds": MAX_ROUNDS,
            "brief": dict(session.brief),
            "next_question": next_question,
            "suggestion": suggestion,
        }

    # ------------------------------------------------------------------
    # Question generation (no LLM required — rule-based priority fill)
    # ------------------------------------------------------------------
    @staticmethod
    def _default_next_question(session: IntakeSession) -> str:
        """Return the next question based on which brief field is still empty."""
        brief = session.brief
        unanswered_first = [
            ("goal",           "你想达到什么学习目标？请具体描述，比如：'掌握 Python 数据分析，能独立完成一个完整的数据项目'"),
            ("current_level", "你目前的基础是什么？例如：'完全零基础'、'学过 HTML/CSS'、'会写 SQL 查询'"),
            ("target_level",   "你希望达到什么水平？例如：'能通过面试'、'能独立做项目'、'竞赛获奖'"),
            ("available_time", "你每周能投入多少小时学习？能坚持多久？例如：'每周 5 小时，持续 3 个月'"),
            ("preferred_style","你偏好哪种学习方式？可选：视频 / 看书 / 做题练习 / 实战项目，或者混着来"),
            ("constraints",   "有什么限制条件吗？比如：'只有周末有时间'、'需要免费资源'、'要考证'"),
            ("success_criteria","你怎么判断自己学会了？例如：'能独立完成 Kaggle 小赛'、'能用英语面试'"),
        ]
        for key, question in unanswered_first:
            if not brief.get(key):
                return question
        return ""  # all filled


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------
_intake_manager: IntakeManager | None = None


def get_intake_manager() -> IntakeManager:
    global _intake_manager
    if _intake_manager is None:
        _intake_manager = IntakeManager()
    return _intake_manager
