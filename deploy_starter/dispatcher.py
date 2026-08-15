"""Core multi-agent tutor dispatcher: plan → execute → synthesize.

Enforces two hard budgets per tutoring turn:
  * at most ``max_iterations`` plan/execute rounds, and
  * at most ``max_llm_calls`` LLM invocations across the planner, synthesizer
    and any LLM-consuming sub-agent (quiz / explain). The retriever, code
    runner and web searcher (Tavily) are deterministic/search APIs and free.

No AgentScope dependency — the tools are injected as plain async callables so
the dispatcher can be unit-tested locally with fakes.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

try:
    from .llm_utils import LlmBudget, chat_completion, chat_completion_json
    from .prompts import TUTOR_PLANNER_PROMPT, TUTOR_SYNTHESIZER_PROMPT
except ImportError:  # Direct execution / local tests
    from llm_utils import LlmBudget, chat_completion, chat_completion_json
    from prompts import TUTOR_PLANNER_PROMPT, TUTOR_SYNTHESIZER_PROMPT

MAX_HISTORY_ROUNDS = 5


class TutorDispatcher:
    def __init__(
        self,
        tools: dict[str, Any],
        max_iterations: int = 5,
        max_llm_calls: int = 8,
    ) -> None:
        self.tools = tools
        self.max_iterations = int(max_iterations)
        self.max_llm_calls = int(max_llm_calls)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        question: str,
        user_id: str = "",
        course_id: str = "",
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        budget = LlmBudget(self.max_llm_calls)
        evidence: list[dict[str, str]] = []
        iterations = 0
        history = (history or [])[-MAX_HISTORY_ROUNDS:]

        for _ in range(self.max_iterations):
            # Reserve one call for the final synthesizer: we only plan again if
            # there is room for this plan *and* the synthesizer.
            if budget.used + 2 > budget.max_calls:
                break

            iterations += 1
            plan = await self._plan(question, evidence, history, budget)
            if not isinstance(plan, dict) or plan.get("tool_error"):
                break

            action = str(plan.get("action") or "finalize").strip().lower()
            if action == "finalize":
                break

            steps = plan.get("steps")
            if not isinstance(steps, list) or not steps:
                break

            # 同一轮计划里的多个子代理互相独立（检索/搜索/跑代码/出题/讲解
            # 互不依赖），并发执行以压缩墙钟时间。budget.charge 是同步的，
            # 在单线程事件循环里原子生效，因此并发下 LLM 调用计数仍严格
            # 不超过 max_llm_calls。asyncio.gather 按传入顺序返回结果，
            # 故 evidence / [参考N] 编号保持与规划器给出的 steps 顺序一致。
            results = await asyncio.gather(
                *(self._run_step(step, question, course_id, budget) for step in steps)
            )
            evidence.extend(r for r in results if r is not None)

        answer = await self._synthesize(question, evidence, history, budget)

        return {
            "answer": answer,
            "evidence": evidence,
            "iterations": iterations,
            "llm_calls": budget.used,
            "max_llm_calls": budget.max_calls,
        }

    # ------------------------------------------------------------------
    # LLM helpers (charge the budget, then call the model)
    # ------------------------------------------------------------------
    async def _llm_json(self, messages: list[dict[str, str]], budget: LlmBudget) -> Any:
        if not budget.can_charge(1):
            return {
                "tool_error": "budget_exhausted",
                "message": "LLM 调用预算已耗尽。",
            }
        budget.charge(1)
        return await chat_completion_json(messages)

    async def _llm_text(self, messages: list[dict[str, str]], budget: LlmBudget) -> str:
        if not budget.can_charge(1):
            return "[LLM 调用预算已耗尽。]"
        budget.charge(1)
        return await chat_completion(messages)

    async def _plan(
        self,
        question: str,
        evidence: list[dict[str, str]],
        history: list[dict[str, str]],
        budget: LlmBudget,
    ) -> Any:
        messages = [
            {"role": "system", "content": TUTOR_PLANNER_PROMPT},
            {"role": "user", "content": self._plan_user_content(question, evidence, history)},
        ]
        return await self._llm_json(messages, budget)

    async def _synthesize(
        self,
        question: str,
        evidence: list[dict[str, str]],
        history: list[dict[str, str]],
        budget: LlmBudget,
    ) -> str:
        if not budget.can_charge(1):
            return self._fallback_answer(question, evidence)

        answer = await self._llm_text(
            [
                {"role": "system", "content": TUTOR_SYNTHESIZER_PROMPT},
                {"role": "user", "content": self._synth_user_content(question, evidence, history)},
            ],
            budget,
        )
        if answer.startswith("[LLM 调用预算已耗尽"):
            return self._fallback_answer(question, evidence)
        return answer

    # ------------------------------------------------------------------
    # Execution of a single planned step
    # ------------------------------------------------------------------
    async def _execute(
        self,
        step: dict[str, Any],
        question: str,
        course_id: str,
        budget: LlmBudget,
    ) -> str | None:
        name = str(step.get("sub_agent") or "").strip()
        tool = self.tools.get(name)
        if tool is None:
            return json.dumps({"error": f"未知子代理: {name}"}, ensure_ascii=False)

        kwargs = self._build_tool_args(name, step, question, course_id)
        kwargs["budget"] = budget

        try:
            result = tool(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return self._coerce_output(result)
        except Exception as exc:
            return json.dumps({"error": f"{name} 执行失败: {exc}"}, ensure_ascii=False)

    async def _run_step(
        self,
        step: dict[str, Any],
        question: str,
        course_id: str,
        budget: LlmBudget,
    ) -> dict[str, str] | None:
        """Execute one planned step and wrap it as an evidence entry (or None)."""
        if not isinstance(step, dict):
            return None
        output = await self._execute(step, question, course_id, budget)
        if output is None:
            return None
        return {
            "sub_agent": str(step.get("sub_agent") or ""),
            "output": output,
        }

    # ------------------------------------------------------------------
    # Prompt / argument builders
    # ------------------------------------------------------------------
    def _build_tool_args(
        self,
        name: str,
        step: dict[str, Any],
        question: str,
        course_id: str,
    ) -> dict[str, Any]:
        if name == "retrieve":
            return {
                "query": step.get("query") or question,
                "top_k": int(step.get("top_k", 5) or 5),
                "course_id": step.get("course_id") or course_id,
            }
        if name == "web_search":
            return {"query": step.get("query") or question}
        if name == "run_code":
            return {"code": step.get("code") or ""}
        if name == "generate_quiz":
            return {
                "topic": step.get("topic") or question,
                "difficulty": step.get("difficulty") or "medium",
            }
        if name == "explain_concept":
            return {
                "concept": step.get("concept") or question,
                "level": step.get("level") or "beginner",
            }
        return {}

    @staticmethod
    def _coerce_output(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        # AgentScope ToolResponse-like objects
        content = getattr(result, "content", None)
        if content:
            parts: list[str] = []
            for block in content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            if parts:
                return "".join(parts)
        return str(result)

    @staticmethod
    def _plan_user_content(
        question: str,
        evidence: list[dict[str, str]],
        history: list[dict[str, str]],
    ) -> str:
        lines = [f"学生问题：{question}"]
        if history:
            lines.append("对话历史（最近几轮）：")
            for h in history:
                lines.append(f"- {h.get('role')}: {h.get('content')}")
        if evidence:
            lines.append("已收集的证据：")
            for e in evidence:
                lines.append(f"[{e['sub_agent']}]: {e['output'][:800]}")
        return "\n".join(lines)

    @staticmethod
    def _synth_user_content(
        question: str,
        evidence: list[dict[str, str]],
        history: list[dict[str, str]],
    ) -> str:
        lines = [f"学生问题：{question}"]
        if evidence:
            lines.append("收集到的证据（回答中引用时请使用 [参考N]，N 为证据编号）：")
            for i, e in enumerate(evidence, start=1):
                lines.append(f"[参考{i}] ({e['sub_agent']}):\n{e['output'][:1500]}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_answer(question: str, evidence: list[dict[str, str]]) -> str:
        if not evidence:
            return "抱歉，本轮未能调用任何工具或 LLM 预算已耗尽，请换个问法再试。"
        parts = ["以下是已收集到的资料（LLM 预算已耗尽，未能进一步综合）："]
        for i, e in enumerate(evidence, start=1):
            parts.append(f"[参考{i}] ({e['sub_agent']}):\n{e['output'][:800]}")
        return "\n\n".join(parts)
