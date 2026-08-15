"""End-to-end demo of the multi-agent tutor dispatcher (real LLM + embedding).

Runs the dispatcher and prints, for each turn:
  * plan/execute iterations and LLM-call budget usage,
  * which sub-agents the planner dispatched (the division of labor),
  * each sub-agent's raw output (evidence),
  * the final synthesized answer (full, uncut).

The demo questions cover all five sub-agents:
  retrieve / web_search / run_code / generate_quiz / explain_concept.

Usage (run from the repo root):
    python tests/test_e2e.py                       # run all demo questions
    python tests/test_e2e.py "Rust 所有权是什么"     # ask a single question
"""

from __future__ import annotations

import asyncio
import os
import sys

_DEPLOY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy_starter")
)
sys.path.insert(0, _DEPLOY_DIR)

from dispatcher import TutorDispatcher  # noqa: E402
from knowledge_store import seed_demo_data  # noqa: E402
from tutor_core import build_tool_registry  # noqa: E402

# One demo per division-of-labor scenario.
DEMOS: list[tuple[str, str]] = [
    ("课程内概念 → retrieve + explain_concept", "什么是 NumPy 广播机制？"),
    ("课程外知识 → retrieve + web_search", "Rust 的所有权系统是什么？请介绍。"),
    ("代码执行 → run_code", "帮我用 Python 算 1 加到 100 的和。"),
    ("出题 → retrieve + generate_quiz", "给我出两道 Pandas 的练习题。"),
]


async def ask(dispatcher: TutorDispatcher, question: str) -> None:
    print("=" * 72)
    print("问题:", question)
    result = await dispatcher.run(question)
    print("-" * 72)
    print(
        f"迭代 {result['iterations']}/{dispatcher.max_iterations}   "
        f"LLM 调用 {result['llm_calls']}/{dispatcher.max_llm_calls}"
    )
    agents = [e["sub_agent"] for e in result["evidence"]]
    print("派发的子代理:", agents if agents else "(直接 finalize，未调用工具)")

    for i, e in enumerate(result["evidence"], 1):
        print(f"\n  ┌─ 子代理[{i}] {e['sub_agent']}")
        print(f"  └─ 输出: {e['output']}")

    print("\n  [最终综合回答]")
    print(" ", result["answer"])
    print("=" * 72, "\n")


async def main() -> None:
    seed_demo_data()
    dispatcher = TutorDispatcher(
        tools=build_tool_registry(),
        max_iterations=5,
        max_llm_calls=8,
    )

    if len(sys.argv) > 1:
        await ask(dispatcher, sys.argv[1])
        return

    for label, question in DEMOS:
        print(f"\n### {label}")
        await ask(dispatcher, question)


if __name__ == "__main__":
    asyncio.run(main())
