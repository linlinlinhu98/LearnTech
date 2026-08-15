"""Local unit tests for Module 3 (multi-agent tutor dispatcher).

Runs offline: no AgentScope, no LLM key, no network. The dispatcher is tested
with fake tools and a monkeypatched chat_completion_json; the code runner and
LLM budget are tested directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest

_DEPLOY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "deploy_starter")
)
sys.path.insert(0, _DEPLOY_DIR)

import code_runner
import dispatcher as dispatcher_mod
import llm_utils
import tutor_core
from knowledge_store import seed_demo_data


class CodeRunnerTests(unittest.TestCase):
    def test_arithmetic(self):
        r = code_runner.run_python("print(1 + 2)")
        self.assertTrue(r["success"], r)
        self.assertIn("3", r["stdout"])

    def test_math_module_allowed(self):
        r = code_runner.run_python("import math\nprint(math.sqrt(16))")
        self.assertTrue(r["success"], r)
        self.assertIn("4.0", r["stdout"])

    def test_os_import_blocked(self):
        r = code_runner.run_python("import os")
        self.assertFalse(r["success"], r)
        self.assertIn("not allowed", r["stderr"])

    def test_empty_code_rejected(self):
        r = code_runner.run_python("")
        self.assertFalse(r["success"])


class LlmBudgetTests(unittest.TestCase):
    def test_budget_accounting(self):
        b = llm_utils.LlmBudget(3)
        self.assertEqual(b.used, 0)
        self.assertTrue(b.can_charge(3))
        b.charge(2)
        self.assertEqual(b.used, 2)
        self.assertFalse(b.can_charge(2))
        self.assertTrue(b.can_charge(1))
        b.charge(1)
        self.assertTrue(b.exhausted)


class DispatcherTests(unittest.TestCase):
    def setUp(self):
        self._orig_json = dispatcher_mod.chat_completion_json
        self._orig_text = dispatcher_mod.chat_completion

    def tearDown(self):
        dispatcher_mod.chat_completion_json = self._orig_json
        dispatcher_mod.chat_completion = self._orig_text

    @staticmethod
    async def _fake_retrieve(query, top_k=5, course_id="", budget=None):
        return '{"results": [{"lesson_title": "测试课"}]}'

    @staticmethod
    async def _fake_web_search(query, budget=None):
        return "WEB_RESULT"

    @staticmethod
    async def _fake_run_code(code, budget=None):
        return '{"stdout": "ok"}'

    @staticmethod
    async def _fake_quiz(topic, difficulty="medium", budget=None):
        return "QUIZ_RESULT"

    @staticmethod
    async def _fake_explain(concept, level="beginner", budget=None):
        return "EXPLAIN_RESULT"

    def _make_dispatcher(self, max_iterations=5, max_llm_calls=8, tools=None):
        tools = tools or {
            "retrieve": self._fake_retrieve,
            "web_search": self._fake_web_search,
            "run_code": self._fake_run_code,
            "generate_quiz": self._fake_quiz,
            "explain_concept": self._fake_explain,
        }
        return dispatcher_mod.TutorDispatcher(tools, max_iterations, max_llm_calls)

    def _set_fake_llm(self, plan, answer="SYNTHESIZED"):
        async def fake_json(messages, temperature=0.3, max_tokens=2048):
            return plan

        async def fake_text(messages, temperature=0.3, max_tokens=2048):
            return answer

        dispatcher_mod.chat_completion_json = fake_json
        dispatcher_mod.chat_completion = fake_text

    def test_iteration_limit(self):
        self._set_fake_llm(
            plan={"action": "execute", "steps": [{"sub_agent": "retrieve"}]}
        )
        d = self._make_dispatcher(max_iterations=3, max_llm_calls=10)
        result = asyncio.run(d.run("问题"))
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(len(result["evidence"]), 3)
        self.assertEqual(result["llm_calls"], 4)  # 3 plans + 1 synthesize
        self.assertEqual(result["answer"], "SYNTHESIZED")

    def test_finalize_early(self):
        self._set_fake_llm(plan={"action": "finalize", "steps": []})
        d = self._make_dispatcher(max_iterations=5, max_llm_calls=10)
        result = asyncio.run(d.run("问题"))
        self.assertEqual(result["iterations"], 1)
        self.assertEqual(len(result["evidence"]), 0)
        self.assertEqual(result["llm_calls"], 2)

    def test_budget_reserves_synthesize(self):
        self._set_fake_llm(
            plan={"action": "execute", "steps": [{"sub_agent": "retrieve"}]}
        )
        d = self._make_dispatcher(max_iterations=5, max_llm_calls=3)
        result = asyncio.run(d.run("问题"))
        self.assertLessEqual(result["llm_calls"], 3)
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["llm_calls"], 3)

    def test_synthesize_fallback_on_budget_exhausted(self):
        async def charging_web_search(query, budget=None):
            if budget is not None and budget.can_charge(1):
                budget.charge(1)
            return "WEB_RESULT_FALLBACK"

        tools = {"retrieve": self._fake_retrieve, "web_search": charging_web_search}
        self._set_fake_llm(
            plan={"action": "execute", "steps": [{"sub_agent": "web_search"}]},
            answer="SHOULD_NOT_APPEAR",
        )
        d = self._make_dispatcher(max_iterations=5, max_llm_calls=2, tools=tools)
        result = asyncio.run(d.run("问题"))
        self.assertEqual(result["llm_calls"], 2)
        self.assertIn("WEB_RESULT_FALLBACK", result["answer"])
        self.assertNotEqual(result["answer"], "SHOULD_NOT_APPEAR")


class RetrieveTests(unittest.TestCase):
    def test_retrieve_returns_seeded_chunks(self):
        seed_demo_data()
        result = asyncio.run(tutor_core.retrieve_course("列表和字典"))
        data = json.loads(result)
        self.assertIn("results", data)
        self.assertGreater(len(data["results"]), 0)
        first = data["results"][0]
        for key in ("lesson_id", "lesson_title", "similarity", "text"):
            self.assertIn(key, first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
