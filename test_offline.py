"""Offline tests (no API key needed):  python -m unittest test_offline -v"""
import json
import unittest
from types import SimpleNamespace as NS

import config
from chatbot import SupportBot
from tools import ToolBox

TOOLBOX = ToolBox()


class RetrievalTests(unittest.TestCase):
    def assert_top_source(self, query: str, expected: str) -> None:
        hits = TOOLBOX.retriever.search(query, top_k=3)
        self.assertTrue(hits, f"no hits for {query!r}")
        self.assertEqual(hits[0][0].source, expected, f"{query!r} -> {[h[0].id for h in hits]}")

    def test_queries_hit_the_right_article(self):
        self.assert_top_source("how long do refunds take to my card", "returns_refunds.md")
        self.assert_top_source("international shipping customs duties", "shipping.md")
        self.assert_top_source("my payment was declined", "payments.md")
        self.assert_top_source("forgot my password reset", "account_orders.md")
        self.assert_top_source("warranty claim for broken headphones", "warranty_membership.md")

    def test_unrelated_query_returns_nothing(self):
        self.assertEqual(TOOLBOX.retriever.search("zzzz qqqq"), [])


class ToolTests(unittest.TestCase):
    def test_order_lookup_requires_matching_email(self):
        ok, err = TOOLBOX.run("lookup_order", {"order_id": "ord-10042", "email": "John.Doe@example.com"})
        self.assertFalse(err)
        self.assertEqual(json.loads(ok)["order"]["status"], "Shipped")
        self.assertNotIn("email", json.loads(ok)["order"])

        bad, _ = TOOLBOX.run("lookup_order", {"order_id": "ORD-10042", "email": "attacker@example.com"})
        self.assertFalse(json.loads(bad)["found"])

    def test_product_search(self):
        out, _ = TOOLBOX.run("search_products", {"query": "espresso"})
        self.assertEqual(json.loads(out)["products"][0]["sku"], "CM-ESP")

    def test_ticket_rejects_bad_email(self):
        _, is_error = TOOLBOX.run("create_support_ticket",
                                  {"email": "nope", "category": "other", "summary": "x", "order_id": ""})
        self.assertTrue(is_error)

    def test_unknown_tool(self):
        _, is_error = TOOLBOX.run("delete_everything", {})
        self.assertTrue(is_error)


class FakeClient:
    """Mimics client.beta.messages.create with scripted responses."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kwargs):
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return next(self._responses)


def text(t):
    return NS(type="text", text=t)


class LoopTests(unittest.TestCase):
    def test_rag_tool_loop(self):
        tool_use = NS(type="tool_use", id="tu_1", name="search_knowledge_base", input={"query": "refund time"})
        client = FakeClient([
            NS(stop_reason="tool_use", content=[tool_use]),
            NS(stop_reason="end_turn", content=[text("Card refunds take 5-10 business days [returns_refunds.md].")]),
        ])
        bot = SupportBot(client=client, toolbox=TOOLBOX)
        reply = bot.chat("How long do refunds take?")

        self.assertIn("5-10 business days", reply.text)
        self.assertEqual(reply.tool_calls[0]["name"], "search_knowledge_base")
        second = client.calls[1]
        self.assertEqual(second["model"], config.MODEL)
        self.assertEqual(second["fallbacks"], "default")
        tool_result = second["messages"][-1]["content"][0]
        self.assertEqual(tool_result["tool_use_id"], "tu_1")
        self.assertIn("returns_refunds.md", tool_result["content"])
        # user, assistant(tool_use), user(tool_result), assistant(final)
        self.assertEqual([m["role"] for m in bot.messages], ["user", "assistant", "user", "assistant"])

    def test_refusal_rolls_back_turn(self):
        client = FakeClient([NS(stop_reason="refusal", content=[])])
        bot = SupportBot(client=client, toolbox=TOOLBOX)
        reply = bot.chat("something off-limits")
        self.assertIn("can't help", reply.text)
        self.assertEqual(bot.messages, [])


if __name__ == "__main__":
    unittest.main()
