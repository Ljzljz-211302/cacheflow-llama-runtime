import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from interview_assistant.knowledge import KnowledgeIndex
from interview_assistant.server import create_server
from interview_assistant.service import InterviewService
from interview_assistant.store import ConversationStore


class FakeModel:
    def healthy(self) -> bool:
        return True

    def stream(self, messages: list[dict[str, str]], max_tokens: int = 512):
        assert "[资料1]" in messages[0]["content"]
        yield "偏差衡量欠拟合，"
        yield "方差衡量对数据扰动的敏感性。[资料1]"


class UserApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        knowledge_root = self.root / "knowledge"
        knowledge_root.mkdir()
        (knowledge_root / "machine-learning.md").write_text(
            "# 偏差与方差\n偏差描述模型预测期望与真实目标之间的差异。"
            "方差描述训练集扰动造成的预测变化。\n# 支持向量机\n间隔最大化。",
            encoding="utf-8",
        )
        self.index = KnowledgeIndex([knowledge_root], max_chunk_chars=200)
        self.db = self.root / "assistant.db"
        self.store = ConversationStore(self.db)
        self.service = InterviewService(self.store, self.index, FakeModel())

    def tearDown(self) -> None:
        self.store.close()
        self.tempdir.cleanup()

    def test_retrieval_returns_auditable_source(self) -> None:
        citations = self.index.search("什么是偏差和方差？")
        self.assertTrue(citations)
        self.assertEqual(citations[0].title, "偏差与方差")
        self.assertEqual(citations[0].source, "knowledge/machine-learning.md")

    def test_completed_answer_and_citations_survive_store_restart(self) -> None:
        session = self.store.create_session("机器学习模拟面试")
        events = list(self.service.answer(session["id"], "解释偏差与方差"))
        self.assertEqual(events[0]["type"], "citations")
        self.assertEqual(events[-1]["type"], "done")

        reopened = ConversationStore(self.db)
        try:
            messages = reopened.messages(session["id"])
            self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
            self.assertIn("资料1", messages[1]["content"])
            self.assertEqual(messages[1]["citations"][0]["title"], "偏差与方差")
        finally:
            reopened.close()

    def test_http_user_journey_creates_session_streams_and_reads_history(self) -> None:
        server = create_server("127.0.0.1", 0, self.service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base + "/") as response:
                self.assertIn("面试学习助手", response.read().decode("utf-8"))
            session = self._post(base + "/api/sessions", {"title": "真实用户旅程"})
            request = urllib.request.Request(
                base + f"/api/sessions/{session['id']}/messages",
                data=json.dumps({"content": "解释偏差与方差"}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request) as response:
                stream = response.read().decode("utf-8")
            self.assertIn('"type":"citations"', stream)
            self.assertIn('"type":"delta"', stream)
            self.assertIn('"type":"done"', stream)
            with urllib.request.urlopen(base + f"/api/sessions/{session['id']}/messages") as response:
                history = json.load(response)["messages"]
            self.assertEqual(len(history), 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    @staticmethod
    def _post(url: str, body: dict[str, str]) -> dict[str, str]:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            return json.load(response)

    def test_non_loopback_binding_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            create_server("0.0.0.0", 0, self.service)


if __name__ == "__main__":
    unittest.main()
