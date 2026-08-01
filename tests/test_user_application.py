import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from interview_assistant.knowledge import KnowledgeIndex
from interview_assistant.llama_client import LlamaClient
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


class ClosingModel(FakeModel):
    def __init__(self) -> None:
        self.closed = False

    def stream(self, messages: list[dict[str, str]], max_tokens: int = 512):
        try:
            yield "尚未完成的答案"
            yield "不应保存"
        finally:
            self.closed = True


class RedirectHandler(BaseHTTPRequestHandler):
    visited: list[str] = []

    def do_GET(self) -> None:
        self.visited.append(self.path)
        if self.path == "/health":
            self.send_response(302)
            self.send_header("Location", "/credential-sink")
            self.end_headers()
        else:
            self.send_response(200)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return None


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
        (knowledge_root / "generic-questions.md").write_text(
            "# 面试自测题\n为什么通常使用某种结构？请解释并给出一个面试追问。",
            encoding="utf-8",
        )
        (knowledge_root / "database.md").write_text(
            "# B+ 树索引\nB+ 树高扇出、低高度，叶子节点有序相连，适合点查和范围查询。",
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

    def test_rare_technical_term_beats_generic_question_wording(self) -> None:
        citations = self.index.search("为什么数据库通常使用 B+ 树？")
        self.assertEqual(citations[0].title, "B+ 树索引")

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

    def test_model_client_rejects_remote_or_ambiguous_key_destinations(self) -> None:
        rejected = [
            "https://example.com:443",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:8080/proxy",
            "http://user@127.0.0.1:8080",
        ]
        for url in rejected:
            with self.subTest(url=url), self.assertRaises(ValueError):
                LlamaClient(url, "secret")
        self.assertEqual(LlamaClient("http://127.0.0.1:8080", "secret").base_url, "http://127.0.0.1:8080")

    def test_model_client_does_not_follow_redirects(self) -> None:
        RedirectHandler.visited = []
        redirect_server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        thread = threading.Thread(target=redirect_server.serve_forever, daemon=True)
        thread.start()
        try:
            client = LlamaClient(f"http://127.0.0.1:{redirect_server.server_port}", "secret")
            self.assertFalse(client.healthy())
            self.assertEqual(RedirectHandler.visited, ["/health"])
        finally:
            redirect_server.shutdown()
            redirect_server.server_close()
            thread.join(timeout=5)

    def test_no_retrieval_match_fails_closed_without_calling_model(self) -> None:
        session = self.store.create_session()
        stream = self.service.answer(session["id"], "量子色动力学重整化群")
        with self.assertRaisesRegex(ValueError, "没有找到可引用依据"):
            next(stream)
        self.assertEqual(self.store.messages(session["id"]), [])

    def test_cancelled_stream_closes_model_and_leaves_no_partial_assistant(self) -> None:
        session = self.store.create_session()
        model = ClosingModel()
        service = InterviewService(self.store, self.index, model)
        stream = service.answer(session["id"], "解释偏差与方差")
        self.assertEqual(next(stream)["type"], "citations")
        self.assertEqual(next(stream)["type"], "delta")
        self.assertEqual(next(stream)["type"], "delta")
        stream.close()
        self.assertTrue(model.closed)
        messages = self.store.messages(session["id"])
        self.assertEqual([message["role"] for message in messages], ["user"])

    def test_message_and_session_timestamp_are_one_transaction(self) -> None:
        session = self.store.create_session()
        with self.store._connection() as connection:
            connection.execute(
                """CREATE TRIGGER reject_session_update BEFORE UPDATE ON sessions
                   BEGIN SELECT RAISE(ABORT, 'injected update failure'); END"""
            )
        try:
            with self.assertRaises(sqlite3.IntegrityError):
                self.store.add_message(session["id"], "user", "must roll back")
        finally:
            with self.store._connection() as connection:
                connection.execute("DROP TRIGGER reject_session_update")
        self.assertEqual(self.store.messages(session["id"]), [])


if __name__ == "__main__":
    unittest.main()
