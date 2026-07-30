import json
import unittest
import urllib.error
import urllib.request

from cacheflow.api import CacheFlowApiServer
from tests.test_cacheflow_service import FakeExecutor, make_engine


class CacheFlowApiTests(unittest.TestCase):
    def setUp(self):
        self.executor = FakeExecutor()
        self.engine = make_engine(self.executor)
        self.server = CacheFlowApiServer(self.engine, port=0).start()
        host, port = self.server.address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.engine.shutdown()

    def request(self, path, *, method="GET", body=None, headers=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return response, json.load(response)

    def test_openai_endpoint_exposes_control_plane_decision(self):
        response, body = self.request(
            "/v1/chat/completions",
            method="POST",
            headers={"X-Conversation-ID": "conversation-a"},
            body={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4,
                "cacheflow": {"quality_floor": 0.7},
            },
        )
        self.assertEqual(response.status, 200)
        self.assertEqual(response.headers["X-CacheFlow-Model"], "q4")
        self.assertEqual(body["cacheflow"]["model"], "q4")
        self.assertEqual(body["cacheflow"]["routing"]["selected_model"], "q4")

    def test_health_metrics_and_debug_state(self):
        _, health = self.request("/health")
        _, state = self.request("/debug/state")
        with urllib.request.urlopen(self.base_url + "/metrics", timeout=2) as response:
            metrics = response.read().decode()
        self.assertEqual(health, {"status": "ok"})
        self.assertIn("scheduler", state)
        self.assertIn("cacheflow_queue_depth", metrics)

    def test_streaming_is_rejected_explicitly(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request(
                "/v1/chat/completions",
                method="POST",
                body={"messages": [{"role": "user", "content": "x"}], "stream": True},
            )
        self.assertEqual(caught.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
