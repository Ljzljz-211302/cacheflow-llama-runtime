import json
import threading
import time
import unittest
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llama_lab.benefit_protocol import run_staggered_wave
from llama_lab.streaming import stream_chat


class _ArrivalHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        prompt = payload["messages"][0]["content"]
        index = int(prompt.rsplit(" ", 1)[1])
        with self.server.arrival_lock:  # type: ignore[attr-defined]
            self.server.arrivals.append(index)  # type: ignore[attr-defined]
        body = (
            'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *args: object) -> None:
        pass


class BenefitProtocolTests(unittest.TestCase):
    def test_staggered_wave_orders_the_real_http_send_seam(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _ArrivalHandler)
        server.arrivals = []
        server.arrival_lock = threading.Lock()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"

            def worker(
                index: int, send_guard: AbstractContextManager[None]
            ) -> str:
                if index == 0:
                    # Reproduces the review finding: later workers reach their
                    # call site first, but may not overtake the registered send.
                    time.sleep(0.040)
                result = stream_chat(
                    base_url,
                    f"request {index}",
                    max_tokens=1,
                    send_guard=send_guard,
                )
                return str(result["text"])

            wave = run_staggered_wave(
                list(range(6)), worker, max_workers=4, admission_stagger_s=0.005
            )

            self.assertEqual(server.arrivals, list(range(6)))
            self.assertEqual(wave.observed_send_order, tuple(range(6)))
            self.assertEqual(wave.results, ("x",) * 6)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_staggered_wave_rejects_nonpositive_concurrency(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers"):
            run_staggered_wave([1], lambda value, guard: value, max_workers=0)

    def test_staggered_wave_preserves_none_results(self) -> None:
        def worker(
            _value: int, send_guard: AbstractContextManager[None]
        ) -> None:
            with send_guard:
                pass
            return None

        wave = run_staggered_wave([1, 2], worker, max_workers=2)
        self.assertEqual(wave.results, (None, None))
        self.assertEqual(wave.observed_send_order, (0, 1))


if __name__ == "__main__":
    unittest.main()
