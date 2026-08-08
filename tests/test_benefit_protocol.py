import json
import threading
import time
import unittest
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llama_lab.benefit_protocol import (
    evaluate_short_lived_acceptance,
    complete_latin_orders,
    prometheus_histogram_quantile_upper_bound,
    joint_williams_orders,
    run_staggered_wave,
)
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
    def test_histogram_quantile_uses_cumulative_prometheus_buckets(self) -> None:
        payload = """
llamacpp:benefit_choose_duration_us_bucket{backend="cuda",le="1"} 3
llamacpp:benefit_choose_duration_us_bucket{backend="cuda",le="5"} 8
llamacpp:benefit_choose_duration_us_bucket{backend="cuda",le="50"} 99
llamacpp:benefit_choose_duration_us_bucket{backend="cuda",le="+Inf"} 100
"""
        self.assertEqual(
            prometheus_histogram_quantile_upper_bound(
                payload,
                "benefit_choose_duration_us",
                {"backend": "cuda"},
                0.99,
            ),
            50.0,
        )

    def test_zero_intervention_requires_bounded_choose_latency(self) -> None:
        rows = [
            {
                "upstream_regression_ratio": 0.08,
                "cacheflow_decisions": 0.0,
                "exploration_decisions": 0.0,
                "benefit_choose_samples": 10.0,
                "benefit_choose_p99_us": 20.0,
            },
            {
                "upstream_regression_ratio": 0.06,
                "cacheflow_decisions": 0.0,
                "exploration_decisions": 0.0,
                "benefit_choose_samples": 10.0,
                "benefit_choose_p99_us": 50.0,
            },
        ]
        result = evaluate_short_lived_acceptance(
            rows, maximum_regression=0.03, maximum_choose_p99_us=50.0
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.performance_status, "inconclusive_no_intervention")
        self.assertEqual(result.non_probe_cacheflow_decisions, 0)

        rows[1]["benefit_choose_p99_us"] = 100.0
        slow = evaluate_short_lived_acceptance(
            rows, maximum_regression=0.03, maximum_choose_p99_us=50.0
        )
        self.assertFalse(slow.passed)
        self.assertIn("choose P99", slow.violation)

    def test_applied_policy_still_requires_wall_clock_non_regression(self) -> None:
        rows = [
            {
                "upstream_regression_ratio": 0.06,
                "cacheflow_decisions": 4.0,
                "exploration_decisions": 1.0,
                "benefit_choose_samples": 10.0,
                "benefit_choose_p99_us": 20.0,
            }
        ]
        result = evaluate_short_lived_acceptance(
            rows, maximum_regression=0.03, maximum_choose_p99_us=50.0
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.performance_status, "regressed_with_intervention")

    def test_choose_acceptance_recomputes_p99_and_count_from_raw_buckets(self) -> None:
        row = {
            "upstream_regression_ratio": 0.0,
            "upstream_decisions": 9.0,
            "cacheflow_decisions": 1.0,
            "exploration_decisions": 1.0,
            "benefit_choose_samples": 10.0,
            "benefit_choose_p99_us": 5.0,
            "benefit_choose_histogram_json": json.dumps(
                {"1": 2, "2": 8, "5": 10, "+Inf": 10}
            ),
        }
        result = evaluate_short_lived_acceptance(
            [row], maximum_regression=0.03, maximum_choose_p99_us=50.0
        )
        self.assertTrue(result.passed)

        row["benefit_choose_p99_us"] = 2.0
        with self.assertRaisesRegex(ValueError, "P99"):
            evaluate_short_lived_acceptance(
                [row], maximum_regression=0.03, maximum_choose_p99_us=50.0
            )

    def test_latin_orders_require_complete_four_trial_blocks(self) -> None:
        modes = ("upstream", "always", "rule", "learned")
        with self.assertRaisesRegex(ValueError, "complete Latin"):
            complete_latin_orders(modes, 10)

        orders = complete_latin_orders(modes, 12)
        self.assertEqual(len(orders), 12)
        for mode in modes:
            positions = [order.index(mode) for order in orders]
            self.assertEqual([positions.count(index) for index in range(4)], [3] * 4)

        # Each directed immediate predecessor pair occurs once per four-trial
        # block. This is stronger than cyclic rotation and controls first-order
        # thermal/process carryover between fresh server processes.
        for block_start in range(0, len(orders), 4):
            pair_counts = {
                (left, right): 0
                for left in modes
                for right in modes
                if left != right
            }
            for order in orders[block_start : block_start + 4]:
                for pair in zip(order, order[1:]):
                    pair_counts[pair] += 1
            self.assertEqual(set(pair_counts.values()), {1})

    def test_two_backend_orders_balance_shared_machine_state(self) -> None:
        orders = complete_latin_orders(("cpu", "cuda"), 12)
        self.assertEqual(orders.count(("cpu", "cuda")), 6)
        self.assertEqual(orders.count(("cuda", "cpu")), 6)

    def test_joint_backend_mode_orders_balance_actual_process_predecessors(self) -> None:
        treatments = tuple(
            (backend, mode)
            for backend in ("cpu", "cuda")
            for mode in ("upstream", "always", "rule", "learned")
        )
        with self.assertRaisesRegex(ValueError, "complete Latin"):
            complete_latin_orders(treatments, 12)

        orders = joint_williams_orders(
            ("cpu", "cuda"),
            ("upstream", "always", "rule", "learned"),
            16,
        )
        position_counts = {
            (treatment, position): 0
            for treatment in treatments
            for position in range(len(treatments))
        }
        predecessor_counts = {
            (left, right): 0
            for left in treatments
            for right in treatments
            if left != right
        }
        for order in orders:
            for position, treatment in enumerate(order):
                position_counts[(treatment, position)] += 1
            for pair in zip(order, order[1:]):
                predecessor_counts[pair] += 1

        self.assertEqual(set(position_counts.values()), {2})
        self.assertEqual(set(predecessor_counts.values()), {2})

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
