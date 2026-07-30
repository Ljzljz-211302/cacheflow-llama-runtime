import unittest

from llama_lab.metrics import percentile, summarize_latency


class MetricsTests(unittest.TestCase):
    def test_percentile_interpolates(self) -> None:
        self.assertEqual(percentile([10, 20, 30, 40], 0.5), 25)
        self.assertAlmostEqual(percentile([10, 20, 30, 40], 0.95), 38.5)

    def test_summary_reports_tail_latency(self) -> None:
        summary = summarize_latency(
            [
                {"ttft_ms": 10, "tpot_ms": 5, "total_ms": 100, "output_tps": 20},
                {"ttft_ms": 30, "tpot_ms": 10, "total_ms": 200, "output_tps": 10},
            ]
        )
        self.assertEqual(summary["ttft_p50_ms"], 20)
        self.assertAlmostEqual(summary["ttft_p95_ms"], 29)
        self.assertAlmostEqual(summary["tpot_p95_ms"], 9.75)
        self.assertEqual(summary["mean_output_tps"], 15)


if __name__ == "__main__":
    unittest.main()
