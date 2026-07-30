import unittest

from llama_lab.prometheus import parse_prometheus_text, require_engine_metrics


class PrometheusTests(unittest.TestCase):
    def test_parses_samples_and_ignores_comments(self) -> None:
        samples = parse_prometheus_text(
            "# HELP ignored help\nllamacpp:kv_cache_tokens 42\nmetric{slot=\"0\"} 1.5e2\n"
        )
        self.assertEqual(samples["llamacpp:kv_cache_tokens"], 42.0)
        self.assertEqual(samples["metric"], 150.0)

    def test_requires_patched_metric_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "patched engine metrics missing"):
            require_engine_metrics({"llamacpp:kv_cache_tokens": 1.0})


if __name__ == "__main__":
    unittest.main()
