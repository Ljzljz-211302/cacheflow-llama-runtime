import tempfile
import unittest
from pathlib import Path

from llama_lab.advisor import (
    MIB,
    ModelArchitecture,
    estimate_kv_cache_mib,
    estimate_memory,
    recommend_largest_context,
)


class AdvisorTests(unittest.TestCase):
    architecture = ModelArchitecture(
        layers=24, kv_heads=2, head_dim=64, max_context=32768
    )

    def test_qwen_kv_cache_formula(self) -> None:
        self.assertEqual(
            estimate_kv_cache_mib(self.architecture, context=4096), 48.0
        )

    def test_rejects_context_above_model_limit(self) -> None:
        with self.assertRaises(ValueError):
            estimate_kv_cache_mib(self.architecture, context=65536)

    def test_recommends_largest_fitting_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"0" * MIB)
            recommendation = recommend_largest_context(
                model,
                self.architecture,
                available_mib=900,
                candidates=(4096, 8192, 16384),
            )
            self.assertIsNotNone(recommendation)
            self.assertEqual(recommendation.context, 16384)

    def test_marks_oversized_model_as_not_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"0" * (2 * MIB))
            estimate = estimate_memory(
                model,
                self.architecture,
                context=4096,
                available_mib=100,
            )
            self.assertFalse(estimate.fits)


if __name__ == "__main__":
    unittest.main()
