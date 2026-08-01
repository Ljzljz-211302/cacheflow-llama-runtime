import tempfile
import unittest
from pathlib import Path

from llama_lab.report import render_report


class ReportTests(unittest.TestCase):
    def test_renders_tracked_acceptance_results(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.md"
            render_report(root / "results", output)
            text = output.read_text(encoding="utf-8")
        self.assertIn("长驻在线收敛与分布切换", text)
        self.assertIn("CUDA Profiling 因果链", text)
        self.assertIn("Paired upstream regression", text)


if __name__ == "__main__":
    unittest.main()
