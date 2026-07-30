import subprocess
import unittest
from unittest.mock import patch

from llama_lab.gpu_memory import GpuMemorySampler, query_gpu_used_mib


class GpuMemoryTests(unittest.TestCase):
    @patch("llama_lab.gpu_memory.subprocess.run")
    def test_parses_nvidia_smi_memory(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess([], 0, "712\n", "")
        self.assertEqual(query_gpu_used_mib(), 712.0)

    @patch("llama_lab.gpu_memory.subprocess.run", side_effect=FileNotFoundError)
    def test_missing_nvidia_smi_is_optional(self, _run_mock) -> None:
        self.assertIsNone(query_gpu_used_mib())

    def test_increment_is_relative_to_baseline(self) -> None:
        sampler = GpuMemorySampler()
        sampler.baseline_mib = 400.0
        sampler.peak_mib = 950.0
        self.assertEqual(sampler.increment_mib, 550.0)


if __name__ == "__main__":
    unittest.main()
