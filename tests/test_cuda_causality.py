import unittest

from llama_lab.cuda_causality import CudaProfileTrial, analyze_cuda_causality


class CudaCausalityTests(unittest.TestCase):
    def test_requires_intervention_mediator_cuda_and_outcome_chain(self) -> None:
        rows = []
        for trial in range(1, 4):
            rows.extend(
                [
                    CudaProfileTrial(
                        "upstream", trial, 0, 12, 1536, 0, 0, 0.0,
                        0.62, 300.0, 210.0, 18000.0,
                    ),
                    CudaProfileTrial(
                        "always", trial, 18, 28, 1280, 4, 98304, 0.42,
                        0.78, 100.0, 165.0, 16000.0,
                    ),
                ]
            )
        result = analyze_cuda_causality(rows, minimum_trials=3)
        self.assertTrue(result.passed)
        self.assertLess(result.ttft_delta_ms, 0)
        self.assertGreater(result.gpu_busy_delta, 0)
        self.assertGreater(result.kernel_launch_delta, 0)

    def test_rejects_scheduler_change_without_cuda_level_evidence(self) -> None:
        rows = [
            CudaProfileTrial("upstream", 1, 0, 10, 1000, 0, 0, 0, 0.5, 100, 200, 1000),
            CudaProfileTrial("always", 1, 5, 20, 900, 0, 0, 0, 0.5, 100, 180, 900),
        ]
        result = analyze_cuda_causality(rows, minimum_trials=1)
        self.assertFalse(result.passed)
        self.assertTrue(any("CUDA mediator" in item for item in result.violations))

    def test_pairs_trials_instead_of_comparing_unmatched_processes(self) -> None:
        rows = [
            CudaProfileTrial("upstream", 1, 0, 10, 1000, 0, 0, 0, 0.5, 100, 200, 1000),
            CudaProfileTrial("always", 2, 5, 20, 900, 1, 10, 0.1, 0.6, 80, 180, 900),
        ]
        with self.assertRaisesRegex(ValueError, "paired"):
            analyze_cuda_causality(rows, minimum_trials=1)

    def test_rejects_cuda_noise_without_material_system_outcome(self) -> None:
        rows = [
            CudaProfileTrial("upstream", 1, 0, 10, 1000, 0, 100, 0.1, 0.5, 100, 200, 10000),
            CudaProfileTrial("always", 1, 5, 20, 900, 0, 200, 0.2, 0.6, 80, 201, 10500),
        ]
        result = analyze_cuda_causality(rows, minimum_trials=1)
        self.assertFalse(result.passed)
        self.assertTrue(any("material" in item for item in result.violations))


if __name__ == "__main__":
    unittest.main()
