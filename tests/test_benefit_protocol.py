import threading
import time
import unittest

from llama_lab.benefit_protocol import run_staggered_wave


class BenefitProtocolTests(unittest.TestCase):
    def test_staggered_wave_admits_requests_in_registered_order(self) -> None:
        admitted: list[int] = []
        admitted_lock = threading.Lock()

        def worker(index: int) -> int:
            with admitted_lock:
                admitted.append(index)
            # Keep requests overlapping so the test covers concurrent execution,
            # rather than accidentally passing through serialized calls.
            time.sleep(0.025)
            return index * 10

        results = run_staggered_wave(
            list(range(6)), worker, max_workers=4, admission_stagger_s=0.005
        )

        self.assertEqual(admitted, list(range(6)))
        self.assertEqual(results, [0, 10, 20, 30, 40, 50])

    def test_staggered_wave_rejects_nonpositive_concurrency(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_workers"):
            run_staggered_wave([1], lambda value: value, max_workers=0)

    def test_staggered_wave_preserves_none_results(self) -> None:
        self.assertEqual(
            run_staggered_wave([1, 2], lambda _value: None, max_workers=2),
            [None, None],
        )


if __name__ == "__main__":
    unittest.main()
