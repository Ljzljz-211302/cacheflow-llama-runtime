import unittest

from llama_lab.scheduler_sim import (
    SimulatedSlot,
    TraceRequest,
    select_slot,
    simulate_trace,
)


class SchedulerSimulationTests(unittest.TestCase):
    def test_cost_aware_preserves_long_cache(self) -> None:
        slots = [
            SimulatedSlot(1, 480, 1),
            SimulatedSlot(2, 70, 2),
        ]
        request = TraceRequest(3, 85)
        # Different conversations share only the 20-token system prefix.
        upstream = select_slot(
            slots, request, policy="lcp", eviction_penalty=0, system_tokens=20
        )
        cost_aware = select_slot(
            slots, request, policy="cost_aware", eviction_penalty=0.5, system_tokens=20
        )
        self.assertEqual(upstream[0], 0)  # LRU tie break destroys the long cache.
        self.assertEqual(cost_aware[0], 1)

    def test_revisiting_cached_conversation_reduces_prefill(self) -> None:
        trace = [TraceRequest(1, 100), TraceRequest(2, 100), TraceRequest(1, 150)]
        result = simulate_trace(trace, slots_count=2, policy="lcp", system_tokens=20)
        self.assertEqual(result["prefill_tokens_total"], 250.0)

    def test_rejects_unknown_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            simulate_trace([], slots_count=1, policy="random")


if __name__ == "__main__":
    unittest.main()
