import unittest

from cacheflow.routing import (
    EwmaCostModel,
    ModelArchitecture,
    ModelProfile,
    ModelRouter,
    RouteRequest,
)


ARCH = ModelArchitecture(24, 2, 64, 32768)


def profile(
    name: str,
    *,
    quality: float,
    weight: float,
    prefill: float,
    decode: float,
    max_slots: int = 4,
) -> ModelProfile:
    return ModelProfile(
        name=name,
        architecture=ARCH,
        quality_score=quality,
        weight_mib=weight,
        runtime_mib=200,
        cache_bytes_per_element=2,
        prefill_ms_per_token=prefill,
        decode_ms_per_token=decode,
        max_slots=max_slots,
    )


class CacheFlowRoutingTests(unittest.TestCase):
    def test_kv_cost_uses_model_geometry_and_active_sequences(self) -> None:
        candidate = profile("q4", quality=0.7, weight=500, prefill=0.1, decode=3)
        self.assertAlmostEqual(candidate.kv_mib(4096, 1), 48.0)
        self.assertAlmostEqual(candidate.kv_mib(4096, 4), 192.0)

    def test_quality_floor_excludes_fast_but_weak_model(self) -> None:
        router = ModelRouter(
            [
                profile("fast", quality=0.5, weight=500, prefill=0.05, decode=1),
                profile("accurate", quality=0.9, weight=1200, prefill=0.2, decode=3),
            ]
        )
        decision = router.route(RouteRequest(512, 64, 0.8, 1000, available_vram_mib=4000))
        self.assertEqual(decision.selected_model, "accurate")
        self.assertIn("quality_floor", decision.candidates[0].reasons)

    def test_vram_budget_accounts_for_weights_and_kv(self) -> None:
        router = ModelRouter(
            [
                profile("large", quality=0.95, weight=3000, prefill=0.1, decode=2),
                profile("small", quality=0.8, weight=500, prefill=0.2, decode=3),
            ]
        )
        decision = router.route(
            RouteRequest(4096, 128, 0.7, 2000, active_sequences=4, available_vram_mib=2500)
        )
        self.assertEqual(decision.selected_model, "small")
        self.assertIn("vram_budget", decision.candidates[0].reasons)

    def test_ewma_adapts_to_observed_slowdown_in_context_bucket(self) -> None:
        fast = profile("fast", quality=0.8, weight=500, prefill=0.05, decode=1)
        stable = profile("stable", quality=0.8, weight=600, prefill=0.1, decode=2)
        cost_model = EwmaCostModel(alpha=1.0)
        router = ModelRouter([fast, stable], cost_model, quality_weight=0, memory_weight=0)
        request = RouteRequest(512, 64, 0.5, 1000, available_vram_mib=4000)
        self.assertEqual(router.route(request).selected_model, "fast")
        router.observe("fast", request, actual_prefill_ms=512, actual_decode_ms=640)
        self.assertEqual(router.route(request).selected_model, "stable")

    def test_decision_trace_explains_every_rejection(self) -> None:
        router = ModelRouter([profile("only", quality=0.4, weight=500, prefill=1, decode=1)])
        with self.assertRaisesRegex(RuntimeError, "quality_floor"):
            router.route(RouteRequest(100, 10, 0.9, 1000, available_vram_mib=4000))


if __name__ == "__main__":
    unittest.main()
