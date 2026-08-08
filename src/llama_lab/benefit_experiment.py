from __future__ import annotations

import re
from dataclasses import dataclass


_SAMPLE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?\s+'
    r'(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$'
)
_LABEL = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>(?:\\.|[^"])*)"')


def labeled_metric(
    text: str, name: str, labels: dict[str, str] | None = None
) -> float:
    expected = labels or {}
    for raw_line in text.splitlines():
        match = _SAMPLE.match(raw_line.strip())
        if not match or match.group("name") != name:
            continue
        actual = {
            label.group("key"): label.group("value")
            for label in _LABEL.finditer(match.group("labels") or "")
        }
        if all(actual.get(key) == value for key, value in expected.items()):
            return float(match.group("value"))
    raise ValueError(f"missing Prometheus metric {name} labels={expected}")


@dataclass(frozen=True)
class BenefitSnapshot:
    upstream_decisions: int
    cacheflow_decisions: int
    exploration_decisions: int
    positive_decisions: int
    drift_events: int
    safety_fallbacks: int
    cooldown_remaining: int
    predicted_benefit_ms: float
    uncertainty_ms: float

    @classmethod
    def from_prometheus(cls, text: str, backend: str) -> "BenefitSnapshot":
        backend_label = {"backend": backend}

        def counter(name: str, extra: dict[str, str] | None = None) -> int:
            labels = dict(backend_label)
            labels.update(extra or {})
            return int(labeled_metric(text, name, labels))

        return cls(
            upstream_decisions=counter(
                "llamacpp:benefit_decisions_total", {"action": "upstream"}
            ),
            cacheflow_decisions=counter(
                "llamacpp:benefit_decisions_total", {"action": "cacheflow"}
            ),
            exploration_decisions=counter("llamacpp:benefit_exploration_total"),
            positive_decisions=counter(
                "llamacpp:benefit_reason_total", {"reason": "positive_lower_bound"}
            ),
            drift_events=counter("llamacpp:benefit_drift_total"),
            safety_fallbacks=counter("llamacpp:benefit_safety_fallback_total"),
            cooldown_remaining=counter("llamacpp:benefit_cooldown_remaining"),
            predicted_benefit_ms=labeled_metric(
                text, "llamacpp:benefit_predicted_benefit_ms", backend_label
            ),
            uncertainty_ms=labeled_metric(
                text, "llamacpp:benefit_uncertainty_ms", backend_label
            ),
        )

    def delta(self, earlier: "BenefitSnapshot") -> "BenefitSnapshot":
        counter_names = (
            "upstream_decisions",
            "cacheflow_decisions",
            "exploration_decisions",
            "positive_decisions",
            "drift_events",
            "safety_fallbacks",
        )
        counters: dict[str, int] = {}
        for name in counter_names:
            value = int(getattr(self, name)) - int(getattr(earlier, name))
            if value < 0:
                raise ValueError(f"benefit counter regressed: {name}")
            counters[name] = value
        return BenefitSnapshot(
            **counters,
            cooldown_remaining=self.cooldown_remaining,
            predicted_benefit_ms=self.predicted_benefit_ms,
            uncertainty_ms=self.uncertainty_ms,
        )


@dataclass(frozen=True)
class PhaseEvidence:
    phase: str
    upstream_decisions: int
    cacheflow_decisions: int
    exploration_decisions: int
    positive_decisions: int
    drift_events: int
    safety_fallbacks: int
    ttft_p95_ms: float
    predicted_benefit_ms: float
    uncertainty_ms: float
    positive_waves: int = 0
    max_consecutive_positive_waves: int = 0
    terminal_consecutive_positive_waves: int = 0


@dataclass(frozen=True)
class LongLivedAcceptance:
    minimum_cold_upstream_decisions: int = 1
    minimum_positive_decisions: int = 1
    maximum_ttft_ms: float = 500.0
    minimum_consecutive_positive_waves: int = 3


@dataclass(frozen=True)
class AcceptanceResult:
    passed: bool
    violations: tuple[str, ...]


def evaluate_long_lived(
    phases: list[PhaseEvidence], acceptance: LongLivedAcceptance
) -> AcceptanceResult:
    by_name = {phase.phase: phase for phase in phases}
    required = {"cold_start", "stable_reuse", "distribution_shift"}
    missing = sorted(required - by_name.keys())
    if missing:
        return AcceptanceResult(False, (f"missing phases: {', '.join(missing)}",))

    cold = by_name["cold_start"]
    stable = by_name["stable_reuse"]
    shift = by_name["distribution_shift"]
    violations: list[str] = []
    if cold.upstream_decisions < acceptance.minimum_cold_upstream_decisions:
        violations.append("cold start did not establish an upstream baseline")
    if cold.positive_decisions:
        violations.append("cold start enabled CacheFlow before confidence evidence")
    if cold.cacheflow_decisions:
        violations.append("cold start executed CacheFlow before the stable phase")
    if stable.positive_decisions < acceptance.minimum_positive_decisions:
        violations.append("stable phase produced no positive-lower-bound decision")
    if stable.positive_decisions > stable.cacheflow_decisions - stable.exploration_decisions:
        violations.append("positive decisions are inconsistent with non-exploration actions")
    if stable.max_consecutive_positive_waves < acceptance.minimum_consecutive_positive_waves:
        violations.append("positive-lower-bound enablement was not persistent across waves")
    if (
        stable.terminal_consecutive_positive_waves
        < acceptance.minimum_consecutive_positive_waves
    ):
        violations.append(
            "positive-lower-bound enablement was absent from terminal waves"
        )
    if shift.cacheflow_decisions or shift.positive_decisions:
        violations.append("distribution shift continued to enable CacheFlow")
    if shift.drift_events + shift.safety_fallbacks == 0:
        violations.append("distribution shift triggered neither drift nor safety fallback")
    if stable.ttft_p95_ms > acceptance.maximum_ttft_ms:
        violations.append("stable learned TTFT exceeded the configured SLO")
    if shift.ttft_p95_ms > acceptance.maximum_ttft_ms:
        violations.append("post-shift TTFT exceeded the configured SLO")
    return AcceptanceResult(not violations, tuple(violations))
