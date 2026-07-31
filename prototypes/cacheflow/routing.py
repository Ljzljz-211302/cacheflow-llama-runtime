"""Archived prototype model router; it is outside the production runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf


MIB = 1024 * 1024


@dataclass(frozen=True)
class ModelArchitecture:
    layers: int
    kv_heads: int
    head_dim: int
    max_context: int


@dataclass(frozen=True)
class ModelProfile:
    name: str
    architecture: ModelArchitecture
    quality_score: float
    weight_mib: float
    runtime_mib: float
    cache_bytes_per_element: float
    prefill_ms_per_token: float
    decode_ms_per_token: float
    max_slots: int
    concurrency_penalty: float = 0.08

    def kv_mib(self, context_tokens: int, active_sequences: int) -> float:
        elements = (
            2
            * self.architecture.layers
            * self.architecture.kv_heads
            * self.architecture.head_dim
            * context_tokens
            * active_sequences
        )
        return elements * self.cache_bytes_per_element / MIB


@dataclass(frozen=True)
class RouteRequest:
    input_tokens: int
    output_tokens: int
    quality_floor: float
    latency_slo_ms: float
    active_sequences: int = 1
    available_vram_mib: float = inf


@dataclass(frozen=True)
class CostEstimate:
    prefill_ms: float
    decode_ms: float
    total_ms: float
    kv_mib: float
    total_memory_mib: float


@dataclass(frozen=True)
class CandidateDecision:
    model: str
    eligible: bool
    reasons: tuple[str, ...]
    score: float | None
    estimate: CostEstimate


@dataclass(frozen=True)
class RoutingDecision:
    selected_model: str
    candidates: tuple[CandidateDecision, ...]


@dataclass
class _Rates:
    prefill_ms_per_token: float
    decode_ms_per_token: float
    observations: int = 0


@dataclass
class EwmaCostModel:
    alpha: float = 0.25
    bucket_size: int = 256
    _rates: dict[tuple[str, int, int], _Rates] = field(default_factory=dict)

    def _bucket(self, input_tokens: int) -> int:
        return max(self.bucket_size, ((input_tokens + self.bucket_size - 1) // self.bucket_size) * self.bucket_size)

    def rates(self, profile: ModelProfile, request: RouteRequest) -> _Rates:
        key = (profile.name, self._bucket(request.input_tokens), request.active_sequences)
        return self._rates.get(
            key,
            _Rates(profile.prefill_ms_per_token, profile.decode_ms_per_token),
        )

    def estimate(self, profile: ModelProfile, request: RouteRequest) -> CostEstimate:
        rates = self.rates(profile, request)
        contention = 1.0 + profile.concurrency_penalty * max(request.active_sequences - 1, 0)
        prefill_ms = rates.prefill_ms_per_token * request.input_tokens * contention
        decode_ms = rates.decode_ms_per_token * request.output_tokens * contention
        kv_mib = profile.kv_mib(request.input_tokens + request.output_tokens, request.active_sequences)
        return CostEstimate(
            prefill_ms=prefill_ms,
            decode_ms=decode_ms,
            total_ms=prefill_ms + decode_ms,
            kv_mib=kv_mib,
            total_memory_mib=profile.weight_mib + profile.runtime_mib + kv_mib,
        )

    def observe(
        self,
        profile: ModelProfile,
        request: RouteRequest,
        *,
        actual_prefill_ms: float,
        actual_decode_ms: float,
    ) -> None:
        key = (profile.name, self._bucket(request.input_tokens), request.active_sequences)
        previous = self.rates(profile, request)
        observed_prefill = actual_prefill_ms / max(request.input_tokens, 1)
        observed_decode = actual_decode_ms / max(request.output_tokens, 1)
        self._rates[key] = _Rates(
            prefill_ms_per_token=(1 - self.alpha) * previous.prefill_ms_per_token
            + self.alpha * observed_prefill,
            decode_ms_per_token=(1 - self.alpha) * previous.decode_ms_per_token
            + self.alpha * observed_decode,
            observations=previous.observations + 1,
        )


class ModelRouter:
    def __init__(
        self,
        profiles: list[ModelProfile],
        cost_model: EwmaCostModel | None = None,
        *,
        quality_weight: float = 1.0,
        latency_weight: float = 1.0,
        memory_weight: float = 0.25,
    ) -> None:
        if not profiles:
            raise ValueError("at least one model profile is required")
        self.profiles = {profile.name: profile for profile in profiles}
        self.cost_model = cost_model or EwmaCostModel()
        self.quality_weight = quality_weight
        self.latency_weight = latency_weight
        self.memory_weight = memory_weight

    def route(self, request: RouteRequest, policy: str = "adaptive") -> RoutingDecision:
        if policy not in {"adaptive", "latency_only", "static"}:
            raise ValueError(f"unknown routing policy: {policy}")
        decisions: list[CandidateDecision] = []
        ordered_profiles = list(self.profiles.values())
        for profile in ordered_profiles:
            estimate = self.cost_model.estimate(profile, request)
            reasons: list[str] = []
            if request.input_tokens + request.output_tokens > profile.architecture.max_context:
                reasons.append("context_limit")
            if request.active_sequences > profile.max_slots:
                reasons.append("slot_capacity")
            if profile.quality_score < request.quality_floor:
                reasons.append("quality_floor")
            if estimate.total_memory_mib > request.available_vram_mib:
                reasons.append("vram_budget")
            if policy == "static" and profile is not ordered_profiles[0]:
                reasons.append("static_policy")

            score: float | None = None
            if not reasons:
                latency_ratio = estimate.total_ms / max(request.latency_slo_ms, 1e-9)
                memory_ratio = estimate.total_memory_mib / max(request.available_vram_mib, 1e-9)
                if policy == "latency_only":
                    score = -estimate.total_ms
                else:
                    score = (
                        self.quality_weight * profile.quality_score
                        - self.latency_weight * latency_ratio
                        - self.memory_weight * memory_ratio
                    )
            decisions.append(
                CandidateDecision(profile.name, not reasons, tuple(reasons), score, estimate)
            )

        eligible = [decision for decision in decisions if decision.eligible]
        if not eligible:
            details = "; ".join(
                f"{decision.model}={','.join(decision.reasons)}" for decision in decisions
            )
            raise RuntimeError(f"no eligible model: {details}")
        selected = max(eligible, key=lambda decision: float(decision.score))
        return RoutingDecision(selected.model, tuple(decisions))

    def observe(
        self,
        model: str,
        request: RouteRequest,
        *,
        actual_prefill_ms: float,
        actual_decode_ms: float,
    ) -> None:
        self.cost_model.observe(
            self.profiles[model],
            request,
            actual_prefill_ms=actual_prefill_ms,
            actual_decode_ms=actual_decode_ms,
        )
