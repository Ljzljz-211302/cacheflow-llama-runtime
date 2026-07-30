from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MIB = 1024 * 1024


@dataclass(frozen=True)
class ModelArchitecture:
    layers: int
    kv_heads: int
    head_dim: int
    max_context: int


@dataclass(frozen=True)
class MemoryEstimate:
    context: int
    slots: int
    weights_mib: float
    kv_cache_mib: float
    runtime_mib: float
    safety_mib: float
    total_mib: float
    fits: bool


def estimate_kv_cache_mib(
    architecture: ModelArchitecture,
    context: int,
    slots: int = 1,
    bytes_per_element: int = 2,
) -> float:
    """Estimate K+V cache storage for a decoder-only transformer."""
    if context <= 0 or slots <= 0 or bytes_per_element <= 0:
        raise ValueError("context, slots and bytes_per_element must be positive")
    if context > architecture.max_context:
        raise ValueError("context exceeds the model maximum")
    byte_count = (
        2
        * architecture.layers
        * architecture.kv_heads
        * architecture.head_dim
        * context
        * slots
        * bytes_per_element
    )
    return byte_count / MIB


def estimate_memory(
    model_path: Path,
    architecture: ModelArchitecture,
    context: int,
    available_mib: float,
    slots: int = 1,
    bytes_per_element: int = 2,
    runtime_mib: float = 384.0,
    safety_ratio: float = 0.15,
) -> MemoryEstimate:
    if available_mib <= 0:
        raise ValueError("available_mib must be positive")
    if not 0 <= safety_ratio < 1:
        raise ValueError("safety_ratio must be in [0, 1)")
    weights_mib = model_path.stat().st_size / MIB
    kv_cache_mib = estimate_kv_cache_mib(
        architecture, context, slots, bytes_per_element
    )
    safety_mib = available_mib * safety_ratio
    total_mib = weights_mib + kv_cache_mib + runtime_mib + safety_mib
    return MemoryEstimate(
        context=context,
        slots=slots,
        weights_mib=weights_mib,
        kv_cache_mib=kv_cache_mib,
        runtime_mib=runtime_mib,
        safety_mib=safety_mib,
        total_mib=total_mib,
        fits=total_mib <= available_mib,
    )


def recommend_largest_context(
    model_path: Path,
    architecture: ModelArchitecture,
    available_mib: float,
    slots: int = 1,
    candidates: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384, 32768),
) -> MemoryEstimate | None:
    estimates = [
        estimate_memory(model_path, architecture, ctx, available_mib, slots)
        for ctx in candidates
        if ctx <= architecture.max_context
    ]
    fitting = [estimate for estimate in estimates if estimate.fits]
    return max(fitting, key=lambda estimate: estimate.context, default=None)
