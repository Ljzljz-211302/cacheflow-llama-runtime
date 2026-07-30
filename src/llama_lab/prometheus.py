from __future__ import annotations

import re


_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


def parse_prometheus_text(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match:
            samples[match.group("name")] = float(match.group("value"))
    return samples


def require_engine_metrics(samples: dict[str, float]) -> None:
    required = {
        "llamacpp:prompt_tokens_cached_total",
        "llamacpp:slot_cache_selections_total",
        "llamacpp:slot_reused_tokens_total",
        "llamacpp:slot_evicted_tokens_total",
        "llamacpp:scheduler_iterations_total",
        "llamacpp:decode_tokens_scheduled_total",
        "llamacpp:prefill_tokens_scheduled_total",
        "llamacpp:prefill_chunks_scheduled_total",
        "llamacpp:kv_cache_tokens",
        "llamacpp:kv_cache_capacity_tokens",
        "llamacpp:memory_model_bytes",
        "llamacpp:memory_context_bytes",
        "llamacpp:memory_compute_bytes",
        "llamacpp:memory_total_bytes",
    }
    missing = sorted(required - samples.keys())
    if missing:
        raise ValueError(f"patched engine metrics missing: {', '.join(missing)}")
