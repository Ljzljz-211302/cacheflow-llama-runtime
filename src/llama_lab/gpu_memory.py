from __future__ import annotations

import subprocess
import threading


def query_gpu_used_mib() -> float | None:
    """Return total memory used on CUDA device 0, or None without nvidia-smi."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(completed.stdout.strip().splitlines()[0])
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        ValueError,
        IndexError,
    ):
        return None


class GpuMemorySampler:
    """Sample system-wide device memory while a benchmark case is running."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.baseline_mib: float | None = None
        self.peak_mib: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        used = query_gpu_used_mib()
        if used is not None:
            self.peak_mib = used if self.peak_mib is None else max(self.peak_mib, used)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample_once()

    def __enter__(self) -> GpuMemorySampler:
        self.baseline_mib = query_gpu_used_mib()
        self.peak_mib = self.baseline_mib
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._sample_once()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))

    @property
    def increment_mib(self) -> float | None:
        if self.baseline_mib is None or self.peak_mib is None:
            return None
        return max(0.0, self.peak_mib - self.baseline_mib)
