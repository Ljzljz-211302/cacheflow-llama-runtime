"""Validation for the research baseline registry.

The registry deliberately separates locally reproducible quantitative baselines
from systems that are useful only as related work on the current hardware.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class BaselineManifestError(ValueError):
    """Raised when a baseline registry could permit an invalid comparison."""


_REQUIRED_BASELINE_FIELDS = {
    "id",
    "kind",
    "comparison_class",
    "runnable",
    "source",
    "license",
    "commands",
    "comparability_limits",
}


def load_baseline_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a research baseline registry."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BaselineManifestError(f"cannot load baseline manifest: {error}") from error

    if manifest.get("schema_version") != 1:
        raise BaselineManifestError("schema_version must be 1")
    revision = manifest.get("upstream_revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise BaselineManifestError("upstream_revision must be a full Git SHA")

    scope = manifest.get("scope")
    if not isinstance(scope, dict) or not {"gpu", "model", "kv_layout"} <= scope.keys():
        raise BaselineManifestError("scope must lock gpu, model, and kv_layout")

    baselines = manifest.get("baselines")
    if not isinstance(baselines, list) or not baselines:
        raise BaselineManifestError("baselines must be a non-empty list")

    seen: set[str] = set()
    for baseline in baselines:
        if not isinstance(baseline, dict):
            raise BaselineManifestError("each baseline must be an object")
        missing = _REQUIRED_BASELINE_FIELDS - baseline.keys()
        if missing:
            raise BaselineManifestError(
                f"baseline is missing fields: {', '.join(sorted(missing))}"
            )
        baseline_id = baseline["id"]
        if not isinstance(baseline_id, str) or not baseline_id:
            raise BaselineManifestError("baseline id must be a non-empty string")
        if baseline_id in seen:
            raise BaselineManifestError(f"duplicate baseline id: {baseline_id}")
        seen.add(baseline_id)

        commands = baseline["commands"]
        if not isinstance(commands, list):
            raise BaselineManifestError(f"{baseline_id}: commands must be a list")
        if baseline["comparison_class"] == "quantitative":
            if baseline["runnable"] is not True:
                raise BaselineManifestError(
                    f"{baseline_id}: quantitative baseline must be runnable"
                )
            if not commands:
                raise BaselineManifestError(
                    f"{baseline_id}: quantitative baseline must declare commands"
                )
        limits = baseline["comparability_limits"]
        if not isinstance(limits, list) or not limits:
            raise BaselineManifestError(
                f"{baseline_id}: comparability_limits must be non-empty"
            )

    return manifest

