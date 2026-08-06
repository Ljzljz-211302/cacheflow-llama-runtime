"""Validation for falsifiable research claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llama_lab.research_baselines import load_baseline_manifest


class CharterError(ValueError):
    """Raised when a research claim cannot be audited or falsified."""


_LIST_FIELDS = {
    "independent_variables",
    "dependent_metrics",
    "confounders",
    "baselines",
    "falsification",
    "evidence_sources",
    "observed_results",
    "scope_limits",
}
_TEXT_FIELDS = {"id", "question", "hypothesis", "mechanism", "negative_result_policy"}
_STATUSES = {"prospective", "limited-evidence", "existing-evidence"}


def load_research_charter(
    path: Path, baseline_manifest_path: Path | None = None
) -> dict[str, Any]:
    """Load a claim registry and reject unfalsifiable or ambiguous claims."""

    try:
        charter = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CharterError(f"cannot load research charter: {error}") from error

    if charter.get("schema_version") != 1:
        raise CharterError("schema_version must be 1")
    if not isinstance(charter.get("charter_version"), str) or not charter["charter_version"]:
        raise CharterError("charter_version must be a non-empty string")
    claims = charter.get("claims")
    if not isinstance(claims, list) or not claims:
        raise CharterError("claims must be a non-empty list")

    known_baselines: set[str] | None = None
    if baseline_manifest_path is not None:
        baseline_manifest = load_baseline_manifest(baseline_manifest_path)
        known_baselines = {item["id"] for item in baseline_manifest["baselines"]}

    seen: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise CharterError("each claim must be an object")
        for field in _TEXT_FIELDS:
            if not isinstance(claim.get(field), str) or not claim[field].strip():
                raise CharterError(f"claim {claim.get('id', '<unknown>')}: {field} is required")
        claim_id = claim["id"]
        if claim_id in seen:
            raise CharterError(f"duplicate claim id: {claim_id}")
        seen.add(claim_id)
        if claim.get("status") not in _STATUSES:
            raise CharterError(f"{claim_id}: invalid evidence status")
        for field in _LIST_FIELDS:
            value = claim.get(field)
            if not isinstance(value, list):
                raise CharterError(f"{claim_id}: {field} must be a list")
            if field != "observed_results" and not value:
                raise CharterError(f"{claim_id}: {field} must not be empty")
        if known_baselines is not None:
            unknown = set(claim["baselines"]) - known_baselines
            if unknown:
                raise CharterError(
                    f"{claim_id}: unknown baselines: {', '.join(sorted(unknown))}"
                )
        if claim["status"] == "prospective" and claim["observed_results"]:
            raise CharterError(
                f"{claim_id}: prospective claim cannot contain observed results"
            )
        if (
            claim["status"] in {"limited-evidence", "existing-evidence"}
            and not claim["observed_results"]
        ):
            raise CharterError(
                f"{claim_id}: evidence-bearing claim must cite observed results"
            )

    return charter
