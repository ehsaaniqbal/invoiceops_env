"""Fixture constants for InvoiceOps."""

from __future__ import annotations

from pathlib import Path

from invoiceops_env.models import TaskId


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
SCENARIOS_DIR = DATA_DIR / "scenarios"

ENV_DESCRIPTION = (
    "Document-centric AP invoice exception handling environment with deterministic "
    "grading for non-PO routing, duplicate-evidence review, partial-release "
    "judgment, chronology-aware exception handling, and safe payment-release "
    "decisions."
)

SCENARIOS_BY_TASK: dict[TaskId, tuple[str, ...]] = {
    TaskId.EASY: ("easy",),
    TaskId.MEDIUM: ("medium",),
    TaskId.MEDIUM_PLUS: ("medium_plus",),
    TaskId.HARD: ("hard",),
}

DEFAULT_SCENARIOS: dict[TaskId, str] = {
    task: scenario_ids[0] for task, scenario_ids in SCENARIOS_BY_TASK.items()
}

DUPLICATE_CHECK_REF_PREFIX = "duplicate_check:"
