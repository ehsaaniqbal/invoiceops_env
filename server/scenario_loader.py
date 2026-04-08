"""Scenario loading helpers for InvoiceOps."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field

from invoiceops_env.models import (
    ArtifactReference,
    ArtifactView,
    DuplicateCandidate,
    ExceptionDetail,
    ExceptionSummary,
    ExceptionType,
    QueueCard,
    RouteTarget,
    TaskId,
)
from invoiceops_env.models import Model as BaseModel
from invoiceops_env.server.fixtures import (
    DEFAULT_SCENARIOS,
    SCENARIOS_DIR,
)


class ResolutionExpectation(BaseModel):
    amount: float = Field(..., ge=0.0)
    score_map: dict[str, float] = Field(default_factory=dict)
    accepted_reason_sets: list[list[str]] = Field(default_factory=list)
    accepted_routes: list[str] = Field(default_factory=list)
    gating_refs: list[str] = Field(default_factory=list)
    safe_gating_refs: list[str] = Field(default_factory=list)
    decisive_refs: list[str] = Field(default_factory=list)
    unsafe_approve: bool = Field(default=False)


class HeaderExpectation(BaseModel):
    score_map: dict[str, float] = Field(default_factory=dict)
    accepted_reason_sets: list[list[str]] = Field(default_factory=list)
    accepted_routes: list[str] = Field(default_factory=list)
    gating_refs: list[str] = Field(default_factory=list)
    safe_gating_refs: list[str] = Field(default_factory=list)
    decisive_refs: list[str] = Field(default_factory=list)
    unsafe_recommendations: list[str] = Field(default_factory=list)
    overconservative_recommendations: list[str] = Field(default_factory=list)


class NoteExpectation(BaseModel):
    issue_id: str
    accepted_reason_sets: list[list[str]] = Field(default_factory=list)
    decisive_refs: list[str] = Field(default_factory=list)


class HiddenTruth(BaseModel):
    line_expectations: dict[str, ResolutionExpectation] = Field(default_factory=dict)
    header_expectation: HeaderExpectation
    note_expectations: list[NoteExpectation] = Field(default_factory=list)
    efficient_step_target: int = Field(default=0, ge=0)


class ScenarioFixture(BaseModel):
    scenario_id: str
    task_id: TaskId
    case_id: str
    title: str
    description: str
    step_limit: int = Field(..., ge=1)
    queue_card: QueueCard
    artifacts: list[ArtifactView] = Field(default_factory=list)
    exceptions: list[ExceptionDetail] = Field(default_factory=list)
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)
    hidden_truth: HiddenTruth


def _scenario_path_for_id(scenario_id: str) -> Path:
    return SCENARIOS_DIR / f"{scenario_id}.json"


def load_scenario(
    task_id: TaskId | str | None = None,
    scenario_id: str | None = None,
) -> ScenarioFixture:
    if scenario_id is None:
        task = TaskId(task_id or TaskId.EASY)
        scenario_id = DEFAULT_SCENARIOS[task]

    scenario_path = _scenario_path_for_id(scenario_id)
    if not scenario_path.exists():
        raise ValueError(f"Unknown scenario_id: {scenario_id}")

    with scenario_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    scenario = ScenarioFixture.model_validate(payload)
    if task_id is not None and scenario.task_id is not TaskId(task_id):
        raise ValueError(
            f"Scenario '{scenario_id}' belongs to task '{scenario.task_id.value}', "
            f"not '{TaskId(task_id).value}'"
        )
    return scenario


QUEUE_SAFE_EXCEPTION_HEADLINES: dict[ExceptionType, str] = {
    ExceptionType.RECEIPT_QUANTITY_VARIANCE: "Receipt variance requires review",
    ExceptionType.NON_PO_MISSING_APPROVAL: "Non-PO approval exception requires review",
    ExceptionType.POSSIBLE_DUPLICATE: "Potential duplicate invoice requires review",
    ExceptionType.PRICE_VARIANCE: "Price variance requires review",
    ExceptionType.CUMULATIVE_BILLING_VARIANCE: "Cumulative billing exception requires review",
    ExceptionType.TAX_VARIANCE: "Tax exception requires review",
    ExceptionType.PAYMENT_TERMS_MISMATCH: "Payment terms exception requires review",
}

QUEUE_SAFE_EXCEPTION_HINTS: dict[ExceptionType, str] = {
    ExceptionType.RECEIPT_QUANTITY_VARIANCE: (
        "Inspect this exception for receipt support and quantity details."
    ),
    ExceptionType.NON_PO_MISSING_APPROVAL: (
        "Inspect this exception for workflow status and approval details."
    ),
    ExceptionType.POSSIBLE_DUPLICATE: (
        "Inspect this exception for duplicate-match details before deciding."
    ),
    ExceptionType.PRICE_VARIANCE: (
        "Inspect this exception for invoice-vs-PO price details."
    ),
    ExceptionType.CUMULATIVE_BILLING_VARIANCE: (
        "Inspect this exception for history-aware billing facts."
    ),
    ExceptionType.TAX_VARIANCE: "Inspect this exception for tax calculation details.",
    ExceptionType.PAYMENT_TERMS_MISMATCH: (
        "Inspect this exception for payment-terms comparison details."
    ),
}


def artifact_lookup(scenario: ScenarioFixture) -> dict[str, ArtifactView]:
    return {artifact.artifact_id: artifact for artifact in scenario.artifacts}


def artifact_references(scenario: ScenarioFixture) -> list[ArtifactReference]:
    return [
        ArtifactReference(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            title=artifact.title,
        )
        for artifact in scenario.artifacts
    ]


def exception_lookup(scenario: ScenarioFixture) -> dict[str, ExceptionDetail]:
    return {exception.exception_id: exception for exception in scenario.exceptions}


def exception_summaries(scenario: ScenarioFixture) -> list[ExceptionSummary]:
    return [
        ExceptionSummary(
            exception_id=exception.exception_id,
            exception_type=exception.exception_type,
            severity=exception.severity,
            headline=QUEUE_SAFE_EXCEPTION_HEADLINES.get(
                exception.exception_type,
                "Exception requires review",
            ),
            impacted_line_ids=exception.impacted_line_ids,
            short_description=QUEUE_SAFE_EXCEPTION_HINTS.get(
                exception.exception_type,
                "Inspect this exception for detailed facts before deciding.",
            ),
        )
        for exception in scenario.exceptions
    ]


def line_ids_for_scenario(scenario: ScenarioFixture) -> set[str]:
    line_ids: set[str] = set(scenario.hidden_truth.line_expectations.keys())
    for artifact in scenario.artifacts:
        for line_item in artifact.line_items:
            line_ids.add(line_item.line_id)
    return line_ids


def route_target_values() -> set[str]:
    return {route.value for route in RouteTarget}
