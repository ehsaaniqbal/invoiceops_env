"""Typed models for the InvoiceOps environment."""

from __future__ import annotations

from enum import Enum

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class TaskId(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    MEDIUM_PLUS = "medium_plus"
    HARD = "hard"


class ActionType(str, Enum):
    OPEN_ARTIFACT = "open_artifact"
    INSPECT_EXCEPTION = "inspect_exception"
    RUN_DUPLICATE_CHECK = "run_duplicate_check"
    ADD_NOTE = "add_note"
    SET_LINE_RESOLUTION = "set_line_resolution"
    SET_HEADER_RESOLUTION = "set_header_resolution"
    SUBMIT_CASE = "submit_case"


class ArtifactType(str, Enum):
    INVOICE_PACKET = "invoice_packet"
    PURCHASE_ORDER = "purchase_order"
    RECEIPT_LOG = "receipt_log"
    VENDOR_MASTER = "vendor_master"
    POLICY_CARD = "policy_card"
    APPROVAL_ARTIFACT = "approval_artifact"
    INVOICE_HISTORY = "invoice_history"


class ExceptionType(str, Enum):
    RECEIPT_QUANTITY_VARIANCE = "receipt_quantity_variance"
    NON_PO_MISSING_APPROVAL = "non_po_missing_approval"
    POSSIBLE_DUPLICATE = "possible_duplicate"
    PRICE_VARIANCE = "price_variance"
    CUMULATIVE_BILLING_VARIANCE = "cumulative_billing_variance"
    TAX_VARIANCE = "tax_variance"
    PAYMENT_TERMS_MISMATCH = "payment_terms_mismatch"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DuplicateMatchStrategy(str, Enum):
    EXACT_INVOICE_NUMBER = "exact_invoice_no"
    NORMALIZED_INVOICE_NUMBER = "normalized_invoice_no"
    VENDOR_AMOUNT_DATE = "vendor_amount_date"


class NoteType(str, Enum):
    ISSUE_SUMMARY = "issue_summary"
    ESCALATION_REQUEST = "escalation_request"
    REVIEW_SUMMARY = "review_summary"


class Disposition(str, Enum):
    APPROVE = "approve"
    HOLD = "hold"
    REJECT = "reject"
    ESCALATE = "escalate"


class PaymentRecommendation(str, Enum):
    RELEASE_APPROVED_LINES = "release_approved_lines"
    HOLD_FULL_INVOICE = "hold_full_invoice"
    REJECT_FULL_INVOICE = "reject_full_invoice"
    ESCALATE_CASE = "escalate_case"


class DecisionBand(str, Enum):
    BEST = "best"
    SAFE_SUBOPTIMAL = "safe_suboptimal"
    WRONG = "wrong"
    UNSAFE = "unsafe"


class RouteTarget(str, Enum):
    RECEIVING = "receiving"
    REQUESTER = "requester"
    PROCUREMENT = "procurement"
    TAX = "tax"
    AP_MANAGER = "ap_manager"


class RiskFlag(str, Enum):
    PO_INVOICE = "po_invoice"
    RECEIPT_VARIANCE = "receipt_variance"
    PARTIAL_RECEIPT = "partial_receipt"
    PRICE_VARIANCE = "price_variance"
    NON_PO_INVOICE = "non_po_invoice"
    MISSING_APPROVAL = "missing_approval"
    POSSIBLE_DUPLICATE = "possible_duplicate"
    CUMULATIVE_BILLING_RISK = "cumulative_billing_risk"
    TAX_VARIANCE = "tax_variance"
    TERMS_MISMATCH = "terms_mismatch"


class ReasonCode(str, Enum):
    MATCHED_TO_PO_AND_RECEIPT = "matched_to_po_and_receipt"
    RECEIPT_NOT_CONFIRMED = "receipt_not_confirmed"
    PARTIAL_RECEIPT_PENDING = "partial_receipt_pending"
    PRICE_EXCEEDS_PO_RATE = "price_exceeds_po_rate"
    NON_PO_APPROVAL_MISSING = "non_po_approval_missing"
    POSSIBLE_DUPLICATE_REVIEW = "possible_duplicate_review"
    CUMULATIVE_BILLING_EXCEEDS_PO = "cumulative_billing_exceeds_po"
    TAX_AMOUNT_MISMATCH = "tax_amount_mismatch"
    PAYMENT_TERMS_MISMATCH = "payment_terms_mismatch"
    SAFE_TO_PAY = "safe_to_pay"
    ESCALATE_FOR_MANUAL_REVIEW = "escalate_for_manual_review"


class ArtifactField(Model):
    label: str = Field(..., description="Field label shown to the reviewer")
    value: str = Field(..., description="Rendered field value")


class ArtifactLineItem(Model):
    line_id: str = Field(..., description="Stable line identifier")
    description: str = Field(..., description="Line description")
    quantity: float | None = Field(default=None, description="Line quantity")
    unit_price: float | None = Field(default=None, description="Unit price")
    amount: float | None = Field(
        default=None,
        description="Extended amount for the line when the artifact exposes it",
    )
    status: str = Field(default="", description="Operational status")
    notes: str = Field(default="", description="Short line note")


class ArtifactEvent(Model):
    event_id: str = Field(..., description="Stable event identifier")
    event_type: str = Field(..., description="Event type label")
    event_date: str = Field(..., description="Event date in ISO format")
    description: str = Field(..., description="Human readable event description")
    quantity: float | None = Field(default=None, description="Event quantity")
    amount: float | None = Field(default=None, description="Event amount")
    status: str = Field(default="", description="Event status")


class ArtifactReference(Model):
    artifact_id: str = Field(..., description="Artifact identifier")
    artifact_type: ArtifactType = Field(..., description="Artifact type")
    title: str = Field(..., description="Artifact title shown in the UI")


class ArtifactView(ArtifactReference):
    summary: str = Field(default="", description="Short artifact summary")
    fields: list[ArtifactField] = Field(
        default_factory=list,
        description="Structured key-value pairs exposed by the artifact",
    )
    line_items: list[ArtifactLineItem] = Field(
        default_factory=list,
        description="Line items exposed by the artifact",
    )
    events: list[ArtifactEvent] = Field(
        default_factory=list,
        description="Timeline or ledger events exposed by the artifact",
    )
    related_refs: list[str] = Field(
        default_factory=list,
        description="Related artifact or issue identifiers",
    )


class QueueCard(Model):
    case_id: str = Field(..., description="Stable case identifier")
    vendor_name: str = Field(..., description="Vendor display name")
    vendor_id: str = Field(..., description="Vendor identifier")
    invoice_number: str = Field(..., description="Invoice number")
    invoice_date: str = Field(..., description="Invoice date in ISO format")
    invoice_total: float = Field(..., description="Gross invoice total")
    currency: str = Field(..., description="Invoice currency")
    po_number: str | None = Field(default=None, description="PO number when present")
    risk_flags: list[RiskFlag] = Field(
        default_factory=list,
        description="Compact risk hints visible from the queue",
    )
    summary: str = Field(default="", description="Short queue summary")


class ExceptionSummary(Model):
    exception_id: str = Field(..., description="Stable exception identifier")
    exception_type: ExceptionType = Field(..., description="Exception category")
    severity: Severity = Field(..., description="Exception severity")
    headline: str = Field(..., description="Queue-visible exception stub headline")
    impacted_line_ids: list[str] = Field(
        default_factory=list,
        description="Invoice lines directly impacted by the exception",
    )
    short_description: str = Field(
        default="",
        description="Queue-safe hint shown before inspection",
    )


class ExceptionDetail(ExceptionSummary):
    fields: list[ArtifactField] = Field(
        default_factory=list,
        description="Structured exception facts shown after inspection",
    )
    reviewer_guidance: str = Field(
        default="",
        description="Short workflow guidance exposed after inspection",
    )


class DuplicateCandidate(Model):
    candidate_id: str = Field(..., description="Ledger invoice identifier")
    vendor_name: str = Field(..., description="Vendor display name")
    invoice_number: str = Field(..., description="Prior or pending invoice number")
    invoice_date: str = Field(..., description="Candidate invoice date")
    gross_amount: float = Field(..., description="Candidate gross amount")
    status: str = Field(..., description="Current ledger or workflow status")
    match_basis: str = Field(..., description="Why the invoice was matched")
    overlap_summary: str = Field(..., description="Human readable overlap summary")
    supported_match_strategies: list[DuplicateMatchStrategy] = Field(
        default_factory=list,
        description="Match strategies that surface this candidate",
    )


class CaseNote(Model):
    note_id: str = Field(..., description="Stable note identifier")
    note_type: NoteType = Field(..., description="Workflow note category")
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        description="Structured reason codes captured in the note",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Artifact or exception references cited in the note",
    )
    text: str = Field(
        ...,
        description="Free-form note text retained for auditability, not prose-quality scoring",
    )
    saved_at_step: int = Field(..., ge=0, description="Step where the note was saved")


class LineResolution(Model):
    resolution_id: str = Field(..., description="Stable resolution identifier")
    line_id: str = Field(..., description="Invoice line identifier")
    disposition: Disposition = Field(..., description="Line disposition")
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        description="Structured reason codes supporting the line disposition",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Artifact or exception references cited by the reviewer",
    )
    route_to: RouteTarget | None = Field(
        default=None,
        description="Next owner or follow-up queue for the line when another team must act",
    )
    saved_at_step: int = Field(
        ...,
        ge=0,
        description="Step where the line disposition was saved",
    )


class HeaderResolution(Model):
    resolution_id: str = Field(..., description="Stable header resolution identifier")
    payment_recommendation: PaymentRecommendation = Field(
        ...,
        description=(
            "Header-level payment recommendation governing whether any payment can "
            "be released now, including case-level blockers that may override "
            "otherwise approved lines"
        ),
    )
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        description="Structured reason codes for the header recommendation",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Artifact or exception references cited by the reviewer",
    )
    route_to: RouteTarget | None = Field(
        default=None,
        description="Next owner or follow-up queue for the case when another team must act",
    )
    saved_at_step: int = Field(
        ...,
        ge=0,
        description="Step where the header recommendation was saved",
    )


class LineScoreReport(Model):
    line_id: str
    line_score: float
    disposition_score: float
    reason_score: float
    route_score: float
    evidence_score: float
    accepted_dispositions: list[Disposition] = Field(default_factory=list)


class HeaderScoreReport(Model):
    header_score: float
    recommendation_score: float
    reason_score: float
    route_score: float
    evidence_score: float
    accepted_recommendations: list[PaymentRecommendation] = Field(default_factory=list)


class IssueNoteReport(Model):
    issue_id: str
    note_score: float
    reason_score: float
    evidence_score: float


class SubmissionReport(Model):
    decision_band: DecisionBand
    total_score: float = Field(..., ge=0.0, le=1.0)
    core_decision_score: float = Field(..., ge=0.0, le=1.0)
    reason_quality_score: float = Field(..., ge=0.0, le=1.0)
    auxiliary_score: float = Field(..., ge=0.0, le=1.0)
    resolution_score: float = Field(..., ge=0.0, le=1.0)
    evidence_score: float = Field(..., ge=0.0, le=1.0)
    documentation_score: float = Field(..., ge=0.0, le=1.0)
    efficiency_score: float = Field(..., ge=0.0, le=1.0)
    safety_cap_applied: float | None = Field(
        default=None,
        description="Cap value applied because the action set was unsafe",
    )
    unsafe_findings: list[str] = Field(
        default_factory=list,
        description="Unsafe findings surfaced by the grader",
    )
    line_reports: list[LineScoreReport] = Field(default_factory=list)
    header_report: HeaderScoreReport | None = None
    note_reports: list[IssueNoteReport] = Field(default_factory=list)


class Progress(Model):
    steps_used: int = Field(..., ge=0, description="Steps used in the episode")
    steps_remaining: int = Field(..., ge=0, description="Steps remaining")
    opened_artifacts: int = Field(..., ge=0, description="Unique artifacts opened")
    inspected_exceptions: int = Field(
        ...,
        ge=0,
        description="Unique exceptions inspected",
    )
    notes_count: int = Field(..., ge=0, description="Saved notes")
    line_resolutions: int = Field(..., ge=0, description="Saved line resolutions")
    duplicate_checks_run: int = Field(
        ...,
        ge=0,
        description="Duplicate check actions executed",
    )
    invalid_actions: int = Field(..., ge=0, description="Invalid actions taken")
    redundant_actions: int = Field(..., ge=0, description="Redundant actions taken")
    submitted: bool = Field(default=False, description="Whether the case is submitted")


class InvoiceOpsObservation(Observation):
    message: str = Field(default="", description="Short environment message")
    task_id: TaskId | None = Field(default=None, description="Task bucket for the case")
    scenario_id: str | None = Field(default=None, description="Scenario identifier")
    title: str = Field(default="", description="Case title")
    description: str = Field(default="", description="Case description")
    queue_card: QueueCard | None = Field(
        default=None,
        description="Queue-level summary of the current invoice case",
    )
    available_artifacts: list[ArtifactReference] = Field(
        default_factory=list,
        description="Artifacts currently available to the reviewer",
    )
    opened_artifact: ArtifactView | None = Field(
        default=None,
        description="Most recently opened artifact",
    )
    visible_exceptions: list[ExceptionSummary] = Field(
        default_factory=list,
        description="Queue-visible exception stubs visible before detailed inspection",
    )
    inspected_exception: ExceptionDetail | None = Field(
        default=None,
        description="Most recently inspected full exception detail",
    )
    duplicate_candidates: list[DuplicateCandidate] = Field(
        default_factory=list,
        description="Candidates surfaced by duplicate search",
    )
    draft_notes: list[CaseNote] = Field(
        default_factory=list,
        description="Saved case notes",
    )
    draft_line_resolutions: list[LineResolution] = Field(
        default_factory=list,
        description="Draft line resolutions saved so far",
    )
    draft_header_resolution: HeaderResolution | None = Field(
        default=None,
        description="Draft header recommendation if saved",
    )
    submission_report: SubmissionReport | None = Field(
        default=None,
        description="Deterministic grading report after submission",
    )
    progress: Progress = Field(
        default_factory=lambda: Progress(
            steps_used=0,
            steps_remaining=0,
            opened_artifacts=0,
            inspected_exceptions=0,
            notes_count=0,
            line_resolutions=0,
            duplicate_checks_run=0,
            invalid_actions=0,
            redundant_actions=0,
            submitted=False,
        ),
        description="Episode progress counters",
    )
    known_refs: list[str] = Field(
        default_factory=list,
        description="Evidence refs that can be cited safely in notes or resolutions",
    )
    episode_score: float | None = Field(
        default=None,
        description="Final episode score when the case is done",
    )


class InvoiceOpsState(State):
    task_id: TaskId | None = Field(default=None, description="Task bucket")
    scenario_id: str | None = Field(default=None, description="Scenario identifier")
    case_id: str | None = Field(default=None, description="Case identifier")
    current_artifact_id: str | None = Field(
        default=None,
        description="Most recently opened artifact",
    )
    submitted: bool = Field(default=False, description="Whether the case is submitted")
    step_limit: int = Field(default=0, ge=0, description="Episode step budget")
    duplicate_checks_run: int = Field(
        default=0,
        ge=0,
        description="Number of duplicate checks executed",
    )
    invalid_actions: int = Field(
        default=0,
        ge=0,
        description="Number of invalid actions taken",
    )
    redundant_actions: int = Field(
        default=0,
        ge=0,
        description="Number of redundant actions taken",
    )


class InvoiceOpsAction(Action):
    action_type: ActionType = Field(..., description="Action to execute")
    artifact_id: str | None = Field(default=None, description="Artifact to open")
    exception_id: str | None = Field(default=None, description="Exception to inspect")
    match_strategy: DuplicateMatchStrategy | None = Field(
        default=None,
        description="Duplicate search strategy to run",
    )
    note_type: NoteType | None = Field(default=None, description="Case note type")
    reason_codes: list[ReasonCode] = Field(
        default_factory=list,
        description="Structured reason codes carried by the action",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="Artifact or exception refs supporting the action",
    )
    text: str | None = Field(default=None, description="Free-form note text")
    line_id: str | None = Field(default=None, description="Invoice line identifier")
    disposition: Disposition | None = Field(default=None, description="Line outcome")
    payment_recommendation: PaymentRecommendation | None = Field(
        default=None,
        description="Header-level payment recommendation",
    )
    route_to: RouteTarget | None = Field(
        default=None,
        description="Next owner or follow-up queue for the action, when applicable",
    )
    note_ids: list[str] = Field(
        default_factory=list,
        description="Optional note identifiers to submit",
    )
    line_resolution_ids: list[str] = Field(
        default_factory=list,
        description="Optional line resolution identifiers to submit",
    )
    header_resolution_id: str | None = Field(
        default=None,
        description="Optional header resolution identifier to submit",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "InvoiceOpsAction":
        action_type = self.action_type

        if action_type is ActionType.OPEN_ARTIFACT:
            if not self.artifact_id:
                raise ValueError("artifact_id is required for open_artifact")
            return self

        if action_type is ActionType.INSPECT_EXCEPTION:
            if not self.exception_id:
                raise ValueError("exception_id is required for inspect_exception")
            return self

        if action_type is ActionType.RUN_DUPLICATE_CHECK:
            if self.match_strategy is None:
                raise ValueError("match_strategy is required for run_duplicate_check")
            return self

        if action_type is ActionType.ADD_NOTE:
            if self.note_type is None:
                raise ValueError("note_type is required for add_note")
            if not self.reason_codes:
                raise ValueError("reason_codes are required for add_note")
            if not self.evidence_refs:
                raise ValueError("evidence_refs are required for add_note")
            if not self.text or not self.text.strip():
                raise ValueError("text is required for add_note")
            return self

        if action_type is ActionType.SET_LINE_RESOLUTION:
            if not self.line_id:
                raise ValueError("line_id is required for set_line_resolution")
            if self.disposition is None:
                raise ValueError("disposition is required for set_line_resolution")
            if not self.reason_codes:
                raise ValueError("reason_codes are required for set_line_resolution")
            if not self.evidence_refs:
                raise ValueError("evidence_refs are required for set_line_resolution")
            if self.disposition is Disposition.ESCALATE and self.route_to is None:
                raise ValueError("route_to is required when escalating a line")
            return self

        if action_type is ActionType.SET_HEADER_RESOLUTION:
            if self.payment_recommendation is None:
                raise ValueError(
                    "payment_recommendation is required for set_header_resolution"
                )
            if not self.reason_codes:
                raise ValueError("reason_codes are required for set_header_resolution")
            if not self.evidence_refs:
                raise ValueError("evidence_refs are required for set_header_resolution")
            if (
                self.payment_recommendation is PaymentRecommendation.ESCALATE_CASE
                and self.route_to is None
            ):
                raise ValueError("route_to is required when escalating the case")
            return self

        if action_type is ActionType.SUBMIT_CASE:
            return self

        raise ValueError(f"Unsupported action_type: {action_type}")
