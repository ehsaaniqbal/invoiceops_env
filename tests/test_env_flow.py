"""End-to-end environment flow tests for the 4-task InvoiceOps ladder."""

from invoiceops_env.models import (
    ActionType,
    Disposition,
    InvoiceOpsAction,
    NoteType,
    PaymentRecommendation,
    ReasonCode,
)
from invoiceops_env.server.invoiceops_env_environment import InvoiceOpsEnvironment
from invoiceops_env.server.scenario_loader import load_scenario


def _run_easy_perfect_case() -> float:
    env = InvoiceOpsEnvironment()
    env.reset(task_id="easy")

    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-invoice")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-approval"
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-NONPO-APPROVAL",
        )
    )
    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-policy")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.ADD_NOTE,
            note_type=NoteType.ISSUE_SUMMARY,
            reason_codes=[ReasonCode.NON_PO_APPROVAL_MISSING],
            evidence_refs=["art-approval", "art-policy"],
            text="Approval workflow is not initiated and the requester must start approval before payment can release.",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L1",
            disposition=Disposition.HOLD,
            reason_codes=[ReasonCode.NON_PO_APPROVAL_MISSING],
            evidence_refs=[
                "art-invoice",
                "art-approval",
                "art-policy",
                "EX-NONPO-APPROVAL",
            ],
            route_to="requester",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_HEADER_RESOLUTION,
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.NON_PO_APPROVAL_MISSING],
            evidence_refs=[
                "art-invoice",
                "art-approval",
                "art-policy",
                "EX-NONPO-APPROVAL",
            ],
            route_to="requester",
        )
    )
    result = env.step(InvoiceOpsAction(action_type=ActionType.SUBMIT_CASE))
    assert result.done is True
    return float(result.episode_score or 0.0)


def _run_medium_perfect_case() -> float:
    env = InvoiceOpsEnvironment()
    env.reset(task_id="medium")

    env.step(InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-po"))
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-receipts"
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-POSSIBLE-DUP",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.RUN_DUPLICATE_CHECK,
            match_strategy="normalized_invoice_no",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.ADD_NOTE,
            note_type=NoteType.REVIEW_SUMMARY,
            reason_codes=[
                ReasonCode.POSSIBLE_DUPLICATE_REVIEW,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=["duplicate_check:normalized_invoice_no", "CAND-NORM-01"],
            text="The normalized duplicate hit is a reversed prior record, so the invoice can release.",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L1",
            disposition=Disposition.APPROVE,
            reason_codes=[
                ReasonCode.MATCHED_TO_PO_AND_RECEIPT,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=[
                "art-po",
                "art-receipts",
                "EX-POSSIBLE-DUP",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L2",
            disposition=Disposition.APPROVE,
            reason_codes=[
                ReasonCode.MATCHED_TO_PO_AND_RECEIPT,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=[
                "art-po",
                "art-receipts",
                "EX-POSSIBLE-DUP",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_HEADER_RESOLUTION,
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[
                ReasonCode.POSSIBLE_DUPLICATE_REVIEW,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=[
                "art-po",
                "art-receipts",
                "EX-POSSIBLE-DUP",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )
    result = env.step(InvoiceOpsAction(action_type=ActionType.SUBMIT_CASE))
    assert result.done is True
    return float(result.episode_score or 0.0)


def _run_medium_plus_perfect_case() -> float:
    env = InvoiceOpsEnvironment()
    env.reset(task_id="medium_plus")

    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-invoice")
    )
    env.step(InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-po"))
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-receipts"
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-POSSIBLE-DUP",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.RUN_DUPLICATE_CHECK,
            match_strategy="normalized_invoice_no",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-RECEIPT-L2",
        )
    )
    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-policy")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.ADD_NOTE,
            note_type=NoteType.REVIEW_SUMMARY,
            reason_codes=[
                ReasonCode.POSSIBLE_DUPLICATE_REVIEW,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=["duplicate_check:normalized_invoice_no", "CAND-NORM-01"],
            text="The normalized duplicate hit is a reversed prior record, so duplicate review is cleared.",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.ADD_NOTE,
            note_type=NoteType.ISSUE_SUMMARY,
            reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
            evidence_refs=[
                "art-invoice",
                "art-receipts",
                "art-policy",
                "EX-RECEIPT-L2",
            ],
            text="L2 remains blocked because the unsupported amount exceeds the de minimis receipt threshold.",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L1",
            disposition=Disposition.APPROVE,
            reason_codes=[
                ReasonCode.MATCHED_TO_PO_AND_RECEIPT,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=[
                "art-po",
                "art-receipts",
                "EX-POSSIBLE-DUP",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L2",
            disposition=Disposition.HOLD,
            reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
            evidence_refs=[
                "art-invoice",
                "art-receipts",
                "art-policy",
                "EX-RECEIPT-L2",
            ],
            route_to="receiving",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_HEADER_RESOLUTION,
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[
                ReasonCode.POSSIBLE_DUPLICATE_REVIEW,
                ReasonCode.RECEIPT_NOT_CONFIRMED,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=[
                "art-policy",
                "art-receipts",
                "EX-RECEIPT-L2",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )
    result = env.step(InvoiceOpsAction(action_type=ActionType.SUBMIT_CASE))
    assert result.done is True
    return float(result.episode_score or 0.0)


def _run_hard_perfect_case() -> float:
    env = InvoiceOpsEnvironment()
    env.reset(task_id="hard")

    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-invoice")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-POSSIBLE-DUP",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.RUN_DUPLICATE_CHECK,
            match_strategy="normalized_invoice_no",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-receipts"
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-RECEIPT-L1",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-RECEIPT-L2",
        )
    )
    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-history")
    )
    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-vendor")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-TAX-001",
        )
    )
    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-policy")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.ADD_NOTE,
            note_type=NoteType.REVIEW_SUMMARY,
            reason_codes=[
                ReasonCode.POSSIBLE_DUPLICATE_REVIEW,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=["duplicate_check:normalized_invoice_no", "CAND-NORM-01"],
            text="The normalized duplicate hit is a reversed prior record, so the duplicate control is cleared.",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.ADD_NOTE,
            note_type=NoteType.ISSUE_SUMMARY,
            reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
            evidence_refs=["art-history", "EX-RECEIPT-L2"],
            text="L2 remains blocked because the latest receiving history shows an open damage hold.",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.ADD_NOTE,
            note_type=NoteType.ESCALATION_REQUEST,
            reason_codes=[ReasonCode.TAX_AMOUNT_MISMATCH],
            evidence_refs=["art-vendor", "art-policy", "EX-TAX-001"],
            text="The project is tax exempt, so payment must remain blocked pending Tax Ops review.",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L1",
            disposition=Disposition.APPROVE,
            reason_codes=[
                ReasonCode.PARTIAL_RECEIPT_PENDING,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=[
                "art-invoice",
                "art-receipts",
                "EX-RECEIPT-L1",
                "art-policy",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L2",
            disposition=Disposition.HOLD,
            reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
            evidence_refs=[
                "art-receipts",
                "art-history",
                "EX-RECEIPT-L2",
                "art-policy",
            ],
            route_to="receiving",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L3",
            disposition=Disposition.APPROVE,
            reason_codes=[
                ReasonCode.MATCHED_TO_PO_AND_RECEIPT,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=[
                "art-invoice",
                "art-receipts",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_HEADER_RESOLUTION,
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.TAX_AMOUNT_MISMATCH],
            evidence_refs=[
                "art-invoice",
                "art-vendor",
                "art-policy",
                "EX-TAX-001",
            ],
            route_to="tax",
        )
    )
    result = env.step(InvoiceOpsAction(action_type=ActionType.SUBMIT_CASE))
    assert result.done is True
    return float(result.episode_score or 0.0)


def test_perfect_cases_score_near_one() -> None:
    assert _run_easy_perfect_case() >= 0.99
    assert _run_medium_perfect_case() >= 0.99
    assert _run_medium_plus_perfect_case() >= 0.99
    assert _run_hard_perfect_case() >= 0.99


def test_tax_hold_can_coexist_with_approved_lines() -> None:
    env = InvoiceOpsEnvironment()
    env.reset(task_id="hard")

    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-invoice")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.RUN_DUPLICATE_CHECK,
            match_strategy="normalized_invoice_no",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-receipts"
        )
    )
    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-vendor")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-RECEIPT-L2",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-TAX-001",
        )
    )
    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-history")
    )
    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-policy")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L1",
            disposition=Disposition.APPROVE,
            reason_codes=[ReasonCode.SAFE_TO_PAY],
            evidence_refs=[
                "art-invoice",
                "art-receipts",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
                "EX-RECEIPT-L1",
            ],
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L2",
            disposition=Disposition.HOLD,
            reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
            evidence_refs=["art-history", "EX-RECEIPT-L2"],
            route_to="receiving",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L3",
            disposition=Disposition.APPROVE,
            reason_codes=[ReasonCode.SAFE_TO_PAY],
            evidence_refs=[
                "art-invoice",
                "art-receipts",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )

    result = env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_HEADER_RESOLUTION,
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.TAX_AMOUNT_MISMATCH],
            evidence_refs=[
                "art-invoice",
                "art-vendor",
                "art-policy",
                "EX-TAX-001",
            ],
            route_to="tax",
        )
    )

    assert result.done is False
    assert result.message == "Saved header recommendation."


def test_release_approved_lines_can_coexist_with_held_lines() -> None:
    env = InvoiceOpsEnvironment()
    env.reset(task_id="medium_plus")

    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-po")
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-receipts"
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.RUN_DUPLICATE_CHECK,
            match_strategy="normalized_invoice_no",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-RECEIPT-L2",
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L1",
            disposition=Disposition.APPROVE,
            reason_codes=[ReasonCode.SAFE_TO_PAY],
            evidence_refs=[
                "art-po",
                "art-receipts",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )
    env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_LINE_RESOLUTION,
            line_id="L2",
            disposition=Disposition.HOLD,
            reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
            evidence_refs=["art-receipts", "EX-RECEIPT-L2"],
            route_to="receiving",
        )
    )

    result = env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_HEADER_RESOLUTION,
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[ReasonCode.SAFE_TO_PAY],
            evidence_refs=[
                "art-receipts",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
        )
    )

    assert result.done is False
    assert result.message == "Saved header recommendation."


def test_release_approved_lines_without_approved_lines_is_invalid() -> None:
    env = InvoiceOpsEnvironment()
    env.reset(task_id="easy")

    env.step(
        InvoiceOpsAction(action_type=ActionType.OPEN_ARTIFACT, artifact_id="art-policy")
    )
    result = env.step(
        InvoiceOpsAction(
            action_type=ActionType.SET_HEADER_RESOLUTION,
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[ReasonCode.SAFE_TO_PAY],
            evidence_refs=["art-policy"],
        )
    )
    assert result.done is False

    submit_result = env.step(InvoiceOpsAction(action_type=ActionType.SUBMIT_CASE))
    assert submit_result.done is False
    assert "requires at least one approved line" in submit_result.message


def test_hard_budget_has_recovery_slack() -> None:
    scenario = load_scenario(task_id="hard")

    assert scenario.step_limit - scenario.hidden_truth.efficient_step_target >= 5


def test_medium_plus_budget_has_recovery_slack() -> None:
    scenario = load_scenario(task_id="medium_plus")

    assert scenario.step_limit - scenario.hidden_truth.efficient_step_target >= 4
