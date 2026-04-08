"""Grader discrimination tests for the 4-task InvoiceOps benchmark."""

from invoiceops_env.models import (
    DecisionBand,
    Disposition,
    HeaderResolution,
    LineResolution,
    PaymentRecommendation,
    ReasonCode,
    RouteTarget,
)
from invoiceops_env.server.grader import ReviewTrace, grade_case
from invoiceops_env.server.scenario_loader import load_scenario


def test_easy_single_line_header_fallback_rewards_correct_route() -> None:
    scenario = load_scenario(task_id="easy")
    trace = ReviewTrace(
        ref_steps={
            "art-invoice": 1,
            "art-approval": 2,
            "EX-NONPO-APPROVAL": 3,
            "art-policy": 4,
        },
        steps_used=6,
    )

    correct_header_only = grade_case(
        scenario,
        line_resolutions={},
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.NON_PO_APPROVAL_MISSING],
            evidence_refs=[
                "art-invoice",
                "art-approval",
                "art-policy",
                "EX-NONPO-APPROVAL",
            ],
            route_to=RouteTarget.REQUESTER,
            saved_at_step=5,
        ),
        notes={},
        trace=trace,
    )

    wrong_route_header_only = grade_case(
        scenario,
        line_resolutions={},
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.NON_PO_APPROVAL_MISSING],
            evidence_refs=[
                "art-invoice",
                "art-approval",
                "art-policy",
                "EX-NONPO-APPROVAL",
            ],
            route_to=RouteTarget.AP_MANAGER,
            saved_at_step=5,
        ),
        notes={},
        trace=trace,
    )

    assert correct_header_only.decision_band is DecisionBand.BEST
    assert wrong_route_header_only.decision_band is DecisionBand.WRONG
    assert correct_header_only.total_score > 0.95
    assert wrong_route_header_only.total_score < 0.30


def test_medium_duplicate_evidence_creates_best_safe_and_wrong_bands() -> None:
    scenario = load_scenario(task_id="medium")

    best_trace = ReviewTrace(
        ref_steps={
            "art-po": 1,
            "art-receipts": 2,
            "EX-POSSIBLE-DUP": 3,
            "duplicate_check:normalized_invoice_no": 4,
            "CAND-NORM-01": 4,
        },
        steps_used=8,
    )
    best_report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
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
                route_to=None,
                saved_at_step=5,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
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
                route_to=None,
                saved_at_step=6,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
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
            route_to=None,
            saved_at_step=7,
        ),
        notes={},
        trace=best_trace,
    )

    safe_hold_report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
                line_id="L1",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.POSSIBLE_DUPLICATE_REVIEW],
                evidence_refs=[
                    "EX-POSSIBLE-DUP",
                    "duplicate_check:normalized_invoice_no",
                    "CAND-NORM-01",
                ],
                route_to=None,
                saved_at_step=5,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.POSSIBLE_DUPLICATE_REVIEW],
                evidence_refs=[
                    "EX-POSSIBLE-DUP",
                    "duplicate_check:normalized_invoice_no",
                    "CAND-NORM-01",
                ],
                route_to=None,
                saved_at_step=6,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.POSSIBLE_DUPLICATE_REVIEW],
            evidence_refs=[
                "EX-POSSIBLE-DUP",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
            route_to=None,
            saved_at_step=7,
        ),
        notes={},
        trace=best_trace,
    )

    wrong_heuristic_hold = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
                line_id="L1",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.POSSIBLE_DUPLICATE_REVIEW],
                evidence_refs=[
                    "EX-POSSIBLE-DUP",
                    "duplicate_check:vendor_amount_date",
                    "CAND-AMT-02",
                ],
                route_to=None,
                saved_at_step=5,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.POSSIBLE_DUPLICATE_REVIEW],
                evidence_refs=[
                    "EX-POSSIBLE-DUP",
                    "duplicate_check:vendor_amount_date",
                    "CAND-AMT-02",
                ],
                route_to=None,
                saved_at_step=6,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.POSSIBLE_DUPLICATE_REVIEW],
            evidence_refs=[
                "EX-POSSIBLE-DUP",
                "duplicate_check:vendor_amount_date",
                "CAND-AMT-02",
            ],
            route_to=None,
            saved_at_step=7,
        ),
        notes={},
        trace=ReviewTrace(
            ref_steps={
                "art-po": 1,
                "art-receipts": 2,
                "EX-POSSIBLE-DUP": 3,
                "duplicate_check:vendor_amount_date": 4,
                "CAND-AMT-02": 4,
            },
            steps_used=8,
        ),
    )

    assert best_report.decision_band is DecisionBand.BEST
    assert safe_hold_report.decision_band is DecisionBand.SAFE_SUBOPTIMAL
    assert wrong_heuristic_hold.decision_band is DecisionBand.WRONG
    assert best_report.total_score > safe_hold_report.total_score > wrong_heuristic_hold.total_score
    assert best_report.total_score > 0.95
    assert 0.55 < safe_hold_report.total_score < 0.75
    assert wrong_heuristic_hold.total_score < 0.30


def test_medium_approval_without_duplicate_clearance_is_wrong() -> None:
    scenario = load_scenario(task_id="medium")
    trace = ReviewTrace(
        ref_steps={
            "art-po": 1,
            "art-receipts": 2,
            "EX-POSSIBLE-DUP": 3,
        },
        steps_used=7,
    )

    report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
                line_id="L1",
                disposition=Disposition.APPROVE,
                reason_codes=[ReasonCode.SAFE_TO_PAY],
                evidence_refs=["art-po", "art-receipts", "EX-POSSIBLE-DUP"],
                route_to=None,
                saved_at_step=4,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.APPROVE,
                reason_codes=[ReasonCode.SAFE_TO_PAY],
                evidence_refs=["art-po", "art-receipts", "EX-POSSIBLE-DUP"],
                route_to=None,
                saved_at_step=5,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[ReasonCode.SAFE_TO_PAY],
            evidence_refs=["art-po", "art-receipts", "EX-POSSIBLE-DUP"],
            route_to=None,
            saved_at_step=6,
        ),
        notes={},
        trace=trace,
    )

    assert report.decision_band is DecisionBand.WRONG
    assert report.total_score < 0.30


def test_medium_observed_evidence_counts_even_if_final_refs_are_sparse() -> None:
    scenario = load_scenario(task_id="medium")
    trace = ReviewTrace(
        ref_steps={
            "art-po": 1,
            "art-receipts": 2,
            "EX-POSSIBLE-DUP": 3,
            "duplicate_check:normalized_invoice_no": 4,
            "CAND-NORM-01": 4,
        },
        steps_used=8,
    )

    report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
                line_id="L1",
                disposition=Disposition.APPROVE,
                reason_codes=[
                    ReasonCode.MATCHED_TO_PO_AND_RECEIPT,
                    ReasonCode.SAFE_TO_PAY,
                ],
                evidence_refs=["art-po", "CAND-NORM-01"],
                route_to=None,
                saved_at_step=5,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.APPROVE,
                reason_codes=[
                    ReasonCode.MATCHED_TO_PO_AND_RECEIPT,
                    ReasonCode.SAFE_TO_PAY,
                ],
                evidence_refs=["art-po"],
                route_to=None,
                saved_at_step=6,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[
                ReasonCode.MATCHED_TO_PO_AND_RECEIPT,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=["art-po", "CAND-NORM-01"],
            route_to=None,
            saved_at_step=7,
        ),
        notes={},
        trace=trace,
    )

    assert report.decision_band is DecisionBand.BEST
    assert 0.80 < report.total_score < 0.98
    assert report.evidence_score < 0.70


def test_medium_duplicate_hit_without_po_and_receipts_stays_wrong() -> None:
    scenario = load_scenario(task_id="medium")
    trace = ReviewTrace(
        ref_steps={
            "EX-POSSIBLE-DUP": 1,
            "duplicate_check:normalized_invoice_no": 2,
            "CAND-NORM-01": 2,
        },
        steps_used=6,
    )

    report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
                line_id="L1",
                disposition=Disposition.APPROVE,
                reason_codes=[ReasonCode.SAFE_TO_PAY],
                evidence_refs=["CAND-NORM-01"],
                route_to=None,
                saved_at_step=3,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.APPROVE,
                reason_codes=[ReasonCode.SAFE_TO_PAY],
                evidence_refs=["CAND-NORM-01"],
                route_to=None,
                saved_at_step=4,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[ReasonCode.SAFE_TO_PAY],
            evidence_refs=["CAND-NORM-01"],
            route_to=None,
            saved_at_step=5,
        ),
        notes={},
        trace=trace,
    )

    assert report.decision_band is DecisionBand.WRONG
    assert report.total_score < 0.30


def test_medium_plus_partial_release_creates_best_safe_wrong_and_unsafe_bands() -> None:
    scenario = load_scenario(task_id="medium_plus")
    full_trace = ReviewTrace(
        ref_steps={
            "art-invoice": 1,
            "art-po": 2,
            "art-receipts": 3,
            "EX-POSSIBLE-DUP": 4,
            "duplicate_check:normalized_invoice_no": 5,
            "CAND-NORM-01": 5,
            "EX-RECEIPT-L2": 6,
            "art-policy": 7,
        },
        steps_used=12,
    )
    conservative_trace = ReviewTrace(
        ref_steps={
            "art-invoice": 1,
            "art-po": 2,
            "art-receipts": 3,
            "EX-POSSIBLE-DUP": 4,
            "duplicate_check:normalized_invoice_no": 5,
            "CAND-NORM-01": 5,
            "EX-RECEIPT-L2": 6,
        },
        steps_used=11,
    )

    best_report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
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
                route_to=None,
                saved_at_step=8,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
                evidence_refs=[
                    "art-invoice",
                    "art-receipts",
                    "EX-RECEIPT-L2",
                    "art-policy",
                ],
                route_to=RouteTarget.RECEIVING,
                saved_at_step=9,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
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
            route_to=None,
            saved_at_step=10,
        ),
        notes={},
        trace=full_trace,
    )

    safe_report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
                line_id="L1",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.POSSIBLE_DUPLICATE_REVIEW],
                evidence_refs=[
                    "art-po",
                    "art-receipts",
                    "EX-POSSIBLE-DUP",
                    "duplicate_check:normalized_invoice_no",
                    "CAND-NORM-01",
                ],
                route_to=None,
                saved_at_step=7,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
                evidence_refs=["art-receipts", "EX-RECEIPT-L2"],
                route_to=RouteTarget.RECEIVING,
                saved_at_step=8,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[
                ReasonCode.POSSIBLE_DUPLICATE_REVIEW,
                ReasonCode.RECEIPT_NOT_CONFIRMED,
            ],
            evidence_refs=[
                "art-receipts",
                "EX-RECEIPT-L2",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
            route_to=None,
            saved_at_step=9,
        ),
        notes={},
        trace=conservative_trace,
    )

    wrong_report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
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
                route_to=None,
                saved_at_step=7,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
                evidence_refs=["art-invoice", "art-receipts", "EX-RECEIPT-L2"],
                route_to=RouteTarget.RECEIVING,
                saved_at_step=8,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[
                ReasonCode.POSSIBLE_DUPLICATE_REVIEW,
                ReasonCode.RECEIPT_NOT_CONFIRMED,
                ReasonCode.SAFE_TO_PAY,
            ],
            evidence_refs=[
                "art-receipts",
                "EX-RECEIPT-L2",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
            route_to=None,
            saved_at_step=9,
        ),
        notes={},
        trace=conservative_trace,
    )

    unsafe_report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
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
                route_to=None,
                saved_at_step=8,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.APPROVE,
                reason_codes=[ReasonCode.SAFE_TO_PAY],
                evidence_refs=[
                    "art-invoice",
                    "art-receipts",
                    "EX-RECEIPT-L2",
                    "art-policy",
                ],
                route_to=None,
                saved_at_step=9,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[ReasonCode.SAFE_TO_PAY],
            evidence_refs=[
                "art-policy",
                "art-receipts",
                "EX-RECEIPT-L2",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
            route_to=None,
            saved_at_step=10,
        ),
        notes={},
        trace=full_trace,
    )

    assert best_report.decision_band is DecisionBand.BEST
    assert safe_report.decision_band is DecisionBand.SAFE_SUBOPTIMAL
    assert wrong_report.decision_band is DecisionBand.WRONG
    assert unsafe_report.decision_band is DecisionBand.UNSAFE
    assert (
        best_report.total_score
        > safe_report.total_score
        > wrong_report.total_score
        > unsafe_report.total_score
    )
    assert best_report.total_score > 0.95
    assert 0.55 < safe_report.total_score < 0.80
    assert wrong_report.total_score < 0.45
    assert unsafe_report.total_score < 0.15


def test_hard_composition_rewards_mixed_judgment_and_penalizes_templates() -> None:
    scenario = load_scenario(task_id="hard")
    full_trace = ReviewTrace(
        ref_steps={
            "art-invoice": 1,
            "EX-POSSIBLE-DUP": 2,
            "duplicate_check:normalized_invoice_no": 3,
            "CAND-NORM-01": 3,
            "art-receipts": 4,
            "EX-RECEIPT-L1": 5,
            "EX-RECEIPT-L2": 6,
            "art-history": 7,
            "art-vendor": 8,
            "EX-TAX-001": 9,
            "art-policy": 10,
        },
        steps_used=17,
    )

    best_report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
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
                route_to=None,
                saved_at_step=12,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
                evidence_refs=[
                    "art-receipts",
                    "art-history",
                    "EX-RECEIPT-L2",
                    "art-policy",
                ],
                route_to=RouteTarget.RECEIVING,
                saved_at_step=13,
            ),
            "L3": LineResolution(
                resolution_id="LR-L3",
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
                route_to=None,
                saved_at_step=14,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.TAX_AMOUNT_MISMATCH],
            evidence_refs=[
                "art-invoice",
                "art-vendor",
                "art-policy",
                "EX-TAX-001",
            ],
            route_to=RouteTarget.TAX,
            saved_at_step=15,
        ),
        notes={},
        trace=full_trace,
    )

    blanket_hold = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
                line_id="L1",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.PARTIAL_RECEIPT_PENDING],
                evidence_refs=[
                    "duplicate_check:normalized_invoice_no",
                    "CAND-NORM-01",
                    "EX-RECEIPT-L1",
                    "art-policy",
                ],
                route_to=None,
                saved_at_step=12,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
                evidence_refs=["art-history", "EX-RECEIPT-L2"],
                route_to=RouteTarget.RECEIVING,
                saved_at_step=13,
            ),
            "L3": LineResolution(
                resolution_id="LR-L3",
                line_id="L3",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.POSSIBLE_DUPLICATE_REVIEW],
                evidence_refs=[
                    "duplicate_check:normalized_invoice_no",
                    "CAND-NORM-01",
                ],
                route_to=None,
                saved_at_step=14,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.TAX_AMOUNT_MISMATCH],
            evidence_refs=["art-vendor", "art-policy", "EX-TAX-001"],
            route_to=RouteTarget.TAX,
            saved_at_step=15,
        ),
        notes={},
        trace=full_trace,
    )

    unsafe_release = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
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
                route_to=None,
                saved_at_step=12,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
                evidence_refs=[
                    "art-receipts",
                    "art-history",
                    "EX-RECEIPT-L2",
                    "art-policy",
                ],
                route_to=RouteTarget.RECEIVING,
                saved_at_step=13,
            ),
            "L3": LineResolution(
                resolution_id="LR-L3",
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
                route_to=None,
                saved_at_step=14,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.RELEASE_APPROVED_LINES,
            reason_codes=[ReasonCode.SAFE_TO_PAY],
            evidence_refs=[
                "art-po",
                "art-receipts",
                "duplicate_check:normalized_invoice_no",
                "CAND-NORM-01",
            ],
            route_to=None,
            saved_at_step=15,
        ),
        notes={},
        trace=full_trace,
    )

    assert best_report.decision_band is DecisionBand.BEST
    assert blanket_hold.decision_band is DecisionBand.SAFE_SUBOPTIMAL
    assert unsafe_release.decision_band is DecisionBand.UNSAFE
    assert best_report.total_score > blanket_hold.total_score > unsafe_release.total_score
    assert best_report.total_score > 0.95
    assert 0.55 < blanket_hold.total_score < 0.75
    assert unsafe_release.total_score <= 0.15


def test_hard_conservative_partial_evidence_scores_safe_suboptimal() -> None:
    scenario = load_scenario(task_id="hard")
    partial_trace = ReviewTrace(
        ref_steps={
            "EX-POSSIBLE-DUP": 1,
            "duplicate_check:normalized_invoice_no": 2,
            "CAND-NORM-01": 2,
            "EX-RECEIPT-L1": 3,
            "EX-RECEIPT-L2": 4,
            "art-receipts": 5,
            "art-vendor": 6,
            "art-policy": 7,
            "EX-TAX-001": 8,
        },
        steps_used=12,
    )

    report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
                line_id="L1",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.PARTIAL_RECEIPT_PENDING],
                evidence_refs=[
                    "art-receipts",
                    "duplicate_check:normalized_invoice_no",
                    "CAND-NORM-01",
                    "EX-RECEIPT-L1",
                ],
                route_to=None,
                saved_at_step=8,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
                evidence_refs=["art-receipts", "EX-RECEIPT-L2"],
                route_to=RouteTarget.RECEIVING,
                saved_at_step=9,
            ),
            "L3": LineResolution(
                resolution_id="LR-L3",
                line_id="L3",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.POSSIBLE_DUPLICATE_REVIEW],
                evidence_refs=[
                    "art-receipts",
                    "duplicate_check:normalized_invoice_no",
                    "CAND-NORM-01",
                ],
                route_to=None,
                saved_at_step=10,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.TAX_AMOUNT_MISMATCH],
            evidence_refs=["EX-TAX-001", "art-vendor", "art-policy"],
            route_to=RouteTarget.TAX,
            saved_at_step=11,
        ),
        notes={},
        trace=partial_trace,
    )

    assert report.decision_band is DecisionBand.SAFE_SUBOPTIMAL
    assert 0.55 < report.total_score < 0.80


def test_hard_shortcut_approvals_without_invoice_and_history_stay_wrong() -> None:
    scenario = load_scenario(task_id="hard")
    shortcut_trace = ReviewTrace(
        ref_steps={
            "EX-POSSIBLE-DUP": 1,
            "duplicate_check:normalized_invoice_no": 2,
            "CAND-NORM-01": 2,
            "EX-RECEIPT-L1": 3,
            "EX-RECEIPT-L2": 4,
            "art-receipts": 5,
            "art-vendor": 6,
            "art-policy": 7,
            "EX-TAX-001": 8,
        },
        steps_used=12,
    )

    report = grade_case(
        scenario,
        line_resolutions={
            "L1": LineResolution(
                resolution_id="LR-L1",
                line_id="L1",
                disposition=Disposition.APPROVE,
                reason_codes=[ReasonCode.PARTIAL_RECEIPT_PENDING],
                evidence_refs=["art-receipts", "EX-RECEIPT-L1"],
                route_to=None,
                saved_at_step=8,
            ),
            "L2": LineResolution(
                resolution_id="LR-L2",
                line_id="L2",
                disposition=Disposition.HOLD,
                reason_codes=[ReasonCode.RECEIPT_NOT_CONFIRMED],
                evidence_refs=["art-receipts", "EX-RECEIPT-L2"],
                route_to=RouteTarget.RECEIVING,
                saved_at_step=9,
            ),
            "L3": LineResolution(
                resolution_id="LR-L3",
                line_id="L3",
                disposition=Disposition.APPROVE,
                reason_codes=[ReasonCode.MATCHED_TO_PO_AND_RECEIPT],
                evidence_refs=["art-receipts"],
                route_to=None,
                saved_at_step=10,
            ),
        },
        header_resolution=HeaderResolution(
            resolution_id="HR-001",
            payment_recommendation=PaymentRecommendation.HOLD_FULL_INVOICE,
            reason_codes=[ReasonCode.TAX_AMOUNT_MISMATCH],
            evidence_refs=["EX-TAX-001", "art-vendor", "art-policy"],
            route_to=RouteTarget.TAX,
            saved_at_step=11,
        ),
        notes={},
        trace=shortcut_trace,
    )

    assert report.decision_band is DecisionBand.WRONG
    assert report.core_decision_score < 0.35
    assert report.total_score < 0.35
