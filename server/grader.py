"""Deterministic grading for InvoiceOps cases."""

from __future__ import annotations

from dataclasses import dataclass

from invoiceops_env.models import (
    CaseNote,
    DecisionBand,
    Disposition,
    HeaderResolution,
    HeaderScoreReport,
    IssueNoteReport,
    LineResolution,
    LineScoreReport,
    PaymentRecommendation,
    ReasonCode,
    SubmissionReport,
)
from invoiceops_env.server.scenario_loader import (
    HeaderExpectation,
    NoteExpectation,
    ResolutionExpectation,
    ScenarioFixture,
)


LINE_DISPOSITION_WEIGHT = 0.55
LINE_REASON_WEIGHT = 0.15
LINE_ROUTE_WEIGHT = 0.30

HEADER_RECOMMENDATION_WEIGHT = 0.55
HEADER_REASON_WEIGHT = 0.15
HEADER_ROUTE_WEIGHT = 0.30

NOTE_REASON_WEIGHT = 0.65
NOTE_EVIDENCE_WEIGHT = 0.35

AUX_REASON_QUALITY_WEIGHT = 0.25
AUX_EVIDENCE_WEIGHT = 0.45
AUX_DOCUMENTATION_WEIGHT = 0.20
AUX_EFFICIENCY_WEIGHT = 0.10

BAND_CORE_WEIGHT = 0.60
BAND_AUXILIARY_WEIGHT = 0.40

BAND_RANGES: dict[DecisionBand, tuple[float, float]] = {
    DecisionBand.BEST: (0.80, 1.00),
    DecisionBand.SAFE_SUBOPTIMAL: (0.50, 0.79),
    DecisionBand.WRONG: (0.05, 0.45),
    DecisionBand.UNSAFE: (0.00, 0.15),
}

HEADER_TO_LINE_DISPOSITION: dict[PaymentRecommendation, Disposition] = {
    PaymentRecommendation.RELEASE_APPROVED_LINES: Disposition.APPROVE,
    PaymentRecommendation.HOLD_FULL_INVOICE: Disposition.HOLD,
    PaymentRecommendation.REJECT_FULL_INVOICE: Disposition.REJECT,
    PaymentRecommendation.ESCALATE_CASE: Disposition.ESCALATE,
}

CONSERVATIVE_LINE_DISPOSITIONS = {
    Disposition.HOLD,
    Disposition.ESCALATE,
    Disposition.REJECT,
}

CONSERVATIVE_HEADER_RECOMMENDATIONS = {
    PaymentRecommendation.HOLD_FULL_INVOICE,
    PaymentRecommendation.ESCALATE_CASE,
    PaymentRecommendation.REJECT_FULL_INVOICE,
}


@dataclass(frozen=True)
class ReviewTrace:
    ref_steps: dict[str, int]
    steps_used: int
    invalid_actions: int = 0
    redundant_actions: int = 0


def _f1(predicted: set[str], expected: set[str]) -> float:
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0

    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted)
    recall = true_positives / len(expected)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _reason_score(
    reason_codes: list[ReasonCode] | None,
    accepted_reason_sets: list[list[str]],
) -> float:
    if not accepted_reason_sets:
        return 1.0
    if not reason_codes:
        return 0.0

    predicted = {reason.value for reason in reason_codes}
    return max(_f1(predicted, set(expected)) for expected in accepted_reason_sets)


def _route_score(route_to_value: str | None, accepted_routes: list[str]) -> float:
    if not accepted_routes:
        return 1.0
    if route_to_value is None:
        return 0.0
    return 1.0 if route_to_value in accepted_routes else 0.0


def _normalized_weighted_score(components: list[tuple[float, float]]) -> float:
    active_weight = sum(weight for _, weight in components if weight > 0.0)
    if active_weight <= 0.0:
        return 0.0
    return sum(score * weight for score, weight in components if weight > 0.0) / active_weight


def _timely_refs(
    cited_refs: list[str] | None,
    ref_steps: dict[str, int],
    saved_at_step: int | None,
) -> set[str]:
    if saved_at_step is None or not cited_refs:
        return set()

    timely_refs: set[str] = set()
    for ref in cited_refs:
        ref_step = ref_steps.get(ref)
        if ref_step is not None and ref_step < saved_at_step:
            timely_refs.add(ref)
    return timely_refs


def _observed_refs_before_step(
    ref_steps: dict[str, int],
    saved_at_step: int | None,
) -> set[str]:
    if saved_at_step is None:
        return set()
    return {
        ref
        for ref, ref_step in ref_steps.items()
        if ref_step < saved_at_step
    }


def _evidence_score(
    cited_refs: list[str] | None,
    decisive_refs: list[str],
    ref_steps: dict[str, int],
    saved_at_step: int | None,
) -> float:
    if not decisive_refs:
        return 1.0

    timely_cited_refs = _timely_refs(cited_refs, ref_steps, saved_at_step)
    if not timely_cited_refs:
        return 0.0
    return _f1(timely_cited_refs, set(decisive_refs))


def _gating_refs_satisfied(
    gating_refs: list[str],
    ref_steps: dict[str, int],
    saved_at_step: int | None,
) -> bool:
    if not gating_refs:
        return True
    # Band gating depends on what the agent had already uncovered in time,
    # not on whether every observed ref was restated in the saved action.
    observed_refs = _observed_refs_before_step(ref_steps, saved_at_step)
    return set(gating_refs).issubset(observed_refs)


def _accepted_dispositions(score_map: dict[str, float]) -> list[Disposition]:
    return [
        disposition
        for disposition in Disposition
        if score_map.get(disposition.value, 0.0) > 0.0
    ]


def _accepted_recommendations(
    score_map: dict[str, float],
) -> list[PaymentRecommendation]:
    return [
        recommendation
        for recommendation in PaymentRecommendation
        if score_map.get(recommendation.value, 0.0) > 0.0
    ]


def _max_positive_score(score_map: dict[str, float]) -> float:
    positive_scores = [score for score in score_map.values() if score > 0.0]
    return max(positive_scores) if positive_scores else 0.0


def _max_suboptimal_positive_score(score_map: dict[str, float]) -> float:
    best_score = _max_positive_score(score_map)
    suboptimal_positive_scores = [
        score for score in score_map.values() if 0.0 < score < best_score
    ]
    return max(suboptimal_positive_scores) if suboptimal_positive_scores else 0.0


def _core_line_score(disposition_score: float, route_score: float, has_route: bool) -> float:
    return _normalized_weighted_score(
        [
            (disposition_score, 0.70),
            (route_score, 0.30 if has_route else 0.0),
        ]
    )


def _core_header_score(
    recommendation_score: float,
    route_score: float,
    has_route: bool,
) -> float:
    return _normalized_weighted_score(
        [
            (recommendation_score, 0.70),
            (route_score, 0.30 if has_route else 0.0),
        ]
    )


def _grade_note_expectation(
    expectation: NoteExpectation,
    notes: list[CaseNote],
    trace: ReviewTrace,
) -> IssueNoteReport:
    best_note_score = 0.0
    best_reason = 0.0
    best_evidence = 0.0
    for note in notes:
        reason_score = _reason_score(note.reason_codes, expectation.accepted_reason_sets)
        evidence_score = _evidence_score(
            note.evidence_refs,
            expectation.decisive_refs,
            trace.ref_steps,
            note.saved_at_step,
        )
        note_score = (
            (NOTE_REASON_WEIGHT * reason_score)
            + (NOTE_EVIDENCE_WEIGHT * evidence_score)
        )
        if note_score > best_note_score:
            best_note_score = note_score
            best_reason = reason_score
            best_evidence = evidence_score

    return IssueNoteReport(
        issue_id=expectation.issue_id,
        note_score=round(best_note_score, 4),
        reason_score=round(best_reason, 4),
        evidence_score=round(best_evidence, 4),
    )


def _grade_header(
    expectation: HeaderExpectation,
    header_resolution: HeaderResolution | None,
    ref_steps: dict[str, int],
) -> HeaderScoreReport:
    recommendation_value = (
        header_resolution.payment_recommendation.value
        if header_resolution is not None
        else None
    )
    recommendation_score = (
        expectation.score_map.get(recommendation_value or "", 0.0)
        if header_resolution is not None
        else 0.0
    )
    reason_score = _reason_score(
        header_resolution.reason_codes if header_resolution is not None else None,
        expectation.accepted_reason_sets,
    )
    route_score = _route_score(
        (
            header_resolution.route_to.value
            if header_resolution is not None and header_resolution.route_to is not None
            else None
        ),
        expectation.accepted_routes,
    )
    evidence_score = _evidence_score(
        header_resolution.evidence_refs if header_resolution is not None else None,
        expectation.decisive_refs,
        ref_steps,
        header_resolution.saved_at_step if header_resolution is not None else None,
    )
    header_score = _normalized_weighted_score(
        [
            (recommendation_score, HEADER_RECOMMENDATION_WEIGHT),
            (
                reason_score,
                HEADER_REASON_WEIGHT if expectation.accepted_reason_sets else 0.0,
            ),
            (route_score, HEADER_ROUTE_WEIGHT if expectation.accepted_routes else 0.0),
        ]
    )
    return HeaderScoreReport(
        header_score=round(header_score, 4),
        recommendation_score=round(recommendation_score, 4),
        reason_score=round(reason_score, 4),
        route_score=round(route_score, 4),
        evidence_score=round(evidence_score, 4),
        accepted_recommendations=_accepted_recommendations(expectation.score_map),
    )


def _mirrored_single_line_resolution(
    scenario: ScenarioFixture,
    line_resolutions: dict[str, LineResolution],
    header_resolution: HeaderResolution | None,
) -> dict[str, LineResolution]:
    # Single-line warm-up cases should not crater solely because the agent saved
    # the correct header decision but omitted the redundant line decision.
    if header_resolution is None or len(scenario.hidden_truth.line_expectations) != 1:
        return line_resolutions

    line_id, expectation = next(iter(scenario.hidden_truth.line_expectations.items()))
    if line_id in line_resolutions:
        return line_resolutions

    route_value = (
        header_resolution.route_to.value
        if header_resolution.route_to is not None
        else None
    )
    if expectation.accepted_routes and route_value not in expectation.accepted_routes:
        return line_resolutions

    mirrored_resolution = LineResolution(
        resolution_id=f"{header_resolution.resolution_id}-mirror-line",
        line_id=line_id,
        disposition=HEADER_TO_LINE_DISPOSITION[header_resolution.payment_recommendation],
        reason_codes=list(header_resolution.reason_codes),
        evidence_refs=list(header_resolution.evidence_refs),
        route_to=header_resolution.route_to,
        saved_at_step=header_resolution.saved_at_step,
    )
    return {**line_resolutions, line_id: mirrored_resolution}


def _line_is_best(
    expectation: ResolutionExpectation,
    disposition_score: float,
    route_score: float,
    gating_ok: bool,
) -> bool:
    if not gating_ok:
        return False
    if disposition_score <= 0.0:
        return False
    if expectation.accepted_routes and route_score <= 0.0:
        return False
    return disposition_score >= _max_positive_score(expectation.score_map)


def _line_is_safe(
    expectation: ResolutionExpectation,
    resolution: LineResolution | None,
    disposition_score: float,
    route_score: float,
    best_gating_ok: bool,
    safe_gating_ok: bool,
) -> bool:
    if resolution is None:
        return False
    if disposition_score <= 0.0:
        return False
    if expectation.accepted_routes and route_score <= 0.0:
        return False
    if best_gating_ok:
        return True
    return (
        safe_gating_ok
        and resolution.disposition in CONSERVATIVE_LINE_DISPOSITIONS
    )


def _header_is_best(
    expectation: HeaderExpectation,
    recommendation_score: float,
    route_score: float,
    gating_ok: bool,
) -> bool:
    if not gating_ok:
        return False
    if recommendation_score <= 0.0:
        return False
    if expectation.accepted_routes and route_score <= 0.0:
        return False
    return recommendation_score >= _max_positive_score(expectation.score_map)


def _header_is_safe(
    expectation: HeaderExpectation,
    header_resolution: HeaderResolution | None,
    recommendation_score: float,
    route_score: float,
    best_gating_ok: bool,
    safe_gating_ok: bool,
) -> bool:
    if header_resolution is None:
        return False
    if recommendation_score <= 0.0:
        return False
    if expectation.accepted_routes and route_score <= 0.0:
        return False
    if best_gating_ok:
        return True
    return (
        safe_gating_ok
        and header_resolution.payment_recommendation
        in CONSERVATIVE_HEADER_RECOMMENDATIONS
    )


def grade_case(
    scenario: ScenarioFixture,
    line_resolutions: dict[str, LineResolution],
    header_resolution: HeaderResolution | None,
    notes: dict[str, CaseNote],
    trace: ReviewTrace,
) -> SubmissionReport:
    line_resolutions = _mirrored_single_line_resolution(
        scenario,
        line_resolutions,
        header_resolution,
    )

    line_reports: list[LineScoreReport] = []
    weighted_line_resolution = 0.0
    weighted_line_core = 0.0
    weighted_line_reason = 0.0
    weighted_line_evidence = 0.0
    total_amount = sum(
        expectation.amount
        for expectation in scenario.hidden_truth.line_expectations.values()
    )
    total_amount = total_amount or 1.0

    unsafe_findings: list[str] = []
    all_lines_best = True
    all_lines_safe = True

    for line_id, expectation in scenario.hidden_truth.line_expectations.items():
        resolution = line_resolutions.get(line_id)
        disposition_value = resolution.disposition.value if resolution is not None else ""
        disposition_score = expectation.score_map.get(disposition_value, 0.0)
        reason_score = _reason_score(
            resolution.reason_codes if resolution is not None else None,
            expectation.accepted_reason_sets,
        )
        route_score = _route_score(
            (
                resolution.route_to.value
                if resolution is not None and resolution.route_to is not None
                else None
            ),
            expectation.accepted_routes,
        )
        evidence_score = _evidence_score(
            resolution.evidence_refs if resolution is not None else None,
            expectation.decisive_refs,
            trace.ref_steps,
            resolution.saved_at_step if resolution is not None else None,
        )
        best_gating_ok = _gating_refs_satisfied(
            expectation.gating_refs,
            trace.ref_steps,
            resolution.saved_at_step if resolution is not None else None,
        )
        safe_gating_ok = _gating_refs_satisfied(
            expectation.safe_gating_refs or expectation.gating_refs,
            trace.ref_steps,
            resolution.saved_at_step if resolution is not None else None,
        )

        line_score = _normalized_weighted_score(
            [
                (disposition_score, LINE_DISPOSITION_WEIGHT),
                (
                    reason_score,
                    LINE_REASON_WEIGHT if expectation.accepted_reason_sets else 0.0,
                ),
                (route_score, LINE_ROUTE_WEIGHT if expectation.accepted_routes else 0.0),
            ]
        )
        core_score = _core_line_score(
            disposition_score,
            route_score,
            bool(expectation.accepted_routes),
        )
        effective_core_score = 0.0
        # Best credit needs the full best gating refs. Conservative actions can
        # still earn capped core credit when the scenario defines safe gating.
        if best_gating_ok:
            effective_core_score = core_score
        elif (
            resolution is not None
            and safe_gating_ok
            and disposition_score > 0.0
            and resolution.disposition in CONSERVATIVE_LINE_DISPOSITIONS
        ):
            capped_disposition_score = min(
                disposition_score,
                _max_suboptimal_positive_score(expectation.score_map),
            )
            if capped_disposition_score > 0.0:
                effective_core_score = _core_line_score(
                    capped_disposition_score,
                    route_score,
                    bool(expectation.accepted_routes),
                )
        weight = expectation.amount / total_amount
        weighted_line_resolution += line_score * weight
        weighted_line_core += effective_core_score * weight
        weighted_line_reason += reason_score * weight
        weighted_line_evidence += evidence_score * weight
        line_reports.append(
            LineScoreReport(
                line_id=line_id,
                line_score=round(line_score, 4),
                disposition_score=round(disposition_score, 4),
                reason_score=round(reason_score, 4),
                route_score=round(route_score, 4),
                evidence_score=round(evidence_score, 4),
                accepted_dispositions=_accepted_dispositions(expectation.score_map),
            )
        )

        all_lines_best = all_lines_best and _line_is_best(
            expectation,
            disposition_score,
            route_score,
            best_gating_ok,
        )
        all_lines_safe = all_lines_safe and _line_is_safe(
            expectation,
            resolution,
            disposition_score,
            route_score,
            best_gating_ok,
            safe_gating_ok,
        )

        if (
            expectation.unsafe_approve
            and resolution is not None
            and resolution.disposition is Disposition.APPROVE
        ):
            unsafe_findings.append(f"unsafe approval on line {line_id}")

    header_report = _grade_header(
        scenario.hidden_truth.header_expectation,
        header_resolution,
        trace.ref_steps,
    )
    header_best_gating_ok = _gating_refs_satisfied(
        scenario.hidden_truth.header_expectation.gating_refs,
        trace.ref_steps,
        header_resolution.saved_at_step if header_resolution is not None else None,
    )
    header_safe_gating_ok = _gating_refs_satisfied(
        scenario.hidden_truth.header_expectation.safe_gating_refs
        or scenario.hidden_truth.header_expectation.gating_refs,
        trace.ref_steps,
        header_resolution.saved_at_step if header_resolution is not None else None,
    )

    header_core_score = _core_header_score(
        header_report.recommendation_score,
        header_report.route_score,
        bool(scenario.hidden_truth.header_expectation.accepted_routes),
    )
    if header_best_gating_ok:
        pass
    elif (
        header_resolution is not None
        and header_safe_gating_ok
        and header_report.recommendation_score > 0.0
        and header_resolution.payment_recommendation
        in CONSERVATIVE_HEADER_RECOMMENDATIONS
    ):
        capped_recommendation_score = min(
            header_report.recommendation_score,
            _max_suboptimal_positive_score(
                scenario.hidden_truth.header_expectation.score_map
            ),
        )
        if capped_recommendation_score > 0.0:
            header_core_score = _core_header_score(
                capped_recommendation_score,
                header_report.route_score,
                bool(scenario.hidden_truth.header_expectation.accepted_routes),
            )
        else:
            header_core_score = 0.0
    else:
        header_core_score = 0.0
    reason_quality_score = (0.80 * weighted_line_reason) + (
        0.20 * header_report.reason_score
    )
    resolution_score = (0.80 * weighted_line_resolution) + (
        0.20 * header_report.header_score
    )
    core_decision_score = (0.80 * weighted_line_core) + (0.20 * header_core_score)
    evidence_score = (0.80 * weighted_line_evidence) + (
        0.20 * header_report.evidence_score
    )

    note_reports = [
        _grade_note_expectation(expectation, list(notes.values()), trace)
        for expectation in scenario.hidden_truth.note_expectations
    ]
    if note_reports:
        documentation_score = sum(
            report.note_score for report in note_reports
        ) / len(note_reports)
    else:
        documentation_score = 1.0

    extra_steps = max(
        0,
        trace.steps_used - scenario.hidden_truth.efficient_step_target,
    )
    efficiency_score = max(
        0.0,
        1.0
        - (0.08 * extra_steps)
        - (0.25 * trace.invalid_actions)
        - (0.08 * trace.redundant_actions),
    )

    if header_resolution is not None:
        header_value = header_resolution.payment_recommendation.value
        if header_value in scenario.hidden_truth.header_expectation.unsafe_recommendations:
            unsafe_findings.append(f"unsafe header recommendation {header_value}")

    # Stage 1 assigns the decision band from the gated essential decisions.
    # Stage 2 later scores within that band using evidence, notes, and efficiency.
    if unsafe_findings:
        decision_band = DecisionBand.UNSAFE
    elif all_lines_best and _header_is_best(
        scenario.hidden_truth.header_expectation,
        header_report.recommendation_score,
        header_report.route_score,
        header_best_gating_ok,
    ):
        decision_band = DecisionBand.BEST
    elif all_lines_safe and _header_is_safe(
        scenario.hidden_truth.header_expectation,
        header_resolution,
        header_report.recommendation_score,
        header_report.route_score,
        header_best_gating_ok,
        header_safe_gating_ok,
    ):
        decision_band = DecisionBand.SAFE_SUBOPTIMAL
    else:
        decision_band = DecisionBand.WRONG

    auxiliary_score = _normalized_weighted_score(
        [
            (reason_quality_score, AUX_REASON_QUALITY_WEIGHT),
            (evidence_score, AUX_EVIDENCE_WEIGHT),
            (documentation_score, AUX_DOCUMENTATION_WEIGHT),
            (efficiency_score, AUX_EFFICIENCY_WEIGHT),
        ]
    )

    band_progress = _normalized_weighted_score(
        [
            (core_decision_score, BAND_CORE_WEIGHT),
            (auxiliary_score, BAND_AUXILIARY_WEIGHT),
        ]
    )
    band_floor, band_ceiling = BAND_RANGES[decision_band]
    total_score = band_floor + ((band_ceiling - band_floor) * band_progress)
    total_score = max(0.0, min(1.0, total_score))

    return SubmissionReport(
        decision_band=decision_band,
        total_score=round(total_score, 4),
        core_decision_score=round(core_decision_score, 4),
        reason_quality_score=round(reason_quality_score, 4),
        auxiliary_score=round(auxiliary_score, 4),
        resolution_score=round(resolution_score, 4),
        evidence_score=round(evidence_score, 4),
        documentation_score=round(documentation_score, 4),
        efficiency_score=round(efficiency_score, 4),
        safety_cap_applied=(
            round(BAND_RANGES[DecisionBand.UNSAFE][1], 4)
            if decision_band is DecisionBand.UNSAFE
            else None
        ),
        unsafe_findings=unsafe_findings,
        line_reports=line_reports,
        header_report=header_report,
        note_reports=note_reports,
    )
