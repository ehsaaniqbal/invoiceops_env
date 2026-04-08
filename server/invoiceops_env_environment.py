"""InvoiceOps environment implementation."""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from invoiceops_env.models import (
    ActionType,
    CaseNote,
    Disposition,
    DuplicateCandidate,
    HeaderResolution,
    InvoiceOpsAction,
    InvoiceOpsObservation,
    InvoiceOpsState,
    LineResolution,
    PaymentRecommendation,
    Progress,
)
from invoiceops_env.server.fixtures import ENV_DESCRIPTION
from invoiceops_env.server.fixtures import DUPLICATE_CHECK_REF_PREFIX
from invoiceops_env.server.grader import ReviewTrace, grade_case
from invoiceops_env.server.reward_engine import DEFAULT_REWARD_CONFIG
from invoiceops_env.server.scenario_loader import (
    ScenarioFixture,
    artifact_lookup,
    artifact_references,
    exception_lookup,
    exception_summaries,
    line_ids_for_scenario,
    load_scenario,
)


class InvoiceOpsEnvironment(
    Environment[InvoiceOpsAction, InvoiceOpsObservation, InvoiceOpsState]
):
    """Accounts-payable invoice exception handling environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._reward_config = DEFAULT_REWARD_CONFIG
        self._scenario: ScenarioFixture | None = None
        self._artifact_map = {}
        self._exception_map = {}
        self._state = InvoiceOpsState(episode_id=str(uuid4()), step_count=0)
        self._opened_artifact_ids: set[str] = set()
        self._inspected_exception_ids: set[str] = set()
        self._duplicate_checks_run: set[str] = set()
        self._duplicate_candidates: list[DuplicateCandidate] = []
        self._notes: dict[str, CaseNote] = {}
        self._line_resolutions: dict[str, LineResolution] = {}
        self._header_resolution: HeaderResolution | None = None
        self._ref_steps: dict[str, int] = {}
        self._invalid_actions = 0
        self._redundant_actions = 0
        self._submitted = False
        self._submission_report = None
        self._current_artifact_id: str | None = None
        self._current_exception_id: str | None = None

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="InvoiceOpsEnvironment",
            description=ENV_DESCRIPTION,
            version="0.1.0",
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        scenario_id: str | None = None,
        **kwargs: object,
    ) -> InvoiceOpsObservation:
        del seed, kwargs

        self._scenario = load_scenario(task_id=task_id, scenario_id=scenario_id)
        self._artifact_map = artifact_lookup(self._scenario)
        self._exception_map = exception_lookup(self._scenario)
        self._state = InvoiceOpsState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._scenario.task_id,
            scenario_id=self._scenario.scenario_id,
            case_id=self._scenario.case_id,
            current_artifact_id=None,
            submitted=False,
            step_limit=self._scenario.step_limit,
            duplicate_checks_run=0,
            invalid_actions=0,
            redundant_actions=0,
        )
        self._opened_artifact_ids = set()
        self._inspected_exception_ids = set()
        self._duplicate_checks_run = set()
        self._duplicate_candidates = []
        self._notes = {}
        self._line_resolutions = {}
        self._header_resolution = None
        self._ref_steps = {}
        self._invalid_actions = 0
        self._redundant_actions = 0
        self._submitted = False
        self._submission_report = None
        self._current_artifact_id = None
        self._current_exception_id = None

        return self._build_observation(
            message=f"{self._scenario.title} ready.",
            reward=0.0,
            done=False,
        )

    def step(
        self,
        action: InvoiceOpsAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> InvoiceOpsObservation:
        del timeout_s, kwargs
        if self._scenario is None:
            raise RuntimeError("reset() must be called before step()")

        if self._submitted:
            return self._invalid_observation(
                "Case already submitted.",
                self._reward_config.invalid_action_penalty,
                done=True,
            )

        self._state.step_count += 1
        reward = self._reward_config.step_cost
        done = False
        message = "Action processed."

        match action.action_type:
            case ActionType.OPEN_ARTIFACT:
                artifact_id = action.artifact_id or ""
                artifact = self._artifact_map.get(artifact_id)
                if artifact is None:
                    return self._invalid_observation(
                        f"Unknown artifact_id: {action.artifact_id}",
                        reward + self._reward_config.invalid_action_penalty,
                    )
                self._current_artifact_id = artifact_id
                self._state.current_artifact_id = artifact_id
                if artifact_id in self._opened_artifact_ids:
                    self._redundant_actions += 1
                    self._state.redundant_actions = self._redundant_actions
                    reward += self._reward_config.redundant_open_penalty
                    message = f"Artifact {artifact_id} was already opened."
                else:
                    self._opened_artifact_ids.add(artifact_id)
                    self._ref_steps[artifact_id] = self._state.step_count
                    reward += self._reward_config.first_open_artifact
                    message = f"Opened artifact {artifact_id}."

            case ActionType.INSPECT_EXCEPTION:
                exception_id = action.exception_id or ""
                exception = self._exception_map.get(exception_id)
                if exception is None:
                    return self._invalid_observation(
                        f"Unknown exception_id: {action.exception_id}",
                        reward + self._reward_config.invalid_action_penalty,
                    )
                self._current_exception_id = exception_id
                if exception_id not in self._inspected_exception_ids:
                    self._inspected_exception_ids.add(exception_id)
                    self._ref_steps[exception_id] = self._state.step_count
                    reward += self._reward_config.inspect_exception
                message = f"Inspected exception {exception_id}."

            case ActionType.RUN_DUPLICATE_CHECK:
                strategy = action.match_strategy
                assert strategy is not None
                if strategy.value in self._duplicate_checks_run:
                    self._redundant_actions += 1
                    self._state.redundant_actions = self._redundant_actions
                    reward += self._reward_config.redundant_duplicate_penalty
                    message = f"Duplicate check {strategy.value} was already run."
                else:
                    self._duplicate_checks_run.add(strategy.value)
                    self._state.duplicate_checks_run = len(self._duplicate_checks_run)
                    self._ref_steps[
                        f"{DUPLICATE_CHECK_REF_PREFIX}{strategy.value}"
                    ] = self._state.step_count
                    reward += self._reward_config.run_duplicate_check
                    self._duplicate_candidates = [
                        candidate
                        for candidate in self._scenario.duplicate_candidates
                        if strategy in candidate.supported_match_strategies
                    ]
                    for candidate in self._duplicate_candidates:
                        self._ref_steps.setdefault(
                            candidate.candidate_id,
                            self._state.step_count,
                        )
                    message = (
                        f"Duplicate search completed with {len(self._duplicate_candidates)} "
                        f"candidate(s)."
                    )

            case ActionType.ADD_NOTE:
                invalid_ref = self._first_invalid_ref(action.evidence_refs)
                if invalid_ref is not None:
                    return self._invalid_observation(
                        f"Unknown evidence ref: {invalid_ref}",
                        reward + self._reward_config.invalid_action_penalty,
                    )
                note_id = f"N-{len(self._notes) + 1:02d}"
                self._notes[note_id] = CaseNote(
                    note_id=note_id,
                    note_type=action.note_type,
                    reason_codes=action.reason_codes,
                    evidence_refs=action.evidence_refs,
                    text=(action.text or "").strip(),
                    saved_at_step=self._state.step_count,
                )
                reward += self._reward_config.valid_note
                message = f"Saved note {note_id}."

            case ActionType.SET_LINE_RESOLUTION:
                line_id = action.line_id or ""
                if line_id not in line_ids_for_scenario(self._scenario):
                    return self._invalid_observation(
                        f"Unknown line_id: {action.line_id}",
                        reward + self._reward_config.invalid_action_penalty,
                    )
                invalid_ref = self._first_invalid_ref(action.evidence_refs)
                if invalid_ref is not None:
                    return self._invalid_observation(
                        f"Unknown evidence ref: {invalid_ref}",
                        reward + self._reward_config.invalid_action_penalty,
                    )
                resolution_id = f"LR-{line_id}"
                is_revision = line_id in self._line_resolutions
                self._line_resolutions[line_id] = LineResolution(
                    resolution_id=resolution_id,
                    line_id=line_id,
                    disposition=action.disposition,
                    reason_codes=action.reason_codes,
                    evidence_refs=action.evidence_refs,
                    route_to=action.route_to,
                    saved_at_step=self._state.step_count,
                )
                reward += (
                    self._reward_config.revision_penalty
                    if is_revision
                    else self._reward_config.valid_line_resolution
                )
                if is_revision:
                    self._redundant_actions += 1
                    self._state.redundant_actions = self._redundant_actions
                message = f"Saved line resolution for {line_id}."

            case ActionType.SET_HEADER_RESOLUTION:
                invalid_ref = self._first_invalid_ref(action.evidence_refs)
                if invalid_ref is not None:
                    return self._invalid_observation(
                        f"Unknown evidence ref: {invalid_ref}",
                        reward + self._reward_config.invalid_action_penalty,
                    )
                is_revision = self._header_resolution is not None
                self._header_resolution = HeaderResolution(
                    resolution_id="HR-001",
                    payment_recommendation=action.payment_recommendation,
                    reason_codes=action.reason_codes,
                    evidence_refs=action.evidence_refs,
                    route_to=action.route_to,
                    saved_at_step=self._state.step_count,
                )
                reward += (
                    self._reward_config.revision_penalty
                    if is_revision
                    else self._reward_config.valid_header_resolution
                )
                if is_revision:
                    self._redundant_actions += 1
                    self._state.redundant_actions = self._redundant_actions
                message = "Saved header recommendation."

            case ActionType.SUBMIT_CASE:
                invalid_submission = self._validate_submission_refs(
                    action.note_ids,
                    action.line_resolution_ids,
                    action.header_resolution_id,
                )
                if invalid_submission is not None:
                    return self._invalid_observation(
                        invalid_submission,
                        reward + self._reward_config.invalid_action_penalty,
                    )
                consistency_error = self._validate_submission_consistency()
                if consistency_error is not None:
                    return self._invalid_observation(
                        consistency_error,
                        reward + self._reward_config.invalid_action_penalty,
                    )
                self._submission_report = grade_case(
                    self._scenario,
                    self._line_resolutions,
                    self._header_resolution,
                    self._notes,
                    ReviewTrace(
                        ref_steps=self._ref_steps,
                        steps_used=self._state.step_count,
                        invalid_actions=self._invalid_actions,
                        redundant_actions=self._redundant_actions,
                    ),
                )
                self._submitted = True
                self._state.submitted = True
                reward = self._submission_report.total_score
                done = True
                message = (
                    f"Case submitted with score {self._submission_report.total_score:.4f}."
                )

        if not done and self._state.step_count >= self._state.step_limit:
            self._submission_report = grade_case(
                self._scenario,
                self._line_resolutions,
                self._header_resolution,
                self._notes,
                ReviewTrace(
                    ref_steps=self._ref_steps,
                    steps_used=self._state.step_count,
                    invalid_actions=self._invalid_actions,
                    redundant_actions=self._redundant_actions,
                ),
            )
            self._submitted = True
            self._state.submitted = True
            reward = self._submission_report.total_score
            done = True
            message = (
                "Step budget exhausted. "
                f"Auto-submitted with score {self._submission_report.total_score:.4f}."
            )

        return self._build_observation(message=message, reward=reward, done=done)

    def _validate_submission_refs(
        self,
        note_ids: list[str],
        line_resolution_ids: list[str],
        header_resolution_id: str | None,
    ) -> str | None:
        if note_ids:
            missing = [note_id for note_id in note_ids if note_id not in self._notes]
            if missing:
                return f"Unknown note_ids in submit_case: {missing}"
        if line_resolution_ids:
            known = {resolution.resolution_id for resolution in self._line_resolutions.values()}
            missing = [resolution_id for resolution_id in line_resolution_ids if resolution_id not in known]
            if missing:
                return f"Unknown line_resolution_ids in submit_case: {missing}"
        if header_resolution_id is not None:
            if self._header_resolution is None or self._header_resolution.resolution_id != header_resolution_id:
                return f"Unknown header_resolution_id in submit_case: {header_resolution_id}"
        return None

    def _validate_submission_consistency(self) -> str | None:
        approved_line_ids = sorted(
            line_id
            for line_id, resolution in self._line_resolutions.items()
            if resolution.disposition is Disposition.APPROVE
        )
        escalated_without_route = sorted(
            line_id
            for line_id, resolution in self._line_resolutions.items()
            if resolution.disposition is Disposition.ESCALATE and resolution.route_to is None
        )
        if escalated_without_route:
            return f"Escalated lines require route_to: {escalated_without_route}"

        header_resolution = self._header_resolution
        if header_resolution is None:
            return None

        recommendation = header_resolution.payment_recommendation
        if (
            recommendation is PaymentRecommendation.ESCALATE_CASE
            and header_resolution.route_to is None
        ):
            return "escalate_case requires route_to."
        if (
            recommendation is PaymentRecommendation.RELEASE_APPROVED_LINES
            and not approved_line_ids
        ):
            return "release_approved_lines requires at least one approved line."
        if (
            recommendation is PaymentRecommendation.REJECT_FULL_INVOICE
            and approved_line_ids
        ):
            return (
                f"{recommendation.value} is inconsistent with approved lines: "
                f"{approved_line_ids}"
            )
        return None

    def _first_invalid_ref(self, evidence_refs: list[str]) -> str | None:
        known_refs = set(self._ref_steps)
        for ref in evidence_refs:
            if ref not in known_refs:
                return ref
        return None

    def _invalid_observation(
        self,
        message: str,
        reward: float,
        done: bool = False,
    ) -> InvoiceOpsObservation:
        self._invalid_actions += 1
        self._state.invalid_actions = self._invalid_actions
        return self._build_observation(message=message, reward=reward, done=done)

    def _build_observation(
        self,
        *,
        message: str,
        reward: float,
        done: bool,
    ) -> InvoiceOpsObservation:
        scenario = self._scenario
        if scenario is None:
            raise RuntimeError("Scenario is not loaded")

        steps_remaining = max(0, self._state.step_limit - self._state.step_count)
        progress = Progress(
            steps_used=self._state.step_count,
            steps_remaining=steps_remaining,
            opened_artifacts=len(self._opened_artifact_ids),
            inspected_exceptions=len(self._inspected_exception_ids),
            notes_count=len(self._notes),
            line_resolutions=len(self._line_resolutions),
            duplicate_checks_run=len(self._duplicate_checks_run),
            invalid_actions=self._invalid_actions,
            redundant_actions=self._redundant_actions,
            submitted=self._submitted,
        )
        opened_artifact = (
            self._artifact_map[self._current_artifact_id]
            if self._current_artifact_id is not None
            else None
        )
        inspected_exception = (
            self._exception_map[self._current_exception_id]
            if self._current_exception_id is not None
            else None
        )
        return InvoiceOpsObservation(
            message=message,
            task_id=scenario.task_id,
            scenario_id=scenario.scenario_id,
            title=scenario.title,
            description=scenario.description,
            queue_card=scenario.queue_card,
            available_artifacts=artifact_references(scenario),
            opened_artifact=opened_artifact,
            visible_exceptions=exception_summaries(scenario),
            inspected_exception=inspected_exception,
            duplicate_candidates=self._duplicate_candidates,
            draft_notes=list(self._notes.values()),
            draft_line_resolutions=list(self._line_resolutions.values()),
            draft_header_resolution=self._header_resolution,
            submission_report=self._submission_report,
            progress=progress,
            known_refs=sorted(self._ref_steps),
            episode_score=(
                self._submission_report.total_score if self._submission_report else None
            ),
            done=done,
            reward=reward,
            metadata={
                "case_id": scenario.case_id,
                "task_id": scenario.task_id.value,
            },
        )

    @property
    def state(self) -> InvoiceOpsState:
        return self._state
