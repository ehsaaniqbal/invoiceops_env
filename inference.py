"""Reproducible baseline for InvoiceOps."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from openai import OpenAI

from invoiceops_env import InvoiceOpsAction, InvoiceOpsEnv
from invoiceops_env.models import (
    ActionType,
    Disposition,
    DuplicateCandidate,
    DuplicateMatchStrategy,
    ExceptionDetail,
    InvoiceOpsObservation,
    NoteType,
    PaymentRecommendation,
    QueueCard,
    ReasonCode,
    RouteTarget,
    TaskId,
)

ENV_URL = os.getenv("ENV_URL", "https://ehsaaniqbal-invoiceops-env.hf.space")
DEFAULT_HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_HF_ROUTER_BASE_URL)
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-5.1")
TEMPERATURE = 0.0
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "3000"))
RETRY_MAX_TOKENS = max(MAX_TOKENS, int(os.getenv("RETRY_MAX_TOKENS", "5000")))
MAX_MODEL_ATTEMPTS = 2
BENCHMARK = "invoiceops_env"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "evals"
EVAL_RUN_NAME = os.getenv("EVAL_RUN_NAME")
TASKS = [
    TaskId.EASY,
    TaskId.MEDIUM,
    TaskId.MEDIUM_PLUS,
    TaskId.HARD,
]
HEADER_DISPOSITION_MAP: dict[Disposition, PaymentRecommendation] = {
    Disposition.APPROVE: PaymentRecommendation.RELEASE_APPROVED_LINES,
    Disposition.HOLD: PaymentRecommendation.HOLD_FULL_INVOICE,
    Disposition.REJECT: PaymentRecommendation.REJECT_FULL_INVOICE,
    Disposition.ESCALATE: PaymentRecommendation.ESCALATE_CASE,
}
ParsedModelOutput = TypeVar("ParsedModelOutput")


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off", ""}


VERBOSE_STDERR = _env_flag("INFERENCE_VERBOSE_STDERR", False)


def strict_task_score(raw_score: float, *, used_fallback: bool) -> float:
    if used_fallback and _env_flag("STRICT_BASELINE_SCORING", True):
        return 0.0
    return raw_score


@dataclass
class EpisodeTrace:
    rewards: list[float] = field(default_factory=list)
    steps_taken: int = 0


@dataclass
class ObservationMemory:
    opened_artifacts: dict[str, Any] = field(default_factory=dict)
    inspected_exceptions: dict[str, ExceptionDetail] = field(default_factory=dict)
    duplicate_candidates: list[DuplicateCandidate] = field(default_factory=list)


def resolve_api_key() -> tuple[str | None, str | None]:
    token = os.getenv("HF_TOKEN")
    return (token, "HF_TOKEN") if token else (None, None)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = slug.strip("-._")
    return slug or "run"


def build_output_path(model_name: str) -> tuple[str, Path]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = _slugify(EVAL_RUN_NAME) if EVAL_RUN_NAME else timestamp
    model_slug = _slugify(model_name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    candidate = OUTPUT_DIR / f"{run_id}__{model_slug}.json"
    suffix = 2
    while candidate.exists():
        candidate = OUTPUT_DIR / f"{run_id}__{model_slug}__{suffix}.json"
        suffix += 1
    return run_id, candidate


def _sanitize_log_value(value: str | None) -> str:
    if not value:
        return "null"
    return value.replace("\n", " ").strip() or "null"


def format_action_for_log(action: InvoiceOpsAction) -> str:
    return json.dumps(
        action.model_dump(mode="json", exclude_none=True),
        separators=(",", ":"),
        sort_keys=True,
    )


def _extract_step_error(
    observation: InvoiceOpsObservation | None,
    *,
    previous_invalid_actions: int,
) -> str | None:
    if observation is None:
        return None
    if observation.progress.invalid_actions > previous_invalid_actions:
        return observation.message or None
    return None


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: str | None
) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={_sanitize_log_value(error)}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _safe_json_load(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(
        r"<reasoning>.*?</reasoning>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    return payload if isinstance(payload, dict) else None


def _normalize_completion_content(raw_content: Any) -> str:
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(raw_content)


def _attempt_trace(
    *,
    completion: Any | None = None,
    content: str = "",
    payload: dict[str, Any] | None = None,
    parsed_ok: bool = False,
    failure_reason: str | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "content": content,
        "content_empty": not bool(content.strip()),
        "json_detected": payload is not None,
        "validation_passed": parsed_ok,
        "failure_reason": failure_reason,
    }

    if error is not None:
        trace["error_type"] = error.__class__.__name__
        trace["error_message"] = str(error)

    if completion is None:
        return trace

    trace["response_id"] = getattr(completion, "id", None)
    choices = getattr(completion, "choices", None) or []
    if choices:
        choice = choices[0]
        trace["finish_reason"] = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        if message is not None:
            if hasattr(message, "model_dump"):
                trace["raw_message"] = message.model_dump(
                    mode="json", exclude_none=True
                )
            else:
                trace["raw_message"] = str(message)

    usage = getattr(completion, "usage", None)
    if usage is not None and hasattr(usage, "model_dump"):
        trace["usage"] = usage.model_dump(mode="json", exclude_none=True)

    return trace


def _query_model_json(
    openai_client: OpenAI,
    *,
    system_prompt: str,
    user_prompt: str,
    validator: Callable[[dict[str, Any] | None], ParsedModelOutput | None],
    retry_feedback: str,
) -> tuple[ParsedModelOutput | None, list[dict[str, Any]]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    attempts: list[dict[str, Any]] = []

    for attempt in range(MAX_MODEL_ATTEMPTS):
        expand_token_budget = bool(
            attempts and attempts[-1].get("finish_reason") == "length"
        )
        try:
            completion = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                max_tokens=(RETRY_MAX_TOKENS if expand_token_budget else MAX_TOKENS),
            )
        except Exception as exc:
            attempts.append(
                _attempt_trace(
                    failure_reason="request_error",
                    error=exc,
                )
            )
            if attempt == MAX_MODEL_ATTEMPTS - 1:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous request failed before a usable response was returned. "
                        f"{retry_feedback} Reply with JSON only and no prose."
                    ),
                }
            )
            continue

        choices = getattr(completion, "choices", None) or []
        if not choices:
            attempts.append(
                _attempt_trace(
                    completion=completion,
                    failure_reason="no_choices",
                )
            )
            if attempt == MAX_MODEL_ATTEMPTS - 1:
                break
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous reply did not contain any choices. "
                        f"{retry_feedback} Reply with JSON only and no prose."
                    ),
                }
            )
            continue

        message = choices[0].message
        content = _normalize_completion_content(getattr(message, "content", None))
        payload = _safe_json_load(content)
        parsed = validator(payload)
        if parsed is not None:
            attempts.append(
                _attempt_trace(
                    completion=completion,
                    content=content,
                    payload=payload,
                    parsed_ok=True,
                )
            )
            return parsed, attempts

        if not content.strip():
            failure_reason = "empty_content"
        elif payload is None:
            failure_reason = "json_not_found"
        else:
            failure_reason = "schema_validation_failed"

        attempts.append(
            _attempt_trace(
                completion=completion,
                content=content,
                payload=payload,
                parsed_ok=False,
                failure_reason=failure_reason,
            )
        )

        if attempt == MAX_MODEL_ATTEMPTS - 1:
            break

        messages.extend(
            [
                {"role": "assistant", "content": content or "<empty_response>"},
                {
                    "role": "user",
                    "content": (
                        "Your previous reply could not be used. "
                        f"{retry_feedback} Reply with JSON only and no prose."
                    ),
                },
            ]
        )

    return None, attempts


def _coerce_reason_codes(values: Any) -> list[ReasonCode]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        return []

    codes: list[ReasonCode] = []
    for value in raw_values:
        if not isinstance(value, str):
            continue
        try:
            code = ReasonCode(value)
        except ValueError:
            continue
        if code not in codes:
            codes.append(code)
    return codes


def _coerce_string_list(values: Any) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        return []

    refs: list[str] = []
    for value in raw_values:
        if not isinstance(value, str):
            continue
        ref = value.strip()
        if not ref or ref in refs:
            continue
        refs.append(ref)
    return refs


def _coerce_action_type(value: Any) -> ActionType | None:
    if not isinstance(value, str):
        return None
    try:
        return ActionType(value)
    except ValueError:
        return None


def _coerce_match_strategy(value: Any) -> DuplicateMatchStrategy | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    aliases = {
        "exact_invoice_no": DuplicateMatchStrategy.EXACT_INVOICE_NUMBER,
        "exact_invoice_number": DuplicateMatchStrategy.EXACT_INVOICE_NUMBER,
        "invoice_number_exact": DuplicateMatchStrategy.EXACT_INVOICE_NUMBER,
        "normalized_invoice_no": DuplicateMatchStrategy.NORMALIZED_INVOICE_NUMBER,
        "normalized_invoice_number": DuplicateMatchStrategy.NORMALIZED_INVOICE_NUMBER,
        "normalized_invoice": DuplicateMatchStrategy.NORMALIZED_INVOICE_NUMBER,
        "vendor_amount_date": DuplicateMatchStrategy.VENDOR_AMOUNT_DATE,
        "vendor_amount": DuplicateMatchStrategy.VENDOR_AMOUNT_DATE,
        "vendor_invoice_amount": DuplicateMatchStrategy.VENDOR_AMOUNT_DATE,
        "exact_vendor_invoice_amount": DuplicateMatchStrategy.VENDOR_AMOUNT_DATE,
        "vendor_amount_and_date": DuplicateMatchStrategy.VENDOR_AMOUNT_DATE,
    }
    strategy = aliases.get(normalized)
    if strategy is not None:
        return strategy
    try:
        return DuplicateMatchStrategy(value)
    except ValueError:
        return None


def _coerce_note_type(value: Any) -> NoteType | None:
    if not isinstance(value, str):
        return None
    try:
        return NoteType(value)
    except ValueError:
        return None


def _coerce_route(value: Any) -> RouteTarget | None:
    if not isinstance(value, str):
        return None
    try:
        return RouteTarget(value)
    except ValueError:
        return None


def _coerce_disposition(value: Any) -> Disposition | None:
    if not isinstance(value, str):
        return None
    try:
        return Disposition(value)
    except ValueError:
        return None


def _coerce_payment_recommendation(
    raw_header: dict[str, Any] | str | None,
) -> PaymentRecommendation | None:
    if isinstance(raw_header, str):
        try:
            return PaymentRecommendation(raw_header)
        except ValueError:
            return None

    if not isinstance(raw_header, dict):
        return None

    for key in ("payment_recommendation", "header_recommendation", "recommendation"):
        raw_value = raw_header.get(key)
        if not isinstance(raw_value, str):
            continue
        try:
            return PaymentRecommendation(raw_value)
        except ValueError:
            continue

    disposition = _coerce_disposition(
        raw_header.get("disposition") or raw_header.get("decision")
    )
    if disposition is None:
        return None
    return HEADER_DISPOSITION_MAP.get(disposition)


def _extract_action_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None

    if isinstance(payload.get("action"), dict):
        raw_action = dict(payload["action"])
        if "action_type" not in raw_action and isinstance(
            payload.get("action_type"), str
        ):
            raw_action["action_type"] = payload["action_type"]
        return raw_action

    if isinstance(payload.get("args"), dict) and isinstance(payload.get("action"), str):
        raw_action = dict(payload["args"])
        raw_action.setdefault("action_type", payload["action"])
        return raw_action

    if isinstance(payload.get("arguments"), dict) and isinstance(
        payload.get("action"), str
    ):
        raw_action = dict(payload["arguments"])
        raw_action.setdefault("action_type", payload["action"])
        return raw_action

    return dict(payload)


def _parse_action_payload(payload: dict[str, Any] | None) -> InvoiceOpsAction | None:
    raw_action = _extract_action_payload(payload)
    if raw_action is None:
        return None

    action_type = _coerce_action_type(
        raw_action.get("action_type")
        or raw_action.get("action")
        or raw_action.get("type")
        or raw_action.get("kind")
        or raw_action.get("name")
    )
    if action_type is None:
        return None

    action_kwargs: dict[str, Any] = {
        "action_type": action_type,
    }

    if action_type is ActionType.OPEN_ARTIFACT:
        action_kwargs["artifact_id"] = (
            raw_action.get("artifact_id")
            or raw_action.get("artifact")
            or raw_action.get("id")
        )
    elif action_type is ActionType.INSPECT_EXCEPTION:
        action_kwargs["exception_id"] = (
            raw_action.get("exception_id")
            or raw_action.get("exception")
            or raw_action.get("id")
        )
    elif action_type is ActionType.RUN_DUPLICATE_CHECK:
        match_strategy = raw_action.get("match_strategy") or raw_action.get("strategy")
        action_kwargs["match_strategy"] = _coerce_match_strategy(match_strategy)
        if action_kwargs["match_strategy"] is None:
            return None
    elif action_type is ActionType.ADD_NOTE:
        action_kwargs["note_type"] = _coerce_note_type(
            raw_action.get("note_type") or raw_action.get("note_kind")
        )
        action_kwargs["reason_codes"] = _coerce_reason_codes(
            raw_action.get("reason_codes") or raw_action.get("reason_code")
        )
        action_kwargs["evidence_refs"] = _coerce_string_list(
            raw_action.get("evidence_refs")
            or raw_action.get("evidence_ref")
            or raw_action.get("refs")
        )
        action_kwargs["text"] = raw_action.get("text")
    elif action_type is ActionType.SET_LINE_RESOLUTION:
        action_kwargs["line_id"] = raw_action.get("line_id") or raw_action.get("line")
        action_kwargs["disposition"] = _coerce_disposition(
            raw_action.get("disposition") or raw_action.get("decision")
        )
        action_kwargs["reason_codes"] = _coerce_reason_codes(
            raw_action.get("reason_codes") or raw_action.get("reason_code")
        )
        action_kwargs["evidence_refs"] = _coerce_string_list(
            raw_action.get("evidence_refs")
            or raw_action.get("evidence_ref")
            or raw_action.get("refs")
        )
        action_kwargs["route_to"] = _coerce_route(
            raw_action.get("route_to")
            or raw_action.get("route")
            or raw_action.get("escalation_target")
        )
    elif action_type is ActionType.SET_HEADER_RESOLUTION:
        action_kwargs["payment_recommendation"] = _coerce_payment_recommendation(
            raw_action
        )
        action_kwargs["reason_codes"] = _coerce_reason_codes(
            raw_action.get("reason_codes") or raw_action.get("reason_code")
        )
        action_kwargs["evidence_refs"] = _coerce_string_list(
            raw_action.get("evidence_refs")
            or raw_action.get("evidence_ref")
            or raw_action.get("refs")
        )
        action_kwargs["route_to"] = _coerce_route(
            raw_action.get("route_to")
            or raw_action.get("route")
            or raw_action.get("escalation_target")
        )
    elif action_type is ActionType.SUBMIT_CASE:
        action_kwargs["note_ids"] = _coerce_string_list(raw_action.get("note_ids"))
        action_kwargs["line_resolution_ids"] = _coerce_string_list(
            raw_action.get("line_resolution_ids")
        )
        header_resolution_id = raw_action.get("header_resolution_id")
        if isinstance(header_resolution_id, str):
            action_kwargs["header_resolution_id"] = header_resolution_id.strip()

    try:
        return InvoiceOpsAction(**action_kwargs)
    except Exception:
        return None


def build_case_snapshot(
    queue_card: QueueCard,
    opened_artifacts: dict[str, Any],
    inspected_exceptions: dict[str, ExceptionDetail],
    duplicate_candidates: list[DuplicateCandidate],
) -> dict[str, Any]:
    def compact_text(value: str, *, limit: int = 180) -> str:
        normalized = re.sub(r"\s+", " ", value.strip())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3].rstrip()}..."

    def compact_fields(fields: list[Any], *, limit: int = 10) -> dict[str, str]:
        compact: dict[str, str] = {}
        for field in fields[:limit]:
            label = field.label.strip()
            value = field.value.strip()
            if not label or not value:
                continue
            compact[label] = compact_text(value, limit=120)
        return compact

    def compact_line_items(
        line_items: list[Any], *, limit: int = 6
    ) -> list[dict[str, Any]]:
        compact_items: list[dict[str, Any]] = []
        for item in line_items[:limit]:
            compact_item: dict[str, Any] = {
                "line_id": item.line_id,
                "description": compact_text(item.description, limit=100),
                "amount": item.amount,
            }
            if item.quantity is not None:
                compact_item["quantity"] = item.quantity
            if item.unit_price is not None:
                compact_item["unit_price"] = item.unit_price
            if item.status:
                compact_item["status"] = compact_text(item.status, limit=60)
            if item.notes:
                compact_item["notes"] = compact_text(item.notes, limit=100)
            compact_items.append(compact_item)
        return compact_items

    def compact_events(events: list[Any], *, limit: int = 8) -> list[dict[str, Any]]:
        compact_events_list: list[dict[str, Any]] = []
        for event in events[:limit]:
            compact_event: dict[str, Any] = {
                "type": event.event_type,
                "date": event.event_date,
                "description": compact_text(event.description, limit=120),
            }
            if event.quantity is not None:
                compact_event["quantity"] = event.quantity
            if event.amount is not None:
                compact_event["amount"] = event.amount
            if event.status:
                compact_event["status"] = compact_text(event.status, limit=60)
            compact_events_list.append(compact_event)
        return compact_events_list

    def compact_artifact(artifact: Any) -> dict[str, Any]:
        compact_artifact_view: dict[str, Any] = {
            "title": artifact.title,
        }
        if artifact.summary:
            compact_artifact_view["summary"] = compact_text(artifact.summary)
        fields = compact_fields(artifact.fields)
        if fields:
            compact_artifact_view["fields"] = fields
        line_items = compact_line_items(artifact.line_items)
        if line_items:
            compact_artifact_view["line_items"] = line_items
        events = compact_events(artifact.events)
        if events:
            compact_artifact_view["events"] = events
        return compact_artifact_view

    def compact_exception(exception: ExceptionDetail) -> dict[str, Any]:
        compact_exception_view: dict[str, Any] = {
            "type": exception.exception_type.value,
            "severity": exception.severity.value,
            "headline": compact_text(exception.headline, limit=120),
        }
        if exception.impacted_line_ids:
            compact_exception_view["impacted_line_ids"] = exception.impacted_line_ids
        if exception.short_description:
            compact_exception_view["summary"] = compact_text(
                exception.short_description,
                limit=140,
            )
        fields = compact_fields(exception.fields, limit=8)
        if fields:
            compact_exception_view["facts"] = fields
        if exception.reviewer_guidance:
            compact_exception_view["guidance"] = compact_text(
                exception.reviewer_guidance,
                limit=160,
            )
        return compact_exception_view

    def compact_duplicate(candidate: DuplicateCandidate) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "invoice_number": candidate.invoice_number,
            "invoice_date": candidate.invoice_date,
            "gross_amount": candidate.gross_amount,
            "status": candidate.status,
            "match_basis": compact_text(candidate.match_basis, limit=80),
            "overlap_summary": compact_text(candidate.overlap_summary, limit=140),
        }

    return {
        "queue_card": {
            "vendor_name": queue_card.vendor_name,
            "vendor_id": queue_card.vendor_id,
            "invoice_number": queue_card.invoice_number,
            "invoice_date": queue_card.invoice_date,
            "invoice_total": queue_card.invoice_total,
            "currency": queue_card.currency,
            "po_number": queue_card.po_number,
            "risk_flags": [flag.value for flag in queue_card.risk_flags],
            "summary": compact_text(queue_card.summary, limit=160),
        },
        "artifacts": {
            artifact.artifact_type.value: compact_artifact(artifact)
            for artifact in opened_artifacts.values()
        },
        "exceptions": [
            compact_exception(exception) for exception in inspected_exceptions.values()
        ],
        "duplicate_candidates": [
            compact_duplicate(candidate) for candidate in duplicate_candidates
        ],
    }


def update_memory(
    memory: ObservationMemory,
    observation: InvoiceOpsObservation,
) -> None:
    if observation.opened_artifact is not None:
        memory.opened_artifacts[observation.opened_artifact.artifact_id] = (
            observation.opened_artifact
        )
    if observation.inspected_exception is not None:
        memory.inspected_exceptions[observation.inspected_exception.exception_id] = (
            observation.inspected_exception
        )
    if observation.duplicate_candidates:
        memory.duplicate_candidates = observation.duplicate_candidates


def build_observation_snapshot(
    observation: InvoiceOpsObservation,
    memory: ObservationMemory,
) -> dict[str, Any]:
    queue_card = observation.queue_card
    assert queue_card is not None

    base_snapshot = build_case_snapshot(
        queue_card,
        memory.opened_artifacts,
        memory.inspected_exceptions,
        memory.duplicate_candidates,
    )
    base_snapshot["message"] = observation.message
    base_snapshot["progress"] = observation.progress.model_dump(mode="json")
    base_snapshot["known_refs"] = observation.known_refs
    base_snapshot["available_artifacts"] = [
        artifact.model_dump(mode="json") for artifact in observation.available_artifacts
    ]
    base_snapshot["visible_exceptions"] = [
        exception.model_dump(mode="json")
        for exception in observation.visible_exceptions
    ]
    base_snapshot["current_focus"] = {
        "opened_artifact_id": (
            observation.opened_artifact.artifact_id
            if observation.opened_artifact is not None
            else None
        ),
        "inspected_exception_id": (
            observation.inspected_exception.exception_id
            if observation.inspected_exception is not None
            else None
        ),
    }
    base_snapshot["draft_state"] = {
        "line_resolutions": [
            line_resolution.model_dump(mode="json")
            for line_resolution in observation.draft_line_resolutions
        ],
        "header_resolution": (
            observation.draft_header_resolution.model_dump(mode="json")
            if observation.draft_header_resolution is not None
            else None
        ),
        "notes": [note.model_dump(mode="json") for note in observation.draft_notes],
    }
    return base_snapshot


def build_action_prompt(
    observation: InvoiceOpsObservation,
    memory: ObservationMemory,
) -> str:
    snapshot = build_observation_snapshot(observation, memory)
    return (
        "You are controlling an AP invoice exception environment one action at a time.\n"
        "Return exactly one JSON object for the single best next action. No prose. No markdown. No multi-action plans.\n"
        "Do not assume you have seen artifacts or exception details that are not in the observation snapshot.\n"
        "Use open_artifact, inspect_exception, and run_duplicate_check to gather evidence before deciding.\n"
        "Only use evidence_refs from known_refs. Invalid refs will be penalized.\n"
        "Only add notes or resolutions when you have enough visible evidence to support them.\n"
        "route_to means the next owner or follow-up queue for the action. Use it whenever another queue must act, including hold actions that still need follow-up.\n"
        "Line resolutions describe content/payment readiness for each line. Header resolution describes whether any payment can be released now.\n"
        "A real case-level blocker can justify hold_full_invoice or escalate_case even when some lines are approved.\n"
        "Submit only when the current draft state is coherent or when no better action remains.\n\n"
        f"Allowed action_type values: {[action.value for action in ActionType]}\n"
        f"Allowed match_strategy values: {[strategy.value for strategy in DuplicateMatchStrategy]}\n"
        f"Allowed disposition values: {[disposition.value for disposition in Disposition]}\n"
        f"Allowed payment_recommendation values: {[recommendation.value for recommendation in PaymentRecommendation]}\n"
        f"Allowed route_to values: {[route.value for route in RouteTarget]}\n"
        f"Allowed note_type values: {[note_type.value for note_type in NoteType]}\n"
        f"Allowed reason_codes values: {[reason.value for reason in ReasonCode]}\n"
        "Action JSON templates (replace angle-bracket placeholders with real values from the observation; omit optional fields when unused):\n"
        '{"action_type":"open_artifact","artifact_id":"<artifact_id>"}\n'
        '{"action_type":"inspect_exception","exception_id":"<exception_id>"}\n'
        '{"action_type":"run_duplicate_check","match_strategy":"normalized_invoice_no"}\n'
        '{"action_type":"set_line_resolution","line_id":"<line_id>","disposition":"<disposition>","reason_codes":["<reason_code>"],"evidence_refs":["<known_ref>"],"route_to":"<optional_route_target>"}\n'
        '{"action_type":"set_header_resolution","payment_recommendation":"<payment_recommendation>","reason_codes":["<reason_code>"],"evidence_refs":["<known_ref>"],"route_to":"<optional_route_target>"}\n'
        '{"action_type":"add_note","note_type":"<note_type>","reason_codes":["<reason_code>"],"evidence_refs":["<known_ref>"],"text":"<brief_handoff_note>"}\n'
        '{"action_type":"submit_case"}\n\n'
        f"Observation snapshot:\n{json.dumps(snapshot, indent=2)}"
    )


def request_action_from_model(
    openai_client: OpenAI,
    *,
    observation: InvoiceOpsObservation,
    memory: ObservationMemory,
) -> tuple[InvoiceOpsAction | None, list[dict[str, Any]]]:
    return _query_model_json(
        openai_client,
        system_prompt=(
            "You are a deterministic AP invoice reviewer acting in an environment. "
            "Return exactly one valid JSON action and nothing else."
        ),
        user_prompt=build_action_prompt(observation, memory),
        validator=_parse_action_payload,
        retry_feedback=(
            "Return exactly one action object with action_type and only the fields required for that action. "
            'Examples: {"action_type":"open_artifact","artifact_id":"art-invoice"} '
            'or {"action_type":"submit_case"}. '
            "Do not output a plan or multiple actions."
        ),
    )


def run_task(
    env: Any,
    openai_client: OpenAI,
    task_id: TaskId,
    trace: EpisodeTrace,
) -> dict[str, Any]:
    try:
        reset_result = env.reset(task_id=task_id.value)
        observation = reset_result.observation
        initial_queue_card = observation.queue_card
        memory = ObservationMemory()
        update_memory(memory, observation)

        model_attempts: list[dict[str, Any]] = []
        action_history: list[dict[str, Any]] = []
        used_fallback = False
        decision_parsed = True
        failure_reason: str | None = None

        while not observation.done:
            action, attempts = request_action_from_model(
                openai_client,
                observation=observation,
                memory=memory,
            )
            model_attempts.append(
                {
                    "turn_index": len(model_attempts) + 1,
                    "attempts": attempts,
                }
            )

            if action is None:
                used_fallback = True
                decision_parsed = False
                failure_reason = (
                    attempts[-1]["failure_reason"] if attempts else "no_attempt"
                )
                action = InvoiceOpsAction(action_type=ActionType.SUBMIT_CASE)
                model_attempts[-1]["fallback_action"] = action.model_dump(
                    mode="json",
                    exclude_none=True,
                )

            previous_invalid_actions = observation.progress.invalid_actions
            result = env.step(action)
            reward = float(result.reward or 0.0)
            trace.steps_taken += 1
            trace.rewards.append(reward)
            log_step(
                trace.steps_taken,
                format_action_for_log(action),
                reward,
                bool(result.done),
                _extract_step_error(
                    result.observation,
                    previous_invalid_actions=previous_invalid_actions,
                ),
            )
            action_history.append(
                {
                    "step": trace.steps_taken,
                    "action": action.model_dump(mode="json", exclude_none=True),
                    "reward": reward,
                    "done": bool(result.done),
                    "message": result.observation.message,
                }
            )
            observation = result.observation
            update_memory(memory, observation)

        raw_score = float(observation.episode_score or 0.0)
        score = strict_task_score(raw_score, used_fallback=used_fallback)
        return {
            "task_id": task_id.value,
            "queue_card": (
                initial_queue_card.model_dump(mode="json")
                if initial_queue_card is not None
                else None
            ),
            "decision_parsed": decision_parsed,
            "used_fallback": used_fallback,
            "failure_reason": failure_reason,
            "parsed_line_count": len(observation.draft_line_resolutions),
            "parsed_header_resolution": observation.draft_header_resolution is not None,
            "model_attempts": model_attempts,
            "action_history": action_history,
            "raw_score": raw_score,
            "score": score,
            "steps_used": trace.steps_taken,
            "reward_trace": trace.rewards,
            "submission_report": (
                observation.submission_report.model_dump(mode="json")
                if observation.submission_report is not None
                else None
            ),
            "error": None,
        }
    except Exception as exc:
        return {
            "task_id": task_id.value,
            "queue_card": None,
            "decision_parsed": False,
            "used_fallback": False,
            "failure_reason": "task_execution_error",
            "parsed_line_count": 0,
            "parsed_header_resolution": False,
            "model_attempts": [],
            "action_history": [],
            "raw_score": 0.0,
            "score": 0.0,
            "steps_used": trace.steps_taken,
            "reward_trace": trace.rewards,
            "submission_report": None,
            "error": str(exc),
        }


def main() -> None:
    api_key, api_key_source = resolve_api_key()
    api_base_url = API_BASE_URL

    if not api_key:
        raise RuntimeError("Set HF_TOKEN before running inference.py.")

    openai_client = OpenAI(api_key=api_key, base_url=api_base_url)

    run_id, output_path = build_output_path(MODEL_NAME)
    results: list[dict[str, Any]] = []

    for task_id in TASKS:
        trace = EpisodeTrace()
        log_start(task=task_id.value, env=BENCHMARK, model=MODEL_NAME)
        task_result: dict[str, Any] | None = None
        try:
            with InvoiceOpsEnv(base_url=ENV_URL).sync() as env:
                task_result = run_task(env, openai_client, task_id, trace)
        finally:
            score = float(task_result["score"]) if task_result is not None else 0.0
            success = task_result is not None and task_result.get("error") is None
            log_end(
                success=success,
                steps=trace.steps_taken,
                score=score,
                rewards=trace.rewards,
            )

        assert task_result is not None
        results.append(task_result)
        if VERBOSE_STDERR:
            sys.stderr.write(
                f"{task_id.value}: score={task_result['score']:.4f} "
                f"raw_score={task_result.get('raw_score', task_result['score']):.4f} "
                f"fallback={str(task_result['used_fallback']).lower()}\n"
            )

    mean_score = sum(result["score"] for result in results) / len(results)
    raw_mean_score = sum(
        result.get("raw_score", result["score"]) for result in results
    ) / len(results)
    payload = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "env_url": ENV_URL,
        "api_base_url": api_base_url,
        "api_key_source": api_key_source,
        "raw_mean_score": round(raw_mean_score, 4),
        "mean_score": round(mean_score, 4),
        "strict_baseline_scoring": _env_flag("STRICT_BASELINE_SCORING", True),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if VERBOSE_STDERR:
        sys.stderr.write(
            f"mean_score={mean_score:.4f} raw_mean_score={raw_mean_score:.4f}\n"
        )
        sys.stderr.write(f"wrote={output_path}\n")


if __name__ == "__main__":
    main()
