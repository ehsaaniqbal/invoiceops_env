from invoiceops_env.inference import (
    API_BASE_URL,
    ObservationMemory,
    TASKS,
    _parse_action_payload,
    _safe_json_load,
    build_action_prompt,
    build_observation_snapshot,
    resolve_api_key,
    strict_task_score,
    update_memory,
)
from invoiceops_env.models import (
    ActionType,
    Disposition,
    DuplicateMatchStrategy,
    InvoiceOpsAction,
    PaymentRecommendation,
    ReasonCode,
)
from invoiceops_env.server.invoiceops_env_environment import InvoiceOpsEnvironment


def test_parse_action_payload_salvages_common_shapes() -> None:
    payload = {
        "action": "set_line_resolution",
        "args": {
            "line": "L1",
            "decision": "hold",
            "reason_code": "receipt_not_confirmed",
            "refs": ["art-history", "EX-RECEIPT-L2"],
            "route": "receiving",
        },
    }

    action = _parse_action_payload(payload)

    assert action is not None
    assert action.action_type is ActionType.SET_LINE_RESOLUTION
    assert action.line_id == "L1"
    assert action.disposition is Disposition.HOLD
    assert action.reason_codes == [ReasonCode.RECEIPT_NOT_CONFIRMED]
    assert action.evidence_refs == ["art-history", "EX-RECEIPT-L2"]


def test_parse_action_payload_rejects_missing_required_fields() -> None:
    payload = {
        "action_type": "set_header_resolution",
        "payment_recommendation": "hold_full_invoice",
        "reason_codes": ["receipt_not_confirmed"],
    }

    assert _parse_action_payload(payload) is None


def test_parse_action_payload_accepts_submit_case() -> None:
    action = _parse_action_payload({"action_type": "submit_case"})

    assert action is not None
    assert action.action_type is ActionType.SUBMIT_CASE


def test_safe_json_load_strips_think_blocks() -> None:
    payload = _safe_json_load(
        '<think>reasoning here</think>{"action_type":"submit_case"}'
    )

    assert payload == {"action_type": "submit_case"}


def test_initial_snapshot_does_not_preload_case_details() -> None:
    env = InvoiceOpsEnvironment()
    observation = env.reset(task_id="hard")
    memory = ObservationMemory()

    snapshot = build_observation_snapshot(observation, memory)

    assert snapshot["artifacts"] == {}
    assert snapshot["exceptions"] == []
    assert snapshot["duplicate_candidates"] == []
    assert snapshot["known_refs"] == []


def test_snapshot_only_contains_explicitly_observed_details() -> None:
    env = InvoiceOpsEnvironment()
    observation = env.reset(task_id="medium")
    memory = ObservationMemory()
    update_memory(memory, observation)

    observation = env.step(
        InvoiceOpsAction(
            action_type=ActionType.OPEN_ARTIFACT,
            artifact_id="art-invoice",
        )
    )
    update_memory(memory, observation)

    first_exception = observation.visible_exceptions[0]
    observation = env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id=first_exception.exception_id,
        )
    )
    update_memory(memory, observation)

    snapshot = build_observation_snapshot(observation, memory)

    assert set(snapshot["artifacts"]) == {"invoice_packet"}
    assert [exception["type"] for exception in snapshot["exceptions"]] == [
        "possible_duplicate"
    ]
    assert observation.known_refs == ["EX-POSSIBLE-DUP", "art-invoice"]


def test_visible_exception_stubs_hide_detailed_facts_until_inspection() -> None:
    env = InvoiceOpsEnvironment()
    observation = env.reset(task_id="medium")

    stub = observation.visible_exceptions[0]
    assert stub.headline == "Potential duplicate invoice requires review"
    assert stub.short_description == (
        "Inspect this exception for duplicate-match details before deciding."
    )

    inspected = env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id=stub.exception_id,
        )
    ).inspected_exception

    assert inspected is not None
    assert inspected.headline == "Duplicate control is open for this invoice"
    assert any(
        field.label == "Invoice number" and field.value == "TL-9205/A"
        for field in inspected.fields
    )


def test_hard_exceptions_do_not_expose_derived_answer_fields() -> None:
    env = InvoiceOpsEnvironment()
    env.reset(task_id="hard")

    l1 = env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-RECEIPT-L1",
        )
    ).inspected_exception
    l2 = env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-RECEIPT-L2",
        )
    ).inspected_exception

    assert l1 is not None
    assert l2 is not None
    assert {field.label for field in l1.fields} == {
        "Invoice quantity",
        "Received quantity",
        "Short quantity",
    }
    assert {field.label for field in l2.fields} == {
        "Invoice quantity",
        "Initial posted receipt",
        "Latest control update",
    }


def test_hard_receipt_log_points_to_history_instead_of_reversal_answer() -> None:
    env = InvoiceOpsEnvironment()
    observation = env.reset(task_id="hard")
    observation = env.step(
        InvoiceOpsAction(
            action_type=ActionType.OPEN_ARTIFACT,
            artifact_id="art-receipts",
        )
    )

    opened = observation.opened_artifact
    assert opened is not None
    l2 = next(item for item in opened.line_items if item.line_id == "L2")

    assert l2.status == "received_under_review"
    assert "history" in l2.notes.lower()
    assert "reversed" not in l2.notes.lower()


def test_medium_plus_exception_does_not_expose_unsupported_amount() -> None:
    env = InvoiceOpsEnvironment()
    env.reset(task_id="medium_plus")

    inspected = env.step(
        InvoiceOpsAction(
            action_type=ActionType.INSPECT_EXCEPTION,
            exception_id="EX-RECEIPT-L2",
        )
    ).inspected_exception

    assert inspected is not None
    assert {field.label for field in inspected.fields} == {
        "Invoice quantity",
        "Received quantity",
        "Short quantity",
    }


def test_action_prompt_describes_single_step_agent_loop() -> None:
    env = InvoiceOpsEnvironment()
    observation = env.reset(task_id="medium")
    prompt = build_action_prompt(observation, ObservationMemory())

    assert "Return exactly one JSON object for the single best next action." in prompt
    assert "Use open_artifact, inspect_exception, and run_duplicate_check" in prompt
    assert "next owner or follow-up queue" in prompt
    assert "A real case-level blocker can justify hold_full_invoice" in prompt
    assert "Allowed match_strategy values" in prompt
    assert "Action JSON templates" in prompt
    assert "<artifact_id>" in prompt
    assert '"match_strategy":"normalized_invoice_no"' in prompt
    assert '"action_type":"submit_case"' in prompt
    assert "If any line is approved, use release_approved_lines" not in prompt


def test_hf_router_configuration_requires_hf_token(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf-secret")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    api_key, source = resolve_api_key()

    assert api_key == "hf-secret"
    assert source == "HF_TOKEN"
    assert API_BASE_URL == "https://router.huggingface.co/v1"


def test_invoice_baseline_defaults_to_strict_scoring(monkeypatch) -> None:
    monkeypatch.delenv("STRICT_BASELINE_SCORING", raising=False)

    assert strict_task_score(0.2136, used_fallback=True) == 0.0
    assert strict_task_score(0.2136, used_fallback=False) == 0.2136


def test_public_task_loop_uses_four_task_progression() -> None:
    assert [task.value for task in TASKS] == [
        "easy",
        "medium",
        "medium_plus",
        "hard",
    ]


def test_duplicate_check_exposes_strategy_and_candidate_refs() -> None:
    env = InvoiceOpsEnvironment()
    observation = env.reset(task_id="medium")
    memory = ObservationMemory()
    update_memory(memory, observation)

    observation = env.step(
        InvoiceOpsAction(
            action_type=ActionType.RUN_DUPLICATE_CHECK,
            match_strategy=DuplicateMatchStrategy.NORMALIZED_INVOICE_NUMBER,
        )
    )
    update_memory(memory, observation)
    snapshot = build_observation_snapshot(observation, memory)

    assert "duplicate_check:normalized_invoice_no" in observation.known_refs
    assert "CAND-NORM-01" in observation.known_refs
    assert snapshot["duplicate_candidates"] == [
        {
            "candidate_id": "CAND-NORM-01",
            "invoice_number": "TL9205A",
            "invoice_date": "2026-03-10",
            "gross_amount": 3800.0,
            "status": "reversed on 2026-03-11 after import duplicate; closed",
            "match_basis": "Normalized invoice number + vendor + gross amount",
            "overlap_summary": "Same normalized invoice number. Prior record was reversed before payment.",
        }
    ]


def test_duplicate_check_action_parses() -> None:
    action = _parse_action_payload(
        {
            "action_type": "run_duplicate_check",
            "match_strategy": DuplicateMatchStrategy.NORMALIZED_INVOICE_NUMBER.value,
        }
    )

    assert action is not None
    assert action.action_type is ActionType.RUN_DUPLICATE_CHECK
    assert (
        action.match_strategy
        is DuplicateMatchStrategy.NORMALIZED_INVOICE_NUMBER
    )


def test_duplicate_check_action_parses_common_aliases() -> None:
    action = _parse_action_payload(
        {
            "action_type": "run_duplicate_check",
            "match_strategy": "vendor_invoice_amount",
        }
    )

    assert action is not None
    assert action.action_type is ActionType.RUN_DUPLICATE_CHECK
    assert action.match_strategy is DuplicateMatchStrategy.VENDOR_AMOUNT_DATE


def test_header_resolution_action_parses_common_aliases() -> None:
    action = _parse_action_payload(
        {
            "action_type": "set_header_resolution",
            "recommendation": PaymentRecommendation.HOLD_FULL_INVOICE.value,
            "reason_code": [
                ReasonCode.NON_PO_APPROVAL_MISSING.value,
                ReasonCode.POSSIBLE_DUPLICATE_REVIEW.value,
            ],
            "refs": ["art-approval", "duplicate_check:normalized_invoice_no"],
            "route": "requester",
        }
    )

    assert action is not None
    assert action.action_type is ActionType.SET_HEADER_RESOLUTION
    assert (
        action.payment_recommendation
        is PaymentRecommendation.HOLD_FULL_INVOICE
    )
    assert action.reason_codes == [
        ReasonCode.NON_PO_APPROVAL_MISSING,
        ReasonCode.POSSIBLE_DUPLICATE_REVIEW,
    ]
    assert action.evidence_refs == [
        "art-approval",
        "duplicate_check:normalized_invoice_no",
    ]
