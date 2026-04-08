"""Client for the InvoiceOps environment."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from invoiceops_env.models import (
    InvoiceOpsAction,
    InvoiceOpsObservation,
    InvoiceOpsState,
)


class InvoiceOpsEnv(EnvClient[InvoiceOpsAction, InvoiceOpsObservation, InvoiceOpsState]):
    """WebSocket client for persistent InvoiceOps sessions."""

    def _step_payload(self, action: InvoiceOpsAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[InvoiceOpsObservation]:
        obs_data = payload.get("observation", {})
        observation = InvoiceOpsObservation.model_validate(
            {
                **obs_data,
                "done": payload.get("done", False),
                "reward": payload.get("reward"),
            }
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> InvoiceOpsState:
        return InvoiceOpsState.model_validate(payload)
