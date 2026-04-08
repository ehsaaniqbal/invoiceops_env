"""Dense reward shaping for InvoiceOps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    step_cost: float = -0.01
    first_open_artifact: float = 0.02
    inspect_exception: float = 0.03
    run_duplicate_check: float = 0.03
    valid_note: float = 0.03
    valid_line_resolution: float = 0.04
    valid_header_resolution: float = 0.05
    invalid_action_penalty: float = -0.05
    redundant_open_penalty: float = -0.03
    revision_penalty: float = -0.02
    redundant_duplicate_penalty: float = -0.03


DEFAULT_REWARD_CONFIG = RewardConfig()
