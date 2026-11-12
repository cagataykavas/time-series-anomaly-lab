from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np


class AlertState(StrEnum):
    INACTIVE = "inactive"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class AlertPolicy:
    trigger_threshold: float
    recovery_threshold: float
    consecutive_trigger: int = 1
    consecutive_recovery: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.trigger_threshold) or not np.isfinite(
            self.recovery_threshold
        ):
            raise ValueError("thresholds must be finite")
        if self.recovery_threshold >= self.trigger_threshold:
            raise ValueError("recovery_threshold must be below trigger_threshold")
        if self.consecutive_trigger < 1 or self.consecutive_recovery < 1:
            raise ValueError("consecutive counts must be positive")


@dataclass(frozen=True, slots=True)
class AlertTransition:
    index: int
    state: AlertState
    score: float
    reason: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlertReport:
    active: np.ndarray
    transitions: tuple[AlertTransition, ...]
    opened_alerts: int
    active_points: int
    alert_durations: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "active": self.active.tolist(),
            "transitions": [transition.to_dict() for transition in self.transitions],
            "opened_alerts": self.opened_alerts,
            "active_points": self.active_points,
            "alert_durations": list(self.alert_durations),
        }


def apply_alert_policy(scores: np.ndarray, policy: AlertPolicy) -> AlertReport:
    """Convert anomaly scores into a debounced, hysteretic alert lifecycle."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")

    active = np.zeros(len(values), dtype=bool)
    transitions: list[AlertTransition] = []
    durations: list[int] = []
    state = AlertState.INACTIVE
    trigger_streak = 0
    recovery_streak = 0
    opened_at: int | None = None

    for index, score in enumerate(values):
        if state == AlertState.INACTIVE:
            trigger_streak = (
                trigger_streak + 1 if score >= policy.trigger_threshold else 0
            )
            if trigger_streak >= policy.consecutive_trigger:
                state = AlertState.ACTIVE
                opened_at = index
                transitions.append(
                    AlertTransition(
                        index, state, float(score), "trigger_streak_satisfied"
                    )
                )
                trigger_streak = 0
        else:
            recovery_streak = (
                recovery_streak + 1 if score <= policy.recovery_threshold else 0
            )
            if recovery_streak >= policy.consecutive_recovery:
                state = AlertState.INACTIVE
                transitions.append(
                    AlertTransition(
                        index, state, float(score), "recovery_streak_satisfied"
                    )
                )
                if opened_at is None:
                    raise RuntimeError("active alert is missing its opening index")
                durations.append(index - opened_at)
                opened_at = None
                recovery_streak = 0

        active[index] = state == AlertState.ACTIVE

    if opened_at is not None:
        durations.append(len(values) - opened_at)

    opened = sum(transition.state == AlertState.ACTIVE for transition in transitions)
    return AlertReport(
        active=active,
        transitions=tuple(transitions),
        opened_alerts=opened,
        active_points=int(np.sum(active)),
        alert_durations=tuple(durations),
    )
