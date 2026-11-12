import numpy as np
import pytest

from anomaly_lab.alerting import AlertPolicy, AlertState, apply_alert_policy


def test_debounce_suppresses_single_sample_spikes():
    policy = AlertPolicy(5.0, 2.0, consecutive_trigger=2, consecutive_recovery=2)
    report = apply_alert_policy(np.array([0, 6, 0, 7, 8, 4, 1, 1]), policy)

    assert report.active.tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert [transition.index for transition in report.transitions] == [4, 7]
    assert report.opened_alerts == 1
    assert report.alert_durations == (3,)


def test_hysteresis_prevents_flapping_between_thresholds():
    policy = AlertPolicy(5.0, 2.0)
    report = apply_alert_policy(np.array([6, 4, 3, 4.9, 2.1, 2.0]), policy)

    assert report.active.tolist() == [True, True, True, True, True, False]
    assert tuple(transition.state for transition in report.transitions) == (
        AlertState.ACTIVE,
        AlertState.INACTIVE,
    )


def test_open_alert_duration_is_closed_at_end_of_observation():
    report = apply_alert_policy(np.array([0, 6, 7]), AlertPolicy(5.0, 2.0))

    assert report.opened_alerts == 1
    assert report.active_points == 2
    assert report.alert_durations == (2,)
    assert report.to_dict()["transitions"][0]["reason"] == "trigger_streak_satisfied"


@pytest.mark.parametrize(
    "policy",
    [
        lambda: AlertPolicy(float("nan"), 1.0),
        lambda: AlertPolicy(5.0, 5.0),
        lambda: AlertPolicy(5.0, 1.0, consecutive_trigger=0),
        lambda: AlertPolicy(5.0, 1.0, consecutive_recovery=0),
    ],
)
def test_invalid_policies_fail_closed(policy):
    with pytest.raises(ValueError):
        policy()


@pytest.mark.parametrize(
    "scores",
    [np.array([]), np.array([[1.0]]), np.array([1.0, np.inf])],
)
def test_invalid_scores_fail_closed(scores):
    with pytest.raises(ValueError):
        apply_alert_policy(scores, AlertPolicy(5.0, 2.0))
