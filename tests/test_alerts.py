import pandas as pd

from aeris.alerts import build_alert_feed


def test_alert_feed_detects_escalation():
    states = pd.DataFrame(
        [
            {
                "run_id": 0,
                "scenario": "test",
                "hour": 0,
                "animal_id": "A1",
                "animal_severity": "GREEN",
                "animal_regime": "stable",
                "animal_action": "standard_monitoring",
            },
            {
                "run_id": 0,
                "scenario": "test",
                "hour": 1,
                "animal_id": "A1",
                "animal_severity": "YELLOW",
                "animal_regime": "transition",
                "animal_action": "next_routine_inspection",
            },
            {
                "run_id": 0,
                "scenario": "test",
                "hour": 2,
                "animal_id": "A1",
                "animal_severity": "RED",
                "animal_regime": "collapse-risk",
                "animal_action": "on_site_inspection",
            },
        ]
    )

    alerts = build_alert_feed(states)
    assert alerts["severity"].tolist() == ["YELLOW", "RED"]


def test_alert_feed_keeps_simulations_and_scenarios_independent():
    # Identical entity IDs in independent scenarios must still raise alerts.
    rows = [
        {
            "run_id": run_id,
            "scenario": scenario,
            "hour": 0,
            "animal_id": "A1",
            "animal_severity": "RED",
            "animal_regime": "collapse-risk",
            "animal_action": "on_site_inspection",
        }
        for run_id, scenario in [(0, "baseline"), (0, "heat"), (1, "baseline")]
    ]
    alerts = build_alert_feed(pd.DataFrame(rows))
    assert len(alerts) == 3
    assert set(zip(alerts["run_id"], alerts["scenario"])) == {
        (0, "baseline"), (0, "heat"), (1, "baseline")
    }


def test_persistence_does_not_combine_short_runs_from_different_simulations():
    rows = [
        {
            "run_id": run_id,
            "scenario": "baseline",
            "hour": 0,
            "animal_id": "A1",
            "animal_severity": "RED",
            "animal_regime": "collapse-risk",
            "animal_action": "on_site_inspection",
        }
        for run_id in (0, 1)
    ]
    alerts = build_alert_feed(
        pd.DataFrame(rows), persistence=True, min_red_duration=2
    )
    # A one-frame RED is downgraded to WATCH by the existing persistence
    # contract; it must not be combined with the next simulation's RED.
    assert alerts["severity"].tolist() == ["WATCH", "WATCH"]
    assert set(alerts["run_id"]) == {0, 1}
