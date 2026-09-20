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
