from experiments.reproducibility_demo import build_demo_package


def test_demo_fixture_shape_and_recovery_path():
    package = build_demo_package()

    assert len(package["animal_states"]) == 24
    assert len(package["group_states"]) == 12
    assert len(package["unit_states"]) == 12

    unit1 = (
        package["unit_states"]
        .loc[lambda frame: frame["unit_id"] == "Unit_1"]
        .sort_values("hour")
    )

    assert unit1["unit_severity"].tolist() == [
        "GREEN",
        "GREEN",
        "YELLOW",
        "RED",
        "YELLOW",
        "GREEN",
    ]

    assert len(package["alerts"]) == 8
