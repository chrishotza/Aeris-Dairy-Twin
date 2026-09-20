from aeris.simulation.simulator import SimulationConfig, simulate


def test_simulator_returns_three_hierarchical_tables():
    result = simulate(
        SimulationConfig(
            monte_carlo=1,
            n_units=1,
            groups_per_unit=1,
            animals_per_group=3,
            hours=6,
            seed=42,
        ),
        scenarios=("stable_baseline", "heat_stress_wave"),
    )

    assert not result["animals"].empty
    assert not result["groups"].empty
    assert not result["units"].empty
    assert set(result["scenario_metrics"]["scenario"]) == {
        "stable_baseline",
        "heat_stress_wave",
    }


def test_simulator_has_animal_group_unit_columns():
    result = simulate(
        SimulationConfig(
            monte_carlo=1,
            n_units=1,
            groups_per_unit=1,
            animals_per_group=3,
            hours=4,
            seed=42,
        ),
        scenarios=("stable_baseline",),
    )

    assert {"animal_score", "animal_severity"} <= set(result["animals"].columns)
    assert {"group_score", "group_severity"} <= set(result["groups"].columns)
    assert {"unit_score", "unit_severity"} <= set(result["units"].columns)
