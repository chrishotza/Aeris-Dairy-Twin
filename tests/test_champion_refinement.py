from experiments.champion_refinement import (
    DEFAULTS,
    SearchConfig,
    evaluate_params,
    run_search,
)
from experiments.reproducibility_demo import build_demo_animals


def test_champion_refinement_accepts_public_fixture():
    result = evaluate_params(build_demo_animals(), DEFAULTS.copy())

    assert "objective" in result
    assert "scenario_metrics" in result
    assert not result["scenario_metrics"].empty


def test_champion_search_is_seeded():
    frame = build_demo_animals()
    config = SearchConfig(
        seed=7,
        random_iters=2,
        top_seeds=1,
        refine_iters=1,
    )

    first = run_search(frame, config)
    second = run_search(frame, config)

    assert first["best"]["params"] == second["best"]["params"]
    assert first["best"]["metrics"] == second["best"]["metrics"]
