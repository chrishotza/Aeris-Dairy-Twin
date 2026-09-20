"""Command-line entry point for AERIS research simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .alerts import build_alert_feed
from .simulation import SimulationConfig, simulate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a reproducible AERIS multimodal research simulation."
    )
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--units", type=int, default=2)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--animals", type=int, default=6)
    parser.add_argument("--monte-carlo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--output", type=Path, default=Path("results/simulation"))
    args = parser.parse_args()

    config = SimulationConfig(
        monte_carlo=args.monte_carlo,
        n_units=args.units,
        groups_per_unit=args.groups,
        animals_per_group=args.animals,
        hours=args.hours,
        seed=args.seed,
    )
    result = simulate(config)

    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    result["animals"].to_csv(output / "animal_states.csv", index=False)
    result["groups"].to_csv(output / "group_states.csv", index=False)
    result["units"].to_csv(output / "unit_states.csv", index=False)
    result["scenario_metrics"].to_csv(output / "scenario_metrics.csv", index=False)

    alerts = build_alert_feed(result["animals"], persistence=True)
    alerts.to_csv(output / "animal_alerts.csv", index=False)

    print(f"AERIS simulation complete: {output}")
    print(f"Animals: {len(result['animals']):,}")
    print(f"Groups:  {len(result['groups']):,}")
    print(f"Units:   {len(result['units']):,}")
    print(f"Alerts:  {len(alerts):,}")


if __name__ == "__main__":
    main()
