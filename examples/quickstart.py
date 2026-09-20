from aeris.core.state_engine import (
    WelfareSignals,
    state_from_signals,
    structural_validity,
)


def main() -> None:
    # Values here are normalized research inputs, not field measurements.
    validity = structural_validity(
        directional_deterioration=0.80,
        persistence=0.75,
        cross_signal_coherence=0.90,
    )

    state = state_from_signals(
        signals=WelfareSignals(
            activity=0.42,
            rumination=0.48,
            locomotion=0.50,
            heat=0.72,
        ),
        validity=validity,
        activity_drop=0.35,
        rumination_drop=0.30,
        locomotion_drop=0.28,
        heat_rise=0.40,
    )

    print("AERIS example state")
    print("-------------------")
    for key, value in state.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
