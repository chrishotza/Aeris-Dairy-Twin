from experiments.synthetic_validation import classify_window, generate_series


def test_stable_series_is_green():
    frame = generate_series("stable")
    label, _, _ = classify_window(frame)
    assert label == "GREEN"
