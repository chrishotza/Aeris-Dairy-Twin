import hashlib
import json
from pathlib import Path

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


def test_checked_in_demo_manifest_matches_csv_hashes():
    root = Path(__file__).parents[1] / "data" / "synthetic" / "demo"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    for name, metadata in manifest["files"].items():
        path = root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == metadata["sha256"]
        assert path.stat().st_size == metadata["bytes"]
