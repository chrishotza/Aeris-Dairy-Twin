from pathlib import Path
import re

root = Path(".")
base = root / "11_real_data" / "cvb_data"
targets = [
    base / "metadata" / "readme-help.md",
    base / "metadata" / "dublincore-000058916v001.xml",
    base / "metadata" / "dublincore-collection.xml",
]

patterns = [
    r'beh\d+',
    r'behavior',
    r'Behaviour',
    r'behaviour',
    r'feeding',
    r'drinking',
    r'walking',
    r'standing',
    r'lying',
    r'ruminating',
    r'grooming',
    r'sniffing',
]

print("\n=== SEARCH FOR BEHAVIOR MAPPING ===")

for path in targets:
    print(f"\n--- FILE: {path} ---")
    if not path.exists():
        print("missing")
        continue

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"unreadable: {e}")
        continue

    lines = text.splitlines()
    hits = 0

    for i, line in enumerate(lines):
        line_low = line.lower()
        if any(re.search(p.lower(), line_low) for p in [r'beh\d+', 'behavior', 'behaviour', 'feeding', 'drinking', 'walking', 'standing', 'lying', 'ruminating', 'grooming', 'sniffing']):
            print(f"{i+1:04d}: {line}")
            hits += 1
            if hits >= 80:
                print("...truncated...")
                break

    if hits == 0:
        print("no obvious mapping found")

