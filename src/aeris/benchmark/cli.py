"""Command-line benchmark bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .channels import diagnose_channels
from .cvb import project_cvb_behavior_table
from .io import read_behavior_table, write_projection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AERIS benchmark adapters on local research data."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cvb = sub.add_parser("cvb", help="Project a CVB behavior table into AERIS states.")
    cvb.add_argument("input", type=Path)
    cvb.add_argument("--output", type=Path, default=Path("results/cvb"))

    channels = sub.add_parser(
        "channels",
        help="Run channel diagnostics on an AERIS animal-state table.",
    )
    channels.add_argument("input", type=Path)
    channels.add_argument("--output", type=Path, default=Path("results/channels"))

    args = parser.parse_args()

    if args.command == "cvb":
        source = read_behavior_table(args.input)
        projected, summary = project_cvb_behavior_table(source)
        write_projection(projected, summary, args.output)
        print(f"CVB projection written to {args.output}")
        return

    data = pd.read_csv(args.input)
    result = diagnose_channels(data)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        if hasattr(frame, "to_csv"):
            frame.to_csv(args.output / f"{name}.csv", index=False)
    print(f"Channel diagnostics written to {args.output}")


if __name__ == "__main__":
    main()
