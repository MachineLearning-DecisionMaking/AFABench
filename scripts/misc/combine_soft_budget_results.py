"""
Combine multiple Parquet files with soft budget evaluation results.

Replaces scripts/misc/combine_soft_budget_results.R with pure Python.

Usage:
    python combine_soft_budget_results.py a.parquet b.parquet combined.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

REQUIRED_COLS = [
    "method",
    "training_seed",
    "cost_parameter",
    "dataset",
    "features_chosen",
    "predicted_label_builtin",
    "predicted_label_external",
    "true_label",
]


def main() -> None:
    """Run the main entry point."""
    parser = argparse.ArgumentParser(
        description="Combine soft budget result Parquet files"
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Input Parquet files (last one is output path)",
    )

    args = parser.parse_args()

    if len(args.files) < 2:
        parser.error("Need at least one input file and one output file")

    output_path = args.files[-1]
    input_paths = args.files[:-1]

    dfs = []
    for path in input_paths:
        df = pl.read_parquet(path)

        # Validate required columns
        missing = set(REQUIRED_COLS) - set(df.columns)
        if missing:
            msg = f"File {path} is missing columns: {', '.join(missing)}"
            raise ValueError(msg)

        dfs.append(df)

    combined = pl.concat(dfs, how="diagonal_relaxed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)

    print(f"Combined {len(input_paths)} files into {output_path}")


if __name__ == "__main__":
    main()
