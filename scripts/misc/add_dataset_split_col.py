"""
Add dataset_split column to a Parquet file.

Replaces scripts/misc/add_dataset_split_col.R with pure Python.

Usage:
    python add_dataset_split_col.py input.parquet output.parquet 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def main() -> None:
    """Run the main entry point."""
    parser = argparse.ArgumentParser(
        description="Add dataset_split column to Parquet"
    )
    parser.add_argument("input_path", type=Path, help="Input Parquet path")
    parser.add_argument("output_path", type=Path, help="Output Parquet path")
    parser.add_argument("dataset_split", type=int, help="Dataset split value")

    args = parser.parse_args()

    df = pl.read_parquet(args.input_path).with_columns(
        pl.lit(args.dataset_split, dtype=pl.Int64).alias("dataset_split")
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.output_path)


if __name__ == "__main__":
    main()
