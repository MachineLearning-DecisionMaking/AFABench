import argparse
from pathlib import Path

import polars as pl

from afabench.evaluation.provenance import save_evaluation_table


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_evaluation_table(df, path, provenance=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split eval performance results by classifier type."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_builtin", type=Path, required=True)
    parser.add_argument("--output_external", type=Path, required=True)
    args = parser.parse_args()

    df = pl.read_parquet(args.input_path)
    if "classifier" not in df.columns:
        msg = "Expected 'classifier' column in eval performance dataframe."
        raise ValueError(msg)

    builtin = df.filter(pl.col("classifier") == "builtin").drop("classifier")
    external = df.filter(pl.col("classifier") == "external").drop("classifier")

    _write_parquet(builtin, args.output_builtin)
    _write_parquet(external, args.output_external)


if __name__ == "__main__":
    main()
