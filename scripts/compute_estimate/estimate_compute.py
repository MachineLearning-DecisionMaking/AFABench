"""
Estimate the compute of a pipeline invocation (`uv run just estimate-compute`).

Takes, besides its own options, the same arguments as the pipeline launcher
(`workflow/run.py`) gets for the real run: pipeline profile, `--site`,
`--target`, pipeline profile overrides and, after `--`, Snakemake
arguments. It plans the jobs that invocation would run, matches each to job
durations and prints the compute estimate: job-hours, core-hours and
GPU-hours, mean and p90, per pipeline stage and device. It warns about
planned job types that failed or timed out before. See
`afabench.pipeline_plan`, `afabench.compute_estimate` and
`docs/how-to/maintainer/estimate_compute.md`.

Options, before or among the launcher's arguments:
    --job-durations PATH  A job duration table, or an output root of job
                          records (default: output/production; a
                          smoke test's job records are under
                          output/smoke).
    --by COLUMN           Group the totals by these job columns instead,
                          repeatable (e.g. --by name --by dataset_key).
    --output CSV          Also write each planned job's estimate here.
    --strict              Exit with 1 when a planned job is unestimated.

Usage:
    uv run just estimate-compute kdd26 --site <site>
    uv run just estimate-compute \
        --job-durations \
            output/production/release_job_duration_table.parquet \
        --by name --output estimate.csv <launcher arguments>
"""

import sys
from pathlib import Path
from typing import Annotated, override

import click
import typer
from typer.core import TyperCommand

from afabench.compute_estimate.estimate import estimate_compute
from afabench.compute_estimate.report import (
    DEFAULT_GROUPING,
    UnknownGroupingColumnError,
    check_grouping,
    format_report,
    per_job_table,
)
from afabench.core.job_duration_table import (
    empty_job_duration_table,
    load_job_duration_table,
)
from afabench.core.output_layout import PRODUCTION_OUTPUT_ROOT
from afabench.pipeline_plan.planning import InvocationError, plan_invocation
from afabench.release.manifest import (
    JOB_DURATION_TABLE_FILENAME,
    RELEASE_MANIFEST_FILENAME,
    read_release_manifest,
)

# Run as a script, this directory is on sys.path, not the checkout root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflow.run import LauncherError, snakemake_arguments

DEFAULT_JOB_DURATIONS = Path(PRODUCTION_OUTPUT_ROOT)
SNAKEMAKE_ARGUMENTS = "snakemake_arguments"

app = typer.Typer()


class _LauncherArgumentsCommand(TyperCommand):
    """Keeps the launcher's Snakemake arguments, and the -- before them."""

    @override
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if "--" in args:
            split = args.index("--")
            # Click would drop the --, leaving them to the launcher's parser.
            ctx.meta[SNAKEMAKE_ARGUMENTS] = args[split:]
            args = args[:split]
        return super().parse_args(ctx, args)


@app.command(
    cls=_LauncherArgumentsCommand,
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def main(
    ctx: typer.Context,
    job_durations: Annotated[
        Path,
        typer.Option(
            help=(
                "A job duration table, or an output root of job records, "
                "such as a downloaded release's."
            )
        ),
    ] = DEFAULT_JOB_DURATIONS,
    by: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Group the totals by this job column; repeat for several. "
                f"Default: {' and '.join(DEFAULT_GROUPING)}."
            )
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Write each planned job's estimate here as CSV."),
    ] = None,
    strict: Annotated[  # noqa: FBT002
        bool,
        typer.Option(help="Exit with 1 when a planned job is unestimated."),
    ] = False,
) -> None:
    """Estimate `uv run workflow/run.py <arguments>`; other options go to it."""
    grouping = by or DEFAULT_GROUPING
    try:
        check_grouping(grouping)
    except UnknownGroupingColumnError as error:
        raise typer.BadParameter(str(error), param_hint="--by") from None
    if job_durations.exists():
        table = load_job_duration_table(job_durations)
    elif job_durations == DEFAULT_JOB_DURATIONS:
        # Nothing has run here yet: every job is unestimated.
        table = empty_job_duration_table()
    else:
        message = f"No job duration table or output root at {job_durations}"
        raise typer.BadParameter(message, param_hint="--job-durations")
    try:
        # Profiles resolve in the working directory, as Snakemake's do.
        arguments = snakemake_arguments(
            [*ctx.args, *ctx.meta.get(SNAKEMAKE_ARGUMENTS, [])],
            checkout=Path.cwd(),
        )
    except LauncherError as error:
        raise typer.BadParameter(str(error)) from None
    try:
        plan = plan_invocation(arguments)
    except InvocationError:
        raise typer.Exit(1) from None
    estimate = estimate_compute(plan.scheduled_jobs, table)
    if output is not None:
        per_job_table(estimate).write_csv(output)
    typer.echo(
        format_report(
            estimate, source=_described_source(job_durations), by=grouping
        ),
        nl=False,
    )
    unestimated = sum(
        job.match_level == "unestimated" for job in estimate.jobs
    )
    if strict and unestimated:
        noun = "job is" if unestimated == 1 else "jobs are"
        typer.echo(
            f"--strict: {unestimated} planned {noun} unestimated", err=True
        )
        raise typer.Exit(1)


def _described_source(job_durations: Path) -> str:
    """Name the release of a restored release's job duration table."""
    manifest_path = job_durations.parent / RELEASE_MANIFEST_FILENAME
    if (
        job_durations.name != JOB_DURATION_TABLE_FILENAME
        or not manifest_path.is_file()
    ):
        return str(job_durations)
    manifest = read_release_manifest(manifest_path)
    entry = manifest.job_duration_table
    # Downloading another release with --overwrite can leave an earlier
    # release's table beside the new manifest.
    if entry is None or entry.size_bytes != job_durations.stat().st_size:
        return (
            f"{job_durations} (not the table of release "
            f"{manifest.release_id}, whose manifest is beside it)"
        )
    return f"{job_durations} (release {manifest.release_id})"


if __name__ == "__main__":
    app()
