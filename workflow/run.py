"""
Pipeline launcher: run the pipeline from its profiles and overrides.

A pipeline profile, an optional site profile and pipeline profile overrides:

    uv run workflow/run.py [<pipeline> | --configfile <path>] [--site <site>]
        [-n | -e | -i [--group-by <dimensions>]]
        [--[no-]soft-budget] [--[no-]per-dataset-plots]
        [--[no-]wandb] [--wandb-group <name>]
        [--target <rule>] [--jobs <n>]
        [key=value ...] [-- <Snakemake arguments>]

It builds the one Snakemake command for them, adding what follows from the
site profile: its site file, which Snakemake would otherwise drop from the
site profile whenever overrides are passed, and, for execution on a site
whose allocations name venvs, workflow/venvs/bin/python, the venv of the
login node's architecture. Execution is refused while any site venv is
missing or built from another uv.lock. Planning uses the checkout environment
and only warns about site venv readiness. See docs/adr/0009 and
docs/adr/0015. --[no-]soft-budget, --[no-]per-dataset-plots and
--[no-]wandb stand for the overrides soft_budget_setting, per_dataset_plots
and use_wandb, true or false; without them the pipeline profile decides.
--wandb-group <name> stands for use_wandb=true wandb_group=<name>. With
W&B logging and no group, the launcher names a new W&B group for the
invocation, from the pipeline profile, the UTC time and random characters.

With -e it runs the compute estimate of that invocation instead
(scripts/compute_estimate/estimate_compute.py), which takes the same
arguments and, with -e only, the estimate's options, in the checkout
environment. With -i (--interactive-preview) it opens the interactive preview
of that invocation instead (python -m afabench.pipeline_preview, with the
resolved Snakemake arguments after --), in the checkout environment; the preview
needs a terminal and runs no job, so -n changes nothing and -e, --wandb
and --wandb-group are refused. --group-by, with --interactive-preview
only, passes the preview its initial grouping: stage, method and dataset,
or some of them, comma-separated in order.

It runs in the project's environment (uv run), started from the checkout.
"""

import os
import random
import shlex
import string
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, override

import click
import typer
import yaml
from typer.core import TyperCommand

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run as a script, this directory is on sys.path, not the checkout root.
sys.path.insert(0, str(REPO_ROOT))
from workflow.src.site_file import read_site_file  # noqa: E402
from workflow.venvs.submit_builds import (  # noqa: E402
    site_venvs,
    venv_problem,
)

PROG = "uv run workflow/run.py"
PIPELINE_PROFILES = "workflow/profiles/pipeline"
SITE_PROFILES = "workflow/profiles/site"
ESTIMATE_SCRIPT = "scripts/compute_estimate/estimate_compute.py"
# The interactive preview, run as a module in the selected environment
INTERACTIVE_PREVIEW_MODULE = "afabench.pipeline_preview"
# The Python of the node's venv, on a site whose allocations name venvs
VENV_PYTHON = "workflow/venvs/bin/python"
ESTIMATE_FLAGS = ("-e", "--estimate")
CATALOGS = {
    "methods": "workflow/conf/methods/all.yaml",
    "datasets": "workflow/conf/datasets/all.yaml",
    "unmaskers": "conf/components/unmasker",
    "initializers": "conf/components/initializer",
}
# The estimate's own options, taken with -e only.
ESTIMATE_OPTIONS = ("job_durations", "by", "output", "strict")
ESTIMATE_PANEL = "Compute estimate options, with -e (docs/reference/estimate_compute_command.md)"
WANDB_GROUP_SUFFIX_LENGTH = 4
# Where the launcher command keeps its arguments, -- and after included
ARGUMENTS = "arguments"


class LauncherError(click.UsageError):
    """Launcher arguments that name no profile or contradict each other."""


@dataclass(frozen=True, kw_only=True)
class LauncherOptions:
    """The launcher's parsed arguments, before any --."""

    profile_and_overrides: list[str]
    configfile: str | None
    list_methods: bool
    list_datasets: bool
    list_unmaskers: bool
    list_initializers: bool
    site: str | None
    dry_run: bool
    estimate: bool
    interactive_preview: bool
    group_by: str | None
    soft_budget: bool | None
    per_dataset_plots: bool | None
    wandb: bool | None
    wandb_group: str | None
    target: str
    jobs: int | None
    job_durations: str | None
    by: list[str] | None
    output: str | None
    strict: bool | None


def launcher_command(
    arguments: list[str], checkout: Path = REPO_ROOT
) -> list[str]:
    """Return the command the launcher runs for `arguments`."""
    options = _parse(arguments)
    if options.interactive_preview:
        _check_interactive_preview(options)
        return [
            *_environment_interpreter(arguments, checkout),
            "-m",
            INTERACTIVE_PREVIEW_MODULE,
            *(
                []
                if options.group_by is None
                else ["--group-by", options.group_by]
            ),
            "--",
            *snakemake_arguments(arguments, checkout),
        ]
    if not options.estimate:
        command = snakemake_command(arguments, checkout)
        if options.configfile is not None:
            return [
                *_environment_interpreter(arguments, checkout),
                "workflow/run_from_config.py",
                *arguments,
            ]
        return command
    if options.dry_run:
        message = (
            "-n and -e: an estimate runs no jobs either; give one of them"
        )
        raise LauncherError(message)
    # Check the run's arguments, as the estimate will.
    snakemake_arguments(arguments, checkout)
    return [
        *_environment_interpreter(arguments, checkout),
        ESTIMATE_SCRIPT,
        *_without_estimate_flag(arguments),
    ]


def _check_interactive_preview(options: LauncherOptions) -> None:
    """Refuse the modes that mean something only when jobs run."""
    if options.estimate:
        message = "-e and --interactive-preview: give one of them"
        raise LauncherError(message)
    if options.wandb or options.wandb_group is not None:
        flag = "--wandb" if options.wandb else "--wandb-group"
        message = f"{flag} logs the jobs a run executes, and --interactive-preview executes none: give one of them"
        raise LauncherError(message)


def _planning_only(arguments: list[str]) -> bool:
    """Planning does not require runnable compute-node environments."""
    options = _parse(arguments)
    extra = arguments[arguments.index("--") + 1 :] if "--" in arguments else []
    return (
        options.interactive_preview
        or options.estimate
        or options.dry_run
        or any(argument in {"-n", "--dry-run"} for argument in extra)
    )


def _environment_interpreter(
    arguments: list[str], checkout: Path
) -> list[str]:
    """Return the Python of the environment the launcher selects."""
    options = _parse(arguments)
    if (
        not _planning_only(arguments)
        and options.site is not None
        and _site_venvs(checkout, options.site)
    ):
        return [VENV_PYTHON]
    return ["uv", "run", "python"]


def _without_estimate_flag(arguments: list[str]) -> list[str]:
    split = arguments.index("--") if "--" in arguments else len(arguments)
    return [
        argument
        for argument in arguments[:split]
        if argument not in ESTIMATE_FLAGS
    ] + arguments[split:]


def snakemake_command(
    arguments: list[str], checkout: Path = REPO_ROOT
) -> list[str]:
    """Return the Snakemake command the launcher runs for `arguments`."""
    interpreter, snakemake = _invocation(arguments, checkout)
    return [*interpreter, *snakemake]


def snakemake_arguments(
    arguments: list[str], checkout: Path = REPO_ROOT
) -> list[str]:
    """Return the arguments of that command after `snakemake`."""
    return _invocation(arguments, checkout)[1]


def _invocation(
    arguments: list[str], checkout: Path
) -> tuple[list[str], list[str]]:
    """Return the interpreter running Snakemake, and Snakemake's arguments."""
    planning_only = _planning_only(arguments)
    extra: list[str] = []
    if "--" in arguments:
        split = arguments.index("--")
        arguments, extra = arguments[:split], arguments[split + 1 :]
    if any(argument.startswith("--config") for argument in extra):
        # A second --config would replace the one carrying the site file.
        message = "--config after --: give pipeline profile overrides as key=value before --"
        raise LauncherError(message)
    options = _parse(arguments)
    pipeline, config = _pipeline_and_overrides(options.profile_and_overrides)
    pipeline, profiles = _configuration_source(options, checkout, pipeline)
    config += _flag_overrides(options, config)
    if _override_is_true(config, "use_wandb") and not _overrides(
        config, "wandb_group"
    ):
        smoke_test = _override_is_true(config, "smoke_test")
        config.append(
            f"wandb_group={new_wandb_group(pipeline, smoke_test=smoke_test)}"
        )
    interpreter = ["uv", "run", "snakemake"]
    if options.site is not None:
        if options.jobs is not None:
            message = f"--jobs {options.jobs}: a cluster run submits as many jobs as the site profile {options.site!r} sets"
            raise LauncherError(message)
        site = _profile(checkout, SITE_PROFILES, options.site)
        if _site_venvs(checkout, options.site) and not planning_only:
            interpreter = [VENV_PYTHON, "-m", "snakemake"]
        profiles += ["--workflow-profile", site]
        config.append(f"execution_site_file={site}/site.yaml")
    # The target first: a multi-value option among the extra arguments
    # would take it as one of its values.
    command = (
        [options.target, *profiles]
        if options.configfile is not None
        else [*profiles, options.target]
    )
    if options.jobs is not None:
        command += ["--jobs", str(options.jobs)]
    # The interactive preview runs nothing either, so -n adds nothing to it.
    if options.dry_run and not options.interactive_preview:
        command += ["--dry-run", "--printshellcmds"]
    command += extra
    if config:
        command += ["--config", *config]
    return interpreter, command


def _configuration_source(
    options: LauncherOptions, checkout: Path, pipeline: str
) -> tuple[str, list[str]]:
    if options.configfile is None:
        return pipeline, [
            "--profile",
            _profile(checkout, PIPELINE_PROFILES, pipeline),
        ]
    if (
        options.profile_and_overrides
        and "=" not in options.profile_and_overrides[0]
    ):
        message = "Give a pipeline profile or --configfile, not both"
        raise LauncherError(message)
    path = Path(options.configfile)
    if not (checkout / path).is_file():
        message = f"No benchmark config file at {path}"
        raise LauncherError(message)
    return path.stem, [
        "--snakefile",
        "workflow/snakefiles/orchestration/pipeline.smk",
        "--configfile",
        str(path),
    ]


def _flag_overrides(
    options: LauncherOptions, overrides: list[str]
) -> list[str]:
    """Return the overrides the launcher's flags stand for."""
    added = []
    if options.soft_budget is not None:
        added.append(
            _flag_override(
                "soft-budget",
                "soft_budget_setting",
                overrides,
                on=options.soft_budget,
            )
        )
    if options.per_dataset_plots is not None:
        added.append(
            _flag_override(
                "per-dataset-plots",
                "per_dataset_plots",
                overrides,
                on=options.per_dataset_plots,
            )
        )
    if options.wandb_group is not None:
        added += _named_wandb_group(options, overrides)
    elif options.wandb is not None:
        added.append(
            _flag_override("wandb", "use_wandb", overrides, on=options.wandb)
        )
    return added


def _flag_override(
    flag: str, key: str, overrides: list[str], *, on: bool
) -> str:
    """Return the override --<flag> or --no-<flag> stands for."""
    if _overrides(overrides, key):
        message = f"--{'' if on else 'no-'}{flag} and {key}=: give one of them"
        raise LauncherError(message)
    return f"{key}={'true' if on else 'false'}"


def _named_wandb_group(
    options: LauncherOptions, overrides: list[str]
) -> list[str]:
    """Return the overrides --wandb-group stands for."""
    if options.wandb is False:
        message = "--no-wandb and --wandb-group: give one of them"
        raise LauncherError(message)
    if not options.wandb_group:
        message = "--wandb-group: give the W&B group a name"
        raise LauncherError(message)
    if _overrides(overrides, "use_wandb"):
        message = "--wandb-group and use_wandb=: give one of them"
        raise LauncherError(message)
    if _overrides(overrides, "wandb_group"):
        message = "--wandb-group and wandb_group=: give one of them"
        raise LauncherError(message)
    return ["use_wandb=true", f"wandb_group={options.wandb_group}"]


def new_wandb_group(pipeline: str, *, smoke_test: bool) -> str:
    """Name a new W&B group for an invocation of `pipeline`."""
    time = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "".join(
        random.choices(
            string.ascii_lowercase + string.digits,
            k=WANDB_GROUP_SUFFIX_LENGTH,
        )
    )
    return f"{pipeline}-{time}-{suffix}" + ("-smoke" if smoke_test else "")


def _overrides(overrides: list[str], key: str) -> list[str]:
    """Return the values the overrides give `key`."""
    return [
        word.split("=", 1)[1]
        for word in overrides
        if word.startswith(f"{key}=")
    ]


def _override_is_true(overrides: list[str], key: str) -> bool:
    values = _overrides(overrides, key)
    return bool(values) and values[-1].lower() == "true"


def _pipeline_and_overrides(words: list[str]) -> tuple[str, list[str]]:
    pipeline = "all"
    if words and "=" not in words[0]:
        pipeline, words = words[0], words[1:]
    for word in words:
        if "=" not in word:
            message = f"{word!r} is no pipeline profile override (key=value); the pipeline profile, {pipeline!r}, comes first"
            raise LauncherError(message)
    return pipeline, list(words)


def _parse(arguments: list[str]) -> LauncherOptions:
    """Parse the launcher's arguments, before any --."""
    if "--" in arguments:
        arguments = arguments[: arguments.index("--")]
    command = typer.main.get_command(app)
    try:
        with command.make_context(PROG, list(arguments)) as context:
            parameters = dict(context.params)
    except click.UsageError as error:
        if isinstance(error, LauncherError):
            raise
        raise LauncherError(error.format_message()) from error
    parameters["profile_and_overrides"] = list(
        parameters["profile_and_overrides"] or []
    )
    parameters["by"] = None if not parameters["by"] else list(parameters["by"])
    options = LauncherOptions(**parameters)
    _check_options(options)
    return options


def _check_options(options: LauncherOptions) -> None:
    """Refuse options that belong to a mode not selected."""
    if not options.estimate:
        for name in ESTIMATE_OPTIONS:
            if getattr(options, name) is not None:
                flag = "--" + name.replace("_", "-")
                message = (
                    f"{flag} is an option of the compute estimate: add -e"
                )
                raise LauncherError(message)
    if options.group_by is not None:
        if not options.interactive_preview:
            message = "--group-by groups the jobs of the interactive preview: add --interactive-preview"
            raise LauncherError(message)
        _check_grouping(options.group_by)


def _check_grouping(grouping: str) -> None:
    """Refuse a grouping the interactive preview would refuse."""
    # Imported here: planning takes seconds to import, needed by -i only.
    from afabench.pipeline_preview.model import (  # noqa: PLC0415
        GroupingError,
        parse_grouping,
    )

    try:
        parse_grouping(grouping)
    except GroupingError as error:
        message = f"--group-by: {error}"
        raise LauncherError(message) from error


def available_names(kind: str, checkout: Path = REPO_ROOT) -> list[str]:
    """List the shipped names of the catalog `kind`."""
    if kind not in CATALOGS:
        msg = f"Unknown catalog {kind!r}; choose from {', '.join(CATALOGS)}"
        raise LauncherError(msg)
    path = checkout / CATALOGS[kind]
    if kind in {"unmaskers", "initializers"}:
        names = sorted(config.stem for config in path.glob("*.yaml"))
        if not names:
            msg = f"{path}: no {kind} configs"
            raise LauncherError(msg)
        return names
    catalog = yaml.safe_load(path.read_text())
    names = catalog.get(kind) if isinstance(catalog, dict) else None
    if not isinstance(names, list) or not all(
        isinstance(name, str) for name in names
    ):
        message = f"{path}: expected {kind}: followed by a list of names"
        raise LauncherError(message)
    if not names:
        msg = f"{path}: no {kind} names"
        raise LauncherError(msg)
    return sorted(set(names))


def _profile(checkout: Path, profiles: str, name: str) -> str:
    """Return the path of the profile `name` under `profiles`."""
    path = f"{profiles}/{name}"
    if not (checkout / path / "config.yaml").is_file():
        shipped = sorted(
            str(config.parent.relative_to(checkout / profiles))
            for config in (checkout / profiles).rglob("config.yaml")
        )
        message = f"No profile {name!r} under {profiles}; there are: {', '.join(shipped)}"
        raise LauncherError(message)
    return path


def _site_venvs(checkout: Path, site: str) -> dict[str, str]:
    """Return the venv each allocation of the site profile `site` names."""
    site_file = f"{_profile(checkout, SITE_PROFILES, site)}/site.yaml"
    if not (checkout / site_file).is_file():
        message = f"The site profile has no site file {site_file}"
        raise LauncherError(message)
    try:
        return site_venvs(read_site_file(checkout / site_file), site_file)
    except TypeError as error:
        raise LauncherError(str(error)) from error


def _check_terminal() -> None:
    """Fail unless the launcher reads from and writes to a terminal."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        message = "--interactive-preview needs an interactive terminal on stdin and stdout; for a preview in plain text, give -n instead"
        raise LauncherError(message)


def check_site_venvs(arguments: list[str], checkout: Path = REPO_ROOT) -> None:
    """Check site build readiness; warn for planning, refuse execution."""
    site = _parse(arguments).site
    if site is None:
        return
    problems = [
        problem
        for venv in sorted(set(_site_venvs(checkout, site).values()))
        if (problem := venv_problem(checkout, venv)) is not None
    ]
    if problems:
        message = f"{'; '.join(problems)}. Build the site's venvs with `uv run just build-venvs {site}` (docs/how-to/run_on_slurm.md) before executing jobs"
        if _planning_only(arguments):
            typer.echo(
                f"Warning: {message}. Continuing planning in the checkout environment; this does not certify execution readiness.",
                err=True,
            )
        else:
            raise LauncherError(message)


class _LauncherCommand(TyperCommand):
    """Keeps every argument, and the Snakemake arguments after -- unparsed."""

    @override
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.meta[ARGUMENTS] = list(args)
        if "--" in args:
            args = args[: args.index("--")]
        return super().parse_args(ctx, args)


app = typer.Typer(
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.command(
    cls=_LauncherCommand,
    epilog="Arguments after -- go to Snakemake unchanged.",
)
def main(
    ctx: typer.Context,
    *,
    profile_and_overrides: Annotated[
        list[str] | None,
        typer.Argument(
            metavar="[PIPELINE] [KEY=VALUE]...",
            help=f"A pipeline profile under {PIPELINE_PROFILES}/ (default: all), then pipeline profile overrides",
            show_default=False,
        ),
    ] = None,
    configfile: Annotated[
        str | None,
        typer.Option(
            metavar="PATH",
            help="Run a benchmark YAML config instead of a named pipeline profile",
        ),
    ] = None,
    list_methods: Annotated[
        bool,
        typer.Option(
            "--list-methods", help="List all shipped methods and exit"
        ),
    ] = False,
    list_datasets: Annotated[
        bool,
        typer.Option(
            "--list-datasets", help="List all shipped datasets and exit"
        ),
    ] = False,
    list_unmaskers: Annotated[
        bool,
        typer.Option(
            "--list-unmaskers", help="List all shipped unmaskers and exit"
        ),
    ] = False,
    list_initializers: Annotated[
        bool,
        typer.Option(
            "--list-initializers",
            help="List all shipped initializers and exit",
        ),
    ] = False,
    site: Annotated[
        str | None,
        typer.Option(
            help=f"A site profile under {SITE_PROFILES}/; without one, jobs run locally"
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "-n",
            "--dry-run",
            help="Plan the run and print its commands without running jobs",
        ),
    ] = False,
    estimate: Annotated[
        bool,
        typer.Option(
            "-e",
            "--estimate",
            help="Print the compute estimate of the run without running jobs",
        ),
    ] = False,
    interactive_preview: Annotated[
        bool,
        typer.Option(
            "-i",
            "--interactive-preview",
            help="Browse the run's pipeline plan in the terminal: the jobs it would run and the jobs it would reuse, each with its job status, running none (-n changes nothing)",
        ),
    ] = False,
    group_by: Annotated[
        str | None,
        typer.Option(
            metavar="DIMENSIONS",
            help="With --interactive-preview, group the jobs first by these comma-separated dimensions in order: stage, method and dataset, or some of them (default: stage,method,dataset)",
        ),
    ] = None,
    soft_budget: Annotated[
        bool | None,
        typer.Option(
            help="Run the soft-budget setting too, or only the hard-budget one (soft_budget_setting=true or false; default: the pipeline profile's)",
            show_default=False,
        ),
    ] = None,
    per_dataset_plots: Annotated[
        bool | None,
        typer.Option(
            help="Add per-dataset Accuracy, Macro F1 and AUROC figures for both classifiers, or not (per_dataset_plots=true or false; default: the pipeline profile's)",
            show_default=False,
        ),
    ] = None,
    wandb: Annotated[
        bool | None,
        typer.Option(
            help="Log to W&B and link each job's W&B run in the terminal, every run in a new W&B group, or not (use_wandb=true or false; default: the pipeline profile's); needs WANDB_ENTITY and WANDB_PROJECT",
            show_default=False,
        ),
    ] = None,
    wandb_group: Annotated[
        str | None,
        typer.Option(
            metavar="NAME",
            help="Log to W&B as --wandb does, every run in the W&B group NAME, such as an earlier invocation's to continue it (use_wandb=true wandb_group=NAME)",
        ),
    ] = None,
    target: Annotated[str, typer.Option(help="The rule to run")] = "all",
    jobs: Annotated[
        int | None,
        typer.Option(
            help="How many jobs run at once locally; a site profile sets it for its cluster"
        ),
    ] = None,
    job_durations: Annotated[
        str | None,
        typer.Option(
            metavar="PATH",
            help="A job duration table, or an output root of job records (default: output/production)",
            rich_help_panel=ESTIMATE_PANEL,
        ),
    ] = None,
    by: Annotated[
        list[str] | None,
        typer.Option(
            metavar="COLUMN",
            help="Group the totals by this job column; repeat for several",
            rich_help_panel=ESTIMATE_PANEL,
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            metavar="CSV",
            help="Also write each planned job's estimate here as CSV",
            rich_help_panel=ESTIMATE_PANEL,
        ),
    ] = None,
    strict: Annotated[
        bool | None,
        typer.Option(
            help="Exit with 1 when a planned job is unestimated",
            rich_help_panel=ESTIMATE_PANEL,
            show_default=False,
        ),
    ] = None,
) -> None:
    """Run the pipeline."""
    options = LauncherOptions(
        profile_and_overrides=profile_and_overrides or [],
        configfile=configfile,
        list_methods=list_methods,
        list_datasets=list_datasets,
        list_unmaskers=list_unmaskers,
        list_initializers=list_initializers,
        site=site,
        dry_run=dry_run,
        estimate=estimate,
        interactive_preview=interactive_preview,
        group_by=group_by,
        soft_budget=soft_budget,
        per_dataset_plots=per_dataset_plots,
        wandb=wandb,
        wandb_group=wandb_group,
        target=target,
        jobs=jobs,
        job_durations=job_durations,
        by=by,
        output=output,
        strict=strict,
    )
    requested = [name for name in CATALOGS if getattr(options, "list_" + name)]
    if requested:
        for name in requested:
            typer.echo(f"Available {name} ({CATALOGS[name]}):")
            typer.echo("\n".join(available_names(name)))
        raise typer.Exit
    arguments: list[str] = ctx.meta[ARGUMENTS]
    try:
        command = launcher_command(arguments)
        check_site_venvs(arguments)
        if options.interactive_preview:
            _check_terminal()
    except LauncherError as error:
        error.ctx = ctx
        raise
    typer.echo(shlex.join(command), err=True)
    # Site profiles expand $PWD when Snakemake reads them, as the checkout
    # root, which a changed working directory alone does not set.
    environment = {**os.environ, "PWD": str(REPO_ROOT)}
    # Snakemake exposes its merged config and files, but not its profile.
    if not (options.estimate or options.interactive_preview):
        environment.pop("AFABENCH_PIPELINE_PROFILE", None)
        if "--profile" in command:
            profile_index = command.index("--profile") + 1
            environment["AFABENCH_PIPELINE_PROFILE"] = command[profile_index]
    raise typer.Exit(
        subprocess.run(
            command, cwd=REPO_ROOT, env=environment, check=False
        ).returncode
    )


if __name__ == "__main__":
    app(prog_name=PROG)
