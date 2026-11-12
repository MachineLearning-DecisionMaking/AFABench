# Set up AFABench

## Check prerequisites

Required:

- **OS**: Linux (x86_64 or aarch64). macOS is untested. Native Windows is
  not supported; WSL2 is untested.
- **bash** on `PATH`. Pipeline jobs run their commands with it.
- **git**, to clone the repository.
- **[uv](https://docs.astral.sh/uv/getting-started/installation/) 0.10.2 or
  newer.** uv installs the exact Python version (3.12.10) and the locked
  dependencies, so you don't need to install Python yourself.

Only for some setups:

- **GPU execution**: an NVIDIA GPU and its host drivers.
- **Cluster execution**: the site's SLURM. See [Run on SLURM](run_on_slurm.md).
- **NixOS**: enter `nix develop` before running `uv sync`. The flake provides
  Git, development tools, a launcher Python and runtime libraries for native
  Python wheels on x86_64 and aarch64 Linux. With direnv installed, run
  `direnv allow` to activate the flake automatically. Other hosts do not need
  the flake.

## Install the checkout

```bash
git clone https://github.com/MachineLearning-DecisionMaking/AFABench.git
cd AFABench
uv sync
```

Run pipeline commands from this repository root.
