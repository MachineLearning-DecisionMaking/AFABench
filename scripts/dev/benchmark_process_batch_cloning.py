"""
Benchmark the input-cloning overhead in `process_batch`'s entry point.

`process_batch` defensively clones `features`, `initial_feature_mask`, and
`initial_masked_features` on entry so that callers keep ownership of the
tensors they pass in. This script measures how much of end-to-end batch
evaluation time those three clones actually account for, on representative
workload sizes, to support an evidence-based decision on whether removing any
of them would be worthwhile. See GH issue #35 for the investigation.

Usage:
    uv run python scripts/dev/benchmark_process_batch_cloning.py
"""

import time

import torch

from afabench.core.types import AFAActionFn
from afabench.evaluation.eval import process_batch
from afabench.testing.helpers import get_direct_unmask_fn


def _budgeted_sequential_action_fn(
    n_selection_choices: int,
) -> AFAActionFn:
    def action_fn(
        masked_features: torch.Tensor,  # noqa: ARG001
        feature_mask: torch.Tensor,  # noqa: ARG001
        selection_mask: torch.Tensor | None = None,
        label: torch.Tensor | None = None,  # noqa: ARG001
        feature_shape: torch.Size | None = None,  # noqa: ARG001
    ) -> torch.Tensor:
        assert selection_mask is not None
        taken = selection_mask.sum(dim=-1)
        actions = torch.where(
            taken >= n_selection_choices,
            torch.zeros_like(taken),
            taken + 1,
        )
        return actions.long().unsqueeze(-1)

    return action_fn


def _benchmark(
    batch_size: int,
    n_features: int,
    selection_budget: int,
    n_repeats: int,
    device: torch.device,
) -> None:
    features = torch.randn(batch_size, n_features, device=device)
    initial_feature_mask = torch.zeros_like(features, dtype=torch.bool)
    initial_masked_features = torch.zeros_like(features)
    true_label = torch.zeros((batch_size, 3), device=device)

    afa_action_fn = _budgeted_sequential_action_fn(n_features)
    afa_unmask_fn = get_direct_unmask_fn()

    def run_once() -> None:
        process_batch(
            afa_action_fn=afa_action_fn,
            afa_unmask_fn=afa_unmask_fn,
            n_selection_choices=n_features,
            features=features,
            initial_feature_mask=initial_feature_mask,
            initial_masked_features=initial_masked_features,
            true_label=true_label,
            feature_shape=torch.Size((n_features,)),
            selection_budget=selection_budget,
        )

    run_once()  # warm up

    start = time.perf_counter()
    for _ in range(n_repeats):
        run_once()
    total_s = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(n_repeats):
        features.clone()
        initial_feature_mask.clone()
        initial_masked_features.clone()
    clones_s = time.perf_counter() - start

    print(
        f"batch_size={batch_size:>4} n_features={n_features:>4} "
        f"selection_budget={selection_budget:>3} device={device!s:>4} "
        f"n_repeats={n_repeats:>3} | "
        f"total={total_s / n_repeats * 1000:8.4f} ms/call  "
        f"three_clones={clones_s / n_repeats * 1000:8.5f} ms/call  "
        f"({100 * clones_s / total_s:6.3f}% of total)"
    )


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    # Workload sizes mirror the pinned AACO eval batch sizes in
    # test/src/common/test_workflow_settings.py: 32 for image datasets
    # (mnist/fashion_mnist, 784 features), 128 for tabular datasets
    # (cube, default). selection_budget caps episode length to a realistic
    # handful of acquisitions rather than exhausting every feature.
    _benchmark(
        batch_size=32,
        n_features=784,
        selection_budget=10,
        n_repeats=50,
        device=device,
    )
    _benchmark(
        batch_size=128,
        n_features=20,
        selection_budget=10,
        n_repeats=50,
        device=device,
    )
    _benchmark(
        batch_size=256,
        n_features=784,
        selection_budget=20,
        n_repeats=20,
        device=device,
    )


if __name__ == "__main__":
    main()
