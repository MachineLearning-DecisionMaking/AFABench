"""L2M real-feature tasks adapted to the TabICL policy experiment."""

import torch
from torch import Tensor

from afabench.components.methods.discriminative.icl_policy.train import (
    AcquisitionBatch,
)
from afabench.components.methods.discriminative.l2m.upstream.data_utils import (
    RealDataSampler,
    TabularDataset,
)


def make_sampler(pool: Tensor, n_classes: int) -> RealDataSampler:
    dataset = TabularDataset(
        data=pool,
        dim_y=1 if n_classes == 2 else n_classes,
        feature_names=[f"feature_{i}" for i in range(pool.shape[-1])],
        dataframe=None,
        free_indices=[],
    )
    return RealDataSampler(dataset, dataset)


def sample_task(
    sampler: RealDataSampler,
    *,
    batch_size: int,
    context_size: int,
    query_size: int,
    device: torch.device,
) -> AcquisitionBatch:
    """Upstream task values, BNN labels and availability masks."""
    raw = sampler.sample(
        batch_size=batch_size,
        num_ctx=context_size,
        max_num_points=context_size + query_size,
        num_observed=sampler.dim_x,
        device="cpu",
    )
    # Keep the upstream task values. Do not add a second custom scaler or
    # imputer; official preprocessing is owned by the model interface.
    context = raw["xc"]
    query = raw["xt"]
    cm = raw["rc"].bool()

    def labels(y: Tensor) -> Tensor:
        return y.squeeze(-1).long() if y.shape[-1] == 1 else y.argmax(-1)

    return AcquisitionBatch(
        context.to(device),
        labels(raw["yc"]).to(device),
        query.to(device),
        labels(raw["yt"]).to(device),
        cm.to(device),
        raw["rt"].bool().to(device),
    )
