"""One L2M acquisition loss evaluated through frozen TabICL."""

import torch
from jaxtyping import Int
from torch.nn import functional as F

from afabench.components.methods.discriminative.l2m.upstream.utils import (
    ConcreteSelector,
)
from afabench.components.methods.discriminative.l2m_tabicl.policy import (
    AcquisitionPolicy,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    ContextLabels,
    FrozenTabICLPredictor,
)
from afabench.components.methods.discriminative.l2m_tabicl.preprocessing import (
    MissingBaseline,
    ObservationMask,
    RowFeatures,
    apply_observation_mask,
)

type QueryLabels = Int[torch.Tensor, " query"]


def acquisition_loss(
    predictor: FrozenTabICLPredictor,
    policy: AcquisitionPolicy,
    *,
    context_features: RowFeatures,
    context_labels: ContextLabels,
    query_features: RowFeatures,
    query_labels: QueryLabels,
    feature_mask: ObservationMask,
    missing_baseline: MissingBaseline,
    n_classes: int,
    temperature: float,
    deterministic: bool = False,
) -> torch.Tensor:
    """
    Return mean query cross-entropy after one feature acquisition.

    This first step supports one task with finite numerical features, all
    retrospectively available. Features must be context-preprocessed;
    context labels have shape [1, context_rows], query labels [query_rows].
    The current mask must be binary and leave a feature for every query.
    """
    if not temperature > 0:
        message = f"temperature={temperature} must be positive"
        raise ValueError(message)

    if feature_mask.shape != query_features.shape:
        message = (
            f"feature_mask shape={feature_mask.shape} must match "
            f"query_features shape={query_features.shape}"
        )
        raise ValueError(message)

    if not ((feature_mask == 0) | (feature_mask == 1)).all():
        message = "feature_mask must contain only zero and one"
        raise ValueError(message)

    if feature_mask.bool().all(dim=-1).any():
        message = "Every query must have an unacquired feature"
        raise ValueError(message)

    if query_labels.shape != (query_features.shape[0],):
        message = (
            f"query_labels shape={query_labels.shape} must be "
            f"({query_features.shape[0]},)"
        )
        raise ValueError(message)

    def table(mask: ObservationMask) -> torch.Tensor:
        masked_query = apply_observation_mask(
            query_features, mask, missing_baseline
        )
        return torch.cat((context_features, masked_query), dim=0).unsqueeze(0)

    # Representation extraction is detached; only the selector is trained.
    representations = predictor.query_representations(
        table(feature_mask),
        context_labels,
        n_classes=n_classes,
    )

    scores = policy(representations, feature_mask.unsqueeze(0))

    # Only features not yet observed may be selected.
    available = (~feature_mask.bool()).to(scores.dtype).unsqueeze(0)
    selection = ConcreteSelector()(
        scores,
        temp=temperature,
        deterministic=deterministic,
        available=available,
    )

    # Match the authors' L2M mask update, preserving its relaxed gradient.
    updated_mask = torch.maximum(feature_mask, selection[0])

    # Keep gradients through this prediction to the selected feature mask.
    logits = predictor(
        table(updated_mask),
        context_labels,
        n_classes=n_classes,
    )

    return F.cross_entropy(
        logits.flatten(0, 1),
        query_labels,
    )
