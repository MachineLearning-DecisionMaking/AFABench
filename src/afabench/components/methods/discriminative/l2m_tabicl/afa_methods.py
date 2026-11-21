"""AFABench adapter for a frozen TabICL classifier and an L2M policy."""

from dataclasses import asdict
from pathlib import Path
from typing import Self, override

import torch
from torch import nn

from afabench.components.methods.common import (
    AFAMethodBase,
    on_method_device,
)
from afabench.components.methods.discriminative.l2m.afa_methods import (
    require_direct_unmasker,
)
from afabench.components.methods.discriminative.l2m_tabicl.checkpoint import (
    load_models,
    save_models,
)
from afabench.components.methods.discriminative.l2m_tabicl.policy import (
    AcquisitionPolicy,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    FrozenTabICLPredictor,
    TableFeatures,
)
from afabench.components.methods.discriminative.l2m_tabicl.preprocessing import (
    ContextPreprocessor,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.types import (
    AFAAction,
    ClassifierOutputKind,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


class L2MTabICLAFAMethod(AFAMethodBase):
    """
    Select individual numerical features with a fixed labeled context set.

    Context labels are one-hot, as required by AFABench. Query labels are
    ignored. This initial adapter requires finite context data and observed
    query values, DirectUnmasker, and a hard budget below the feature count.
    It has no learned stop action or soft-budget objective.
    """

    def __init__(
        self,
        predictor: FrozenTabICLPredictor,
        policy: AcquisitionPolicy,
        context_features: Features,
        context_labels: Label,
        *,
        unmasker: UnmaskerConfig,
    ) -> None:
        require_direct_unmasker(unmasker)
        first_layer = policy.network[0]
        last_layer = policy.network[-1]
        if not isinstance(first_layer, nn.Linear) or not isinstance(
            last_layer, nn.Linear
        ):
            message = "The policy must start and end with linear layers"
            raise TypeError(message)
        representation_dim = (
            predictor.model.embed_dim * predictor.model.row_num_cls
        )
        if policy.representation_dim != representation_dim:
            message = (
                f"Policy representation width={policy.representation_dim} "
                f"must match "
                f"TabICL representation width={representation_dim}"
            )
            raise ValueError(message)
        self.n_features: int = last_layer.out_features
        if (
            context_features.ndim != 2
            or context_features.shape[0] < 2
            or context_features.shape[1] != self.n_features
        ):
            message = (
                f"context_features shape={context_features.shape} must have "
                f"at least two rows and {self.n_features} features"
            )
            raise ValueError(message)
        if (
            context_labels.ndim != 2
            or context_labels.shape[0] != len(context_features)
            or not 2 <= context_labels.shape[1] <= predictor.model.max_classes
        ):
            message = f"Invalid context_labels shape={context_labels.shape}"
            raise ValueError(message)
        if not (
            ((context_labels == 0) | (context_labels == 1)).all()
            and (context_labels.sum(dim=-1) == 1).all()
        ):
            message = "context_labels must be one-hot class labels"
            raise ValueError(message)
        if any(
            parameter.requires_grad for parameter in predictor.parameters()
        ):
            message = "All TabICL parameters must be frozen"
            raise ValueError(message)

        device = next(predictor.parameters()).device
        super().__init__(device)
        self.predictor: FrozenTabICLPredictor = predictor.eval()
        self.policy: AcquisitionPolicy = policy.to(device).eval()
        self.n_classes: int = context_labels.shape[1]
        self.context_features: Features = (
            context_features.detach()
            .to(device=device, dtype=torch.float32)
            .clone()
        )
        self.context_labels: Label = (
            context_labels.detach()
            .to(device=device, dtype=torch.float32)
            .clone()
        )
        self.unmasker: UnmaskerConfig = unmasker
        self.preprocessor: ContextPreprocessor = ContextPreprocessor(
            self.context_features
        )
        self.processed_context: Features = self.preprocessor.transform(
            self.context_features
        )

    def _table(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> TableFeatures:
        if (
            masked_features.ndim != 2
            or masked_features.shape[0] < 1
            or masked_features.shape[1] != self.n_features
        ):
            message = f"Invalid masked_features shape={masked_features.shape}"
            raise ValueError(message)
        if (
            feature_mask.shape != masked_features.shape
            or feature_mask.dtype != torch.bool
        ):
            message = "feature_mask must be Boolean and match masked_features"
            raise ValueError(message)
        if not torch.isfinite(masked_features[feature_mask]).all():
            message = "Observed query features must be finite"
            raise ValueError(message)
        # Hidden placeholders must never be treated as observed raw values.
        raw_query = torch.where(
            feature_mask, masked_features.float(), float("nan")
        )
        query = self.preprocessor.transform(raw_query)
        return torch.cat((self.processed_context, query)).unsqueeze(0)

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return "logits"

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.predictor(
                self._table(masked_features, feature_mask),
                self.context_labels.argmax(dim=-1).unsqueeze(0),
                n_classes=self.n_classes,
            )[0]

    @override
    @on_method_device
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        if selection_mask is not None and (
            selection_mask.shape != feature_mask.shape
            or selection_mask.dtype != torch.bool
        ):
            message = "selection_mask must be Boolean and match feature_mask"
            raise ValueError(message)
        performed = (
            feature_mask
            if selection_mask is None
            else feature_mask | selection_mask
        )
        if performed.all(dim=-1).any():
            message = (
                "Selections are exhausted; use a hard budget below "
                "the number of features"
            )
            raise ValueError(message)
        with torch.no_grad():
            representations = self.predictor.query_representations(
                self._table(masked_features, feature_mask),
                self.context_labels.argmax(dim=-1).unsqueeze(0),
                n_classes=self.n_classes,
            )
            scores = self.policy(representations, feature_mask.unsqueeze(0))[0]
            return (
                scores.masked_fill(performed, -torch.inf).argmax(
                    dim=-1, keepdim=True
                )
                + 1
            )

    @override
    def save(self, path: Path) -> None:
        save_models(path / "models", self.predictor, self.policy)
        torch.save(
            {
                "context_features": self.context_features.cpu(),
                "context_labels": self.context_labels.cpu(),
                "unmasker": asdict(self.unmasker),
            },
            path / "method.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        predictor, policy = load_models(path / "models", device)
        checkpoint = torch.load(
            path / "method.pt", map_location="cpu", weights_only=True
        )
        return cls(
            predictor,
            policy,
            checkpoint["context_features"],
            checkpoint["context_labels"],
            unmasker=UnmaskerConfig(**checkpoint["unmasker"]),
        )
