"""Experimental L2M backbone/policy with a separate full TabICL classifier."""

import json
from pathlib import Path
from typing import Self, override

import torch

from afabench.components.methods.common.afa_method_base import on_method_device
from afabench.components.methods.discriminative.l2m.afa_methods import (
    L2MAFAMethod,
)
from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
    UpstreamBatch,
)
from afabench.components.methods.discriminative.l2m_tabicl.predictor import (
    FrozenTabICLPredictor,
)
from afabench.components.methods.discriminative.l2m_tabicl_backbone.numerical import (
    NumericalTransform,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.types import (
    AFAAction,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


class L2MTabICLBackboneAFAMethod(L2MAFAMethod):
    def __init__(
        self,
        model: L2MModel,
        predictor: FrozenTabICLPredictor,
        context_features: Features,
        context_labels: Label,
        *,
        unmasker: UnmaskerConfig,
    ) -> None:
        if not torch.isfinite(context_features).all():
            msg = "inference context_features must be finite"
            raise ValueError(msg)
        super().__init__(
            model, context_features, context_labels, unmasker=unmasker
        )
        self.predictor: FrozenTabICLPredictor = predictor.to(model.device)
        self.transform: NumericalTransform = NumericalTransform(
            self.context_features
        )
        self.model.ltm.predictor.requires_grad_(False)
        self.training_record: dict[str, object] = {}

    @override
    def _batch(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> UpstreamBatch:
        safe = torch.where(feature_mask.bool(), masked_features, 0.0)
        if not torch.isfinite(safe).all():
            msg = "observed query measurements must be finite"
            raise ValueError(msg)
        return super()._batch(safe, feature_mask)

    @override
    def save(self, path: Path) -> None:
        super().save(path)
        (path / "training.json").write_text(
            json.dumps(self.training_record, indent=2)
        )
        torch.save(
            {
                "config": self.predictor.config,
                "state_dict": self.predictor.model.state_dict(),
            },
            path / "tabicl.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        state = torch.load(
            path / "method.pt", map_location=device, weights_only=True
        )
        result = cls(
            L2MModel.load(path / "model", device),
            FrozenTabICLPredictor(path / "tabicl.pt").to(device),
            state["context_features"],
            state["context_labels"],
            unmasker=UnmaskerConfig(**state["unmasker"]),
        )
        result.training_record = json.loads(
            (path / "training.json").read_text()
        )
        return result

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        self._batch(masked_features, feature_mask)
        with torch.no_grad():
            table = torch.cat(
                (
                    self.transform(self.context_features),
                    self.transform.reveal(
                        masked_features, feature_mask.float()
                    ),
                )
            )
            return self.predictor(
                table.unsqueeze(0),
                self.context_labels.argmax(-1).unsqueeze(0),
                n_classes=self.model.n_classes,
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
        with torch.no_grad():
            scores = self.model.ltm.select_action(
                self._batch(masked_features, feature_mask)
            )[0]
            blocked = feature_mask.bool()
            if selection_mask is not None:
                if selection_mask.shape != blocked.shape:
                    msg = "selection_mask must match feature_mask"
                    raise ValueError(msg)
                blocked = blocked | selection_mask.bool()
            action = (
                scores.masked_fill(blocked, -torch.inf).argmax(
                    -1, keepdim=True
                )
                + 1
            )
            return torch.where(blocked.all(-1, keepdim=True), 0, action)
