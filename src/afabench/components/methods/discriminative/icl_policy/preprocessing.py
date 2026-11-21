"""
Actual TabICL numeric preprocessing, fitted on labeled context only.

Calls the authors' unchanged components. Production uses the native ensemble
generator with none/power views, feature permutations and class shifts.
The one-view numeric helper remains for parity/gradient tests. No
custom mean computation replaces the imputer. NumPy fitting/transforms are
not differentiated; relaxed acquisition mixes their fixed numeric endpoints.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from afabench.components.methods.discriminative.icl_policy.upstream.preprocessing import (
    EnsembleGenerator,
    PreprocessingPipeline,
    TransformToNumerical,
    UniqueFeatureFilter,
)


class OfficialNumericPreprocessor:
    def __init__(self, context: Tensor) -> None:
        self.encoder: TransformToNumerical = TransformToNumerical()
        numeric = self.encoder.fit_transform(context.detach().cpu().numpy())
        self.filter: UniqueFeatureFilter = UniqueFeatureFilter()
        filtered = self.filter.fit_transform(numeric)
        self.pipeline: PreprocessingPipeline = PreprocessingPipeline(
            normalization_method="none",
            outlier_threshold=4.0,
            random_state=17,
        )
        self.pipeline.fit(filtered)
        self.kept: Tensor = torch.as_tensor(
            self.filter.features_to_keep_,
            device=context.device,
            dtype=torch.bool,
        )

    def transform(self, features: Tensor) -> Tensor:
        numeric = self.encoder.transform(features.detach().cpu().numpy())
        output = self.pipeline.transform(self.filter.transform(numeric))
        return torch.as_tensor(
            output.copy(), dtype=features.dtype, device=features.device
        )


@dataclass
class EnsembleView:
    features: Tensor
    labels: Tensor
    feature_order: Tensor
    class_order: Tensor


class OfficialEnsemblePreprocessor:
    """Native TabICL ensemble generation; true labels never replaced in views."""

    def __init__(
        self,
        context: Tensor,
        labels: Tensor,
        n_classes: int,
        n_estimators: int,
        seed: int,
    ) -> None:
        self.encoder: TransformToNumerical = TransformToNumerical()
        numeric = self.encoder.fit_transform(context.detach().cpu().numpy())
        # Native generator uses y only for class count and permutations.
        # Reserve ALL task classes even when one is absent from this context.
        if len(context) < n_classes:
            message = "Ensemble context must have at least n_classes instances"
            raise ValueError(message)
        slots = np.arange(len(context)) % n_classes
        self.generator: EnsembleGenerator = EnsembleGenerator(
            classification=True,
            n_estimators=n_estimators,
            norm_methods=["none"] if n_estimators == 1 else ["none", "power"],
            feat_shuffle_method="none" if n_estimators == 1 else "latin",
            class_shuffle_method="none" if n_estimators == 1 else "shift",
            outlier_threshold=4.0,
            random_state=seed,
        )
        self.generator.fit(numeric, slots)
        self.labels: Tensor = labels
        self.kept: Tensor = torch.as_tensor(
            self.generator.unique_filter_.features_to_keep_,
            device=context.device,
            dtype=torch.bool,
        )

    def transform(self, features: Tensor) -> list[EnsembleView]:
        numeric = self.encoder.transform(features.detach().cpu().numpy())
        filtered = self.generator.unique_filter_.transform(numeric)
        kept = torch.where(self.kept)[0]
        dropped = torch.where(~self.kept)[0]
        views = []
        # Native grouping uses set order; stabilize accumulation across restarts.
        for method in self.generator.norm_methods_:
            if method not in self.generator.ensemble_configs_:
                continue
            pipeline = self.generator.preprocessors_[method]
            n_context = self.labels.shape[0]
            prepared = np.concatenate(
                (
                    pipeline.X_transformed_,
                    pipeline.transform(filtered[n_context:]),
                ),
                axis=0,
            )
            for (
                feature_pattern,
                class_pattern,
            ) in self.generator.ensemble_configs_[method]:
                order = torch.as_tensor(
                    feature_pattern, device=features.device, dtype=torch.long
                )
                classes = torch.as_tensor(
                    class_pattern, device=features.device, dtype=torch.long
                )
                # Head outputs use view positions; filtered columns occupy the
                # tail, in original order, and remain eligible external choices.
                full_order = torch.cat((kept[order], dropped))
                tensor = torch.as_tensor(
                    prepared[:, feature_pattern].copy(),
                    device=features.device,
                    dtype=features.dtype,
                )
                views.append(
                    EnsembleView(
                        tensor, classes[self.labels], full_order, classes
                    )
                )
        return views
