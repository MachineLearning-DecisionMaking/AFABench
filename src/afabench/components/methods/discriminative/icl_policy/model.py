"""
Local pretrained TabICL backbone and classifier with a separate policy branch.

Joint mode updates all components; frozen mode updates only the policy. Source implementations live in
upstream/; the policy and acquisition-design TODOs live in policy.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, override

import torch
from jaxtyping import Float, Int
from torch import nn

from afabench.components.methods.discriminative.icl_policy.policy import (
    HeadScores,
    PolicyHead,
    QueryRepresentations,
)
from afabench.components.methods.discriminative.icl_policy.preprocessing import (
    OfficialEnsemblePreprocessor,
)
from afabench.components.methods.discriminative.icl_policy.upstream.tabicl import (
    TabICL,
)

type TableFeatures = Float[torch.Tensor, "batch rows features"]
type ContextLabels = Int[torch.Tensor, "batch context"]
type ClassLogits = Float[torch.Tensor, "batch query classes"]


@dataclass
class ICLPolicyOutput:
    class_logits: ClassLogits
    policy_scores: HeadScores
    query_representations: list[
        Float[torch.Tensor, "views query representation"]
    ]


class ICLPolicy(nn.Module):
    """
    Shared TabICL backbone with classifier and embedding-only policy heads.

    Inputs are numeric tables with NaNs for hidden/missing values.
    Official TabICL preprocessing runs inside this model interface. Context instances come
    first. Only their labels are passed in. This is a differentiable research
    path using native preprocessing/ensemble definitions, not cached inference
    or an AFABench method contract.
    """

    def __init__(
        self,
        tabicl: TabICL,
        *,
        n_features: int,
        training_mode: Literal["joint", "frozen"] = "joint",
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        ensemble_seed: int = 17,
    ) -> None:
        super().__init__()
        if tabicl.max_classes < 2:
            message = f"max_classes={tabicl.max_classes} must be at least 2"
            raise ValueError(message)
        if tabicl.dropout != 0.0:
            message = f"dropout={tabicl.dropout} must be zero for this path"
            raise ValueError(message)
        if n_estimators < 1 or softmax_temperature <= 0:
            message = "n_estimators and softmax_temperature must be positive"
            raise ValueError(message)
        self.n_estimators: int = n_estimators
        self.softmax_temperature: float = softmax_temperature
        self.ensemble_seed: int = ensemble_seed
        self.tabicl: TabICL = tabicl
        # P07 UPDATES: two optimizer groups; pretrained weights are a starting
        # point, not fixed in joint mode. No mask enters the backbone/head.
        if training_mode not in {"joint", "frozen"}:
            message = f"Unknown training_mode={training_mode!r}"
            raise ValueError(message)
        self.training_mode: Literal["joint", "frozen"] = training_mode
        # Preserve upstream's fixed parameters (for example RoPE frequencies).
        # Globally enabling gradients would turn fixed positional constants
        # into trainable weights while their upstream cache remains fixed.
        if training_mode == "frozen":
            self.tabicl.requires_grad_(False)
        self.policy: PolicyHead = PolicyHead(
            representation_dim=tabicl.embed_dim * tabicl.row_num_cls,
            n_features=n_features,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        n_features: int,
        training_mode: Literal["joint", "frozen"] = "joint",
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        ensemble_seed: int = 17,
    ) -> "ICLPolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        tabicl = TabICL(**checkpoint["config"])
        tabicl.load_state_dict(checkpoint["state_dict"], strict=True)
        return cls(
            tabicl,
            n_features=n_features,
            training_mode=training_mode,
            n_estimators=n_estimators,
            softmax_temperature=softmax_temperature,
            ensemble_seed=ensemble_seed,
        )

    def encode_prepared(
        self,
        features: TableFeatures,
        context_labels: ContextLabels,
        *,
        n_classes: int,
    ) -> QueryRepresentations:
        if not 2 <= n_classes <= self.tabicl.max_classes:
            message = (
                f"n_classes={n_classes} must be between 2 and "
                f"{self.tabicl.max_classes}"
            )
            raise ValueError(message)
        if features.ndim != 3 or context_labels.ndim != 2:
            message = "Expected features [B, N+M, F] and context_labels [B, N]"
            raise ValueError(message)
        n_context = context_labels.shape[1]
        if features.shape[0] != context_labels.shape[0] or not (
            0 < n_context < features.shape[1]
        ):
            message = (
                f"Need matching batches and nonempty context/query sets: "
                f"features={features.shape}, labels={context_labels.shape}"
            )
            raise ValueError(message)

        # Select differentiable TabICL execution even after wrapper.eval().
        # Dropout is required to be zero above; train() does not unfreeze
        # parameters. Use torch.no_grad() externally when gradients aren't
        # needed. Inference-manager paths would disable input gradients.
        self.tabicl.train()

        # 1. Backbone: column embedding, then row interaction.
        cells = self.tabicl.col_embedder(
            features,
            y_train=context_labels,
            embed_with_test=False,
        )
        rows = self.tabicl.row_interactor(cells)

        # 2. ICL: add labels only to context, then dataset-wise attention.
        icl = self.tabicl.icl_predictor
        labeled_context = rows[:, :n_context] + icl.y_encoder(
            context_labels.float()
        )
        states = torch.cat((labeled_context, rows[:, n_context:]), dim=1)
        states = icl.tf_icl(states, train_size=n_context)
        if icl.norm_first:
            states = icl.ln(states)
        queries = states[:, n_context:]

        return queries

    def encode_grouped(
        self,
        features: list[torch.Tensor],
        labels: list[torch.Tensor],
        *,
        n_classes: int,
    ) -> list[torch.Tensor]:
        """Batch independent views of equal shape; preserve their original order."""
        groups: dict[tuple[int, ...], list[int]] = {}
        for index, values in enumerate(features):
            groups.setdefault(tuple(values.shape), []).append(index)
        outputs: dict[int, torch.Tensor] = {}
        for indices in groups.values():
            states = self.encode_prepared(
                torch.stack([features[i] for i in indices]),
                torch.stack([labels[i] for i in indices]),
                n_classes=n_classes,
            )
            for position, index in enumerate(indices):
                outputs[index] = states[position : position + 1]
        return [outputs[i] for i in range(len(features))]

    def _processor(
        self, features: TableFeatures, labels: ContextLabels, n_classes: int
    ) -> OfficialEnsemblePreprocessor:
        if not 2 <= n_classes <= self.tabicl.max_classes:
            message = f"n_classes={n_classes} exceeds classifier capacity"
            raise ValueError(message)
        return OfficialEnsemblePreprocessor(
            features[: labels.shape[0]],
            labels,
            n_classes,
            self.n_estimators,
            self.ensemble_seed,
        )

    def classify(
        self,
        features: TableFeatures,
        context_labels: ContextLabels,
        *,
        n_classes: int,
    ) -> ClassLogits:
        return self(features, context_labels, n_classes=n_classes).class_logits

    def classify_reveal(
        self,
        masked_features: TableFeatures,
        context_labels: ContextLabels,
        query_values: TableFeatures,
        selection: TableFeatures,
        *,
        n_classes: int,
    ) -> ClassLogits:
        """Reveal the SAME original feature in every view; average class logits."""
        n_context = context_labels.shape[1]
        prepared_views, prepared_labels, class_orders, counts = [], [], [], []
        for i in range(masked_features.shape[0]):
            processor = self._processor(
                masked_features[i], context_labels[i], n_classes
            )
            baseline = processor.transform(masked_features[i])
            candidates = processor.transform(
                torch.cat(
                    (masked_features[i, :n_context], query_values[i]), dim=0
                )
            )
            revealed_raw = torch.cat(
                (
                    masked_features[i, :n_context],
                    torch.where(
                        selection[i].detach().bool(),
                        query_values[i],
                        masked_features[i, n_context:],
                    ),
                ),
                dim=0,
            )
            hard_views = processor.transform(revealed_raw)
            counts.append(len(baseline))
            for view, candidate, hard in zip(
                baseline, candidates, hard_views, strict=True
            ):
                kept_count = view.features.shape[-1]
                order = view.feature_order[:kept_count]
                gate = selection[i, :, order]
                query = view.features[n_context:] + gate * (
                    candidate.features[n_context:] - view.features[n_context:]
                )
                # Exact official forward even if a power-transform fallback
                # depends on another cell. Only backward uses the surrogate.
                query = hard.features[n_context:] + (query - query.detach())
                prepared_views.append(
                    torch.cat((hard.features[:n_context], query), dim=0)
                )
                prepared_labels.append(view.labels)
                class_orders.append(view.class_order)
        states = self.encode_grouped(
            prepared_views, prepared_labels, n_classes=n_classes
        )
        raw = self.tabicl.icl_predictor.decoder(torch.cat(states))[
            ..., :n_classes
        ]
        outputs, offset = [], 0
        for count in counts:
            logits = [
                raw[j : j + 1, ..., class_orders[j]]
                for j in range(offset, offset + count)
            ]
            outputs.append(
                torch.stack(logits).mean(0) / self.softmax_temperature
            )
            offset += count
        return torch.cat(outputs, dim=0)

    @override
    def forward(
        self,
        features: TableFeatures,
        context_labels: ContextLabels,
        *,
        n_classes: int,
    ) -> ICLPolicyOutput:
        if (
            features.ndim != 3
            or context_labels.ndim != 2
            or features.shape[0] != context_labels.shape[0]
            or features.shape[-1] != self.policy.n_features
        ):
            message = "Need matching batched tables/labels and configured feature width"
            raise ValueError(message)
        batches = []
        for i in range(features.shape[0]):
            processor = self._processor(
                features[i], context_labels[i], n_classes
            )
            batches.append(processor.transform(features[i]))
        views = [view for batch_views in batches for view in batch_views]
        states = self.encode_grouped(
            [v.features for v in views],
            [v.labels for v in views],
            n_classes=n_classes,
        )
        combined = torch.cat(states)
        raw = self.tabicl.icl_predictor.decoder(combined)[..., :n_classes]
        policy_scores = self.policy(combined)
        all_logits, all_scores, all_states = [], [], []
        offset = 0
        for batch_views in batches:
            logits, scores = [], []
            for j, view in enumerate(batch_views, start=offset):
                logits.append(raw[j : j + 1, ..., view.class_order])
                scores.append(
                    policy_scores[
                        j : j + 1, ..., torch.argsort(view.feature_order)
                    ]
                )
            count = len(batch_views)
            all_logits.append(
                torch.stack(logits).mean(0) / self.softmax_temperature
            )
            all_scores.append(torch.stack(scores).mean(0))
            all_states.append(combined[offset : offset + count])
            offset += count
        return ICLPolicyOutput(
            torch.cat(all_logits), torch.cat(all_scores), all_states
        )
