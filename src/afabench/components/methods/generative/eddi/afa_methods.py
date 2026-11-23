import math
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, override

import torch
from torch import nn

from afabench.components.methods.common import AFAMethodBase, on_method_device
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.types import (
    AFAAction,
    AFAClassifier,
    ClassifierOutputKind,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.core.utils import flatten_afa_input

_SUPPORTED_CLASSIFIER_OUTPUT_KINDS = ("logits", "probabilities")

if TYPE_CHECKING:
    from afabench.components.methods.rl.odin.models import PartialVAE


def _flat_feature_shape(n_features: int) -> torch.Size:
    return torch.Size([n_features])


def _flatten_inputs(
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    label: Label | None,
    feature_shape: torch.Size | None,
) -> tuple[MaskedFeatures, FeatureMask, Label | None, torch.Size]:
    if feature_shape is None:
        batch_shape = torch.Size(masked_features.shape[:-1])
        return masked_features, feature_mask, label, batch_shape

    batch_shape = torch.Size(masked_features.shape[: -len(feature_shape)])
    flat_masked_features, flat_feature_mask, flat_label = flatten_afa_input(
        masked_features, feature_mask, label, feature_shape
    )
    return flat_masked_features, flat_feature_mask, flat_label, batch_shape


def _patch_add_masks(
    n_selections: int,
    feature_shape: torch.Size,
    device: torch.device,
) -> torch.Tensor:
    assert len(feature_shape) in (2, 3)
    if len(feature_shape) == 3:
        n_channels, height, width = feature_shape
    else:
        n_channels = 1
        height, width = feature_shape

    mask_width = int(n_selections**0.5)
    assert mask_width * mask_width == n_selections
    assert height % mask_width == 0
    assert width % mask_width == 0
    patch_height = height // mask_width
    patch_width = width // mask_width
    add_masks = torch.zeros(
        n_selections,
        n_channels,
        height,
        width,
        device=device,
        dtype=torch.bool,
    )
    for patch_idx in range(n_selections):
        row = patch_idx // mask_width
        col = patch_idx % mask_width
        add_masks[
            patch_idx,
            :,
            row * patch_height : (row + 1) * patch_height,
            col * patch_width : (col + 1) * patch_width,
        ] = True
    return add_masks.flatten(start_dim=1)


def _selection_add_masks(
    n_selections: int,
    n_features: int,
    n_contexts: int | None,
    feature_shape: torch.Size | None,
    device: torch.device,
) -> torch.Tensor:
    if n_selections == n_features:
        return torch.eye(n_features, device=device, dtype=torch.bool)
    if (
        feature_shape is not None
        and len(feature_shape) in (2, 3)
        and n_selections < n_features
    ):
        return _patch_add_masks(n_selections, feature_shape, device)
    if n_contexts is None:
        msg = "n_contexts must be set when using context selection."
        raise ValueError(msg)
    expected = 1 + (n_features - n_contexts)
    if n_selections != expected:
        msg = (
            f"Got n_sel={n_selections}, expected {expected} "
            f"for n_contexts={n_contexts}."
        )
        raise ValueError(msg)
    add_masks = torch.zeros(
        n_selections, n_features, device=device, dtype=torch.bool
    )
    add_masks[0, :n_contexts] = True
    remaining = n_features - n_contexts
    add_masks[1:, n_contexts:] = torch.eye(
        remaining, device=device, dtype=torch.bool
    )
    return add_masks


class EDDIAFAMethod(AFAMethodBase):
    def __init__(
        self,
        sampler: nn.Module,
        predictor: nn.Module,
        num_classes: int,
        device: torch.device | None = None,
        lambda_threshold: float | None = None,
        selection_costs: torch.Tensor | None = None,
        n_contexts: int | None = None,
        num_mc_samples: int = 128,
        classifier_bundle_path: Path | None = None,
    ) -> None:
        super().__init__(device or torch.device("cpu"))
        self.sampler: nn.Module = sampler
        self.predictor: nn.Module = predictor
        self.num_classes: int = num_classes
        if lambda_threshold is None:
            self.lambda_threshold: float = -math.inf
        else:
            self.lambda_threshold = lambda_threshold
        self._selection_costs: torch.Tensor | None = selection_costs
        self.n_contexts: int | None = n_contexts
        self.num_mc_samples: int = num_mc_samples
        self.classifier: AFAClassifier | None = None
        if classifier_bundle_path is not None:
            self.classifier = load_bundle(
                classifier_bundle_path,
                expect=AFAClassifier,
                device=self._device,
            ).obj

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2 and logits.size(1) > 1:
            return logits.softmax(dim=1)
        probs = logits.sigmoid().view(-1, 1)
        return torch.cat([1 - probs, probs], dim=1)

    def _external_classifier_probs(self, output: torch.Tensor) -> torch.Tensor:
        """
        Convert an external classifier's output to probabilities.

        Follows `AFAClassifier.output_kind` explicitly instead of
        guessing from the values, since EDDI uses the result in a KL
        divergence where negative logits or unnormalized positive
        scores silently produce invalid (NaN or meaningless) scores.
        """
        assert self.classifier is not None
        output_kind = getattr(self.classifier, "output_kind", None)
        if output_kind not in _SUPPORTED_CLASSIFIER_OUTPUT_KINDS:
            msg = (
                "External classifier for EDDI must declare "
                "output_kind as 'logits' or 'probabilities', got "
                f"{output_kind!r}."
            )
            raise ValueError(msg)
        if output_kind == "logits":
            return output.softmax(dim=-1)
        return output

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        masked_features, feature_mask, _label, batch_shape = _flatten_inputs(
            masked_features, feature_mask, label, feature_shape
        )
        batch_size, _n_features = masked_features.shape
        zeros_label = torch.zeros(
            batch_size, self.num_classes, device=self._device
        )
        zeros_mask = torch.zeros(
            batch_size,
            self.num_classes,
            device=self._device,
            dtype=feature_mask.dtype,
        )
        augmented_masked_feature = torch.cat(
            [masked_features, zeros_label], dim=-1
        ).to(self._device)
        augmented_feature_mask = torch.cat(
            [feature_mask, zeros_mask], dim=-1
        ).to(self._device)
        x_rep = augmented_masked_feature.repeat_interleave(
            self.num_mc_samples, dim=0
        )
        m_rep = augmented_feature_mask.repeat_interleave(
            self.num_mc_samples, dim=0
        )

        with torch.no_grad():
            _, _, _, z, _ = self.sampler(x_rep, m_rep)
            logits = self.predictor(z)
            probs = logits.softmax(dim=-1)
            probs = (
                probs.view(batch_size, self.num_mc_samples, -1)
                .transpose(0, 1)
                .contiguous()
            )
            probs_mean = probs.mean(dim=0)
        return probs_mean.view(*batch_shape, self.num_classes)

    @override
    @on_method_device
    def act(  # noqa: PLR0915
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        device = self._device
        masked_features, feature_mask, _label, batch_shape = _flatten_inputs(
            masked_features, feature_mask, label, feature_shape
        )
        batch_size, n_features = masked_features.shape
        n_mc_samples = self.num_mc_samples
        zeros_label = torch.zeros(
            batch_size, self.num_classes, device=self._device
        )
        zeros_mask = torch.zeros(
            batch_size,
            self.num_classes,
            device=self._device,
            dtype=feature_mask.dtype,
        )
        augmented_masked_feature = torch.cat(
            [masked_features, zeros_label], dim=-1
        ).to(self._device)
        augmented_feature_mask = torch.cat(
            [feature_mask, zeros_mask], dim=-1
        ).to(self._device)

        with torch.no_grad():
            x_rep = augmented_masked_feature.repeat_interleave(
                n_mc_samples, dim=0
            )
            m_rep = augmented_feature_mask.repeat_interleave(
                n_mc_samples, dim=0
            )
            _, _, _, z_base, x_full = self.sampler.forward(x_rep, m_rep)
            if self.classifier is None:
                logits_base = self.predictor(z_base)
                probs_base = self._logits_to_probs(logits_base).view(
                    n_mc_samples, batch_size, -1
                )
                base_probs = probs_base.mean(dim=0)
            else:
                base_probs = self._external_classifier_probs(
                    self.classifier(
                        masked_features=masked_features,
                        feature_mask=feature_mask,
                        feature_shape=_flat_feature_shape(n_features),
                    )
                )
        x_full = x_full.view(n_mc_samples, batch_size, -1)[:, :, :n_features]
        missing = ~feature_mask.bool()
        x_filled = (
            masked_features.unsqueeze(0)
            .expand(n_mc_samples, batch_size, n_features)
            .clone()
        )
        x_filled[:, missing] = x_full[:, missing]
        # (S, B, F+num_classes)
        x_full = torch.cat(
            [
                x_filled,
                zeros_label.unsqueeze(0).expand(n_mc_samples, batch_size, -1),
            ],
            dim=-1,
        )
        selection_mask_flat = None
        if selection_mask is not None:
            n_sel = selection_mask.shape[-1]
            selection_mask_flat = selection_mask.view(-1, n_sel).to(device)
        else:
            n_sel = n_features
        add_masks = _selection_add_masks(
            n_selections=n_sel,
            n_features=n_features,
            n_contexts=self.n_contexts,
            feature_shape=feature_shape,
            device=device,
        )

        with torch.no_grad():
            if self.classifier is None:
                # Encoding of x_O plus x_i is the observed set's pointnet sum
                # plus the feature maps of the newly added elements.
                sampler = cast("PartialVAE", self.sampler)
                pointnet = sampler.pointnet
                base = pointnet(
                    augmented_masked_feature, augmented_feature_mask
                )
                maps = pointnet.element_maps(x_full)[..., :n_features, :]
                maps = maps * missing.unsqueeze(-1)
                sums = base.unsqueeze(1) + torch.einsum(
                    "if,sbfh->sbih", add_masks.to(maps.dtype), maps
                )
                encoding = sampler.encoder(sums.flatten(end_dim=-2))
                mu, logvar = encoding.chunk(2, dim=-1)
                z_all = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
                logits_all = self.predictor(z_all)
                preds_all = self._logits_to_probs(logits_all).view(
                    n_mc_samples, batch_size * n_sel, -1
                )
            else:
                mask_tests = (
                    feature_mask.bool().unsqueeze(1) | add_masks.unsqueeze(0)
                ).to(feature_mask.dtype)
                x_masks_raw = x_full[:, :, None, :n_features] * mask_tests
                preds_flat = self._external_classifier_probs(
                    self.classifier(
                        masked_features=x_masks_raw.flatten(end_dim=-2),
                        feature_mask=mask_tests.expand(
                            n_mc_samples, -1, -1, -1
                        ).flatten(end_dim=-2),
                        feature_shape=_flat_feature_shape(n_features),
                    )
                )
                preds_all = preds_flat.view(
                    n_mc_samples, batch_size * n_sel, -1
                )

        # S: num_mc_samples
        num_samples, _batch_selections, C = preds_all.shape
        # 1/n Σ p(y|x_s, x_i^j)
        base_probs_flat = (
            base_probs.unsqueeze(1)
            .expand(batch_size, n_sel, C)
            .reshape(1, batch_size * n_sel, C)
            .expand(num_samples, batch_size * n_sel, C)
        )
        # mean_preds = preds_all.mean(dim=0)
        # KL(p_s || mean), (S, B*n_sel)
        kl_all = (
            preds_all
            * ((preds_all + 1e-6).log() - (base_probs_flat + 1e-6).log())
        ).sum(dim=-1)
        kl_mean_flat = kl_all.mean(dim=0)

        scores = kl_mean_flat.view(batch_size, n_sel)
        # avoid choosing the already masked features
        if selection_mask_flat is not None:
            assert scores.shape == selection_mask_flat.shape
            scores = scores.masked_fill(
                selection_mask_flat.bool(), float("-inf")
            )
        else:
            assert scores.shape == feature_mask.shape
            scores = scores.masked_fill(feature_mask.bool(), float("-inf"))
        if self._selection_costs is not None:
            costs = self._selection_costs.to(self._device)
            costs = torch.clamp(costs, min=1e-12)
            scores = scores / costs.unsqueeze(0)
        # best_scores, best_idx = scores.max(dim=1)
        best_scores = scores.max(dim=1).values
        is_best = scores == best_scores.unsqueeze(1)
        tie_weights = is_best.float()
        best_idx = torch.multinomial(tie_weights, num_samples=1).squeeze(1)
        lam = self.lambda_threshold
        stop_mask = best_scores < lam
        stop_mask = stop_mask | (best_scores < -1e5)
        selections = (best_idx + 1).to(torch.long).unsqueeze(-1)
        selections = selections.masked_fill(stop_mask.unsqueeze(-1), 0)
        return selections.view(*batch_shape, 1)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(
            str(path / "model.pt"), map_location=device, weights_only=False
        )
        sampler = checkpoint["sampler"]
        predictor = checkpoint["predictor"]
        num_classes = checkpoint["num_classes"]
        n_contexts = checkpoint.get("n_contexts", None)
        lambda_threshold = checkpoint.get("lambda_threshold", None)
        selection_costs = checkpoint.get("selection_costs", None)
        feature_costs = checkpoint.get("feature_costs", None)
        if selection_costs is not None:
            selection_costs = selection_costs.to(device)
        elif feature_costs is not None:
            selection_costs = feature_costs.to(device)

        method = cls(
            sampler=sampler,
            predictor=predictor,
            num_classes=num_classes,
            device=device,
            lambda_threshold=lambda_threshold,
            selection_costs=selection_costs,
            n_contexts=n_contexts,
        )
        classifier = checkpoint.get("classifier", None)
        if classifier is not None:
            classifier = cast("AFAClassifier", cast("object", classifier))
            classifier = classifier.to(device)
            method.classifier = classifier
        return method.to(device)

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sampler": self.sampler,
                "predictor": self.predictor,
                "classifier": self.classifier,
                "num_classes": self.num_classes,
                "lambda_threshold": float(self.lambda_threshold),
                "selection_costs": (
                    self._selection_costs.detach().cpu()
                    if self._selection_costs is not None
                    else None
                ),
                "n_contexts": self.n_contexts,
            },
            str(path / "model.pt"),
        )

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return "probabilities"

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @property
    @override
    def cost_param(self) -> float | None:
        return float(self.lambda_threshold)

    @override
    def set_cost_param(self, cost_param: float) -> None:
        self.lambda_threshold = cost_param
