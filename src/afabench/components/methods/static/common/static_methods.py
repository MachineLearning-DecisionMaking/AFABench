from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Literal, Self, override

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import RelaxedOneHotCategorical

from afabench.components.methods.common import AFAMethodBase
from afabench.components.methods.static.common.utils import restore_parameters
from afabench.core.types import (
    AFAAction,
    ClassifierOutputKind,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)

type BatchLoader = Iterable[tuple[torch.Tensor, torch.Tensor]]


class ConcreteMask(nn.Module):
    """For differentiable global feature selection."""

    def __init__(
        self,
        num_features: int,
        num_select: int,
        append: bool = False,  # noqa: FBT002
        gamma: float = 0.2,
    ):
        super().__init__()
        self.logits: nn.Parameter = nn.Parameter(
            torch.randn(num_select, num_features, dtype=torch.float32)
        )
        self.append: bool = append
        self.gamma: float = gamma

    @override
    def forward(self, x: torch.Tensor, temp: float) -> torch.Tensor:
        dist = RelaxedOneHotCategorical(
            temp, logits=self.logits / self.gamma, validate_args=False
        )
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out


class ConcreteMask2d(nn.Module):
    def __init__(
        self,
        width: int,
        patch_size: int,
        num_select: int,
        gamma: float = 0.2,
    ):
        super().__init__()
        self.logits: nn.Parameter = nn.Parameter(
            torch.randn(num_select, width**2, dtype=torch.float32)
        )
        self.upsample: torch.nn.Upsample = torch.nn.Upsample(
            scale_factor=patch_size
        )
        self.width: int = width
        self.patch_size: int = patch_size
        self.gamma: float = gamma

    @override
    def forward(self, x: torch.Tensor, temp: float) -> torch.Tensor:
        dist = RelaxedOneHotCategorical(
            temp, logits=self.logits / self.gamma, validate_args=False
        )
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        m = self.upsample(m.reshape(-1, 1, self.width, self.width))
        out = x * m
        return out


class DifferentiableSelector(nn.Module):
    """Differentiable global feature selection."""

    def __init__(
        self,
        model: nn.Module,
        selector_layer: ConcreteMask | ConcreteMask2d,
    ):
        super().__init__()
        self.model: nn.Module = model
        self.selector_layer: ConcreteMask | ConcreteMask2d = selector_layer

    def _to_class_indices(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim >= 2:
            return y.argmax(dim=-1).long()
        return y.long()

    def fit(  # noqa: C901, PLR0912, PLR0915
        self,
        train_loader: BatchLoader,
        val_loader: BatchLoader,
        lr: float,
        nepochs: int,
        loss_fn: nn.Module,
        val_loss_fn: nn.Module | None = None,
        val_loss_mode: Literal["min", "max"] = "min",
        factor: float = 0.2,
        patience: int = 2,
        min_lr: float = 1e-5,
        early_stopping_epochs: int | None = None,
        start_temp: float = 10.0,
        end_temp: float = 0.01,
        temp_steps: int = 10,
        verbose: bool = True,  # noqa: FBT002
        metric_logger: Callable[[dict[str, float]], None] | None = None,
        metric_prefix: str = "static_selector",
    ) -> None:
        """Train model to perform global feature selection."""
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1

        # More setup.
        model = self.model
        selector_layer = self.selector_layer
        device = next(model.parameters()).device
        val_loss_fn.to(device)

        # For tracking best models with zero temperature.
        best_val: torch.Tensor | None = None
        best_zerotemp_model: nn.Module | None = None
        best_zerotemp_selector: ConcreteMask | ConcreteMask2d | None = None

        # Train separately with each temperature.
        total_epochs = 0
        for temp in np.geomspace(start_temp, end_temp, temp_steps):
            if verbose:
                print(f"Starting training with temp = {temp:.4f}\n")

            # Set up optimizer and lr scheduler.
            opt = optim.Adam(
                list(model.parameters()) + list(selector_layer.parameters()),
                lr=lr,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode=val_loss_mode,
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

            # For tracking best models and early stopping.
            best_model = deepcopy(model)
            best_selector = deepcopy(selector_layer)
            num_bad_epochs = 0
            epoch = 0
            for epoch in range(nepochs):
                # Switch model to training mode.
                model.train()
                epoch_train_loss = torch.zeros((), device=device)
                train_batches = 0

                for x_batch, y_batch in train_loader:
                    train_batches += 1
                    # Move to device.
                    x = x_batch.to(device)
                    y = self._to_class_indices(y_batch).to(device)

                    # Select features and make prediction.
                    x_masked = selector_layer(x, temp)
                    pred = model(x_masked)

                    # Calculate loss.
                    loss = loss_fn(pred, y)

                    # Take gradient step.
                    loss.backward()
                    opt.step()
                    model.zero_grad()
                    selector_layer.zero_grad()
                    epoch_train_loss += loss.detach()

                train_loss = epoch_train_loss.item() / train_batches

                # Reinitialize logits as necessary.
                logits = selector_layer.logits
                argmax = logits.argmax(dim=1).cpu().data.numpy()
                selected = []
                for i, ind in enumerate(argmax):
                    if ind in selected:
                        logits.data[i] = 0
                    else:
                        selected.append(ind)

                # Calculate validation loss.
                model.eval()
                with torch.no_grad():
                    # For mean loss.
                    pred_list = []
                    hard_pred_list = []
                    label_list = []

                    for x_batch, y_batch in val_loader:
                        # Move to device.
                        x = x_batch.to(device)
                        y = self._to_class_indices(y_batch).to(device)

                        # Evaluate model with soft sample.
                        x_masked = selector_layer(x, temp)
                        pred = model(x_masked)

                        # Evaluate model with hard sample.
                        x_masked = selector_layer(x, 1e-6)
                        hard_pred = model(x_masked)

                        # Append to lists.
                        pred_list.append(pred)
                        hard_pred_list.append(hard_pred)
                        label_list.append(y)

                    # Calculate mean loss.
                    pred = torch.cat(pred_list, 0)
                    hard_pred = torch.cat(hard_pred_list, 0)
                    y = torch.cat(label_list, 0)
                    val_loss = val_loss_fn(pred, y)
                    val_hard_loss = val_loss_fn(hard_pred, y)

                # Print progress.
                if verbose:
                    print(
                        f"{'-' * 8}Epoch {epoch + 1} ({
                            epoch + 1 + total_epochs
                        } total){'-' * 8}"
                    )
                    print(
                        f"Val loss = {val_loss:.4f}, Zero-temp loss = {val_hard_loss:.4f}\n"
                    )

                if metric_logger is not None:
                    metric_logger(
                        {
                            f"{metric_prefix}/epoch": float(
                                epoch + 1 + total_epochs
                            ),
                            f"{metric_prefix}/temperature": float(temp),
                            f"{metric_prefix}/train_loss": float(train_loss),
                            f"{metric_prefix}/val_loss": float(
                                val_loss.mean().item()
                            ),
                            f"{metric_prefix}/val_hard_loss": float(
                                val_hard_loss.mean().item()
                            ),
                        }
                    )

                # Update scheduler.
                scheduler.step(val_loss)

                # See if best model.
                if val_loss == scheduler.best:
                    best_model = deepcopy(model)
                    best_selector = deepcopy(selector_layer)
                    num_bad_epochs = 0
                else:
                    num_bad_epochs += 1

                # Check if best model with zero temperature.
                if (
                    (best_val is None)
                    or (val_loss_mode == "min" and val_hard_loss < best_val)
                    or (val_loss_mode == "max" and val_hard_loss > best_val)
                ):
                    best_val = val_hard_loss
                    best_zerotemp_model = deepcopy(model)
                    best_zerotemp_selector = deepcopy(selector_layer)

                # Early stopping.
                if num_bad_epochs > early_stopping_epochs:
                    break

            # Update total epoch count.
            if verbose:
                print(f"Stopping temp = {temp:.4f} at epoch {epoch + 1}\n")
            total_epochs += epoch + 1

            # Copy parameters from best model.
            restore_parameters(model, best_model)
            restore_parameters(selector_layer, best_selector)

        # Copy parameters from best model with zero temperature.
        assert best_zerotemp_model is not None
        assert best_zerotemp_selector is not None
        restore_parameters(model, best_zerotemp_model)
        restore_parameters(selector_layer, best_zerotemp_selector)


class StaticBaseMethod(AFAMethodBase):
    def __init__(
        self,
        selected_history: dict[int, list[int]],
        predictors: dict[int, nn.Module],
        device: torch.device,
        image_size: int | None = None,
        patch_size: int | None = None,
    ):
        super().__init__(device)
        self.selected_history: dict[int, list[int]] = selected_history
        self.predictors: dict[int, nn.Module] = {
            b: m.to(device) for b, m in predictors.items()
        }
        self.image_size: int | None = image_size
        self.patch_size: int | None = patch_size

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        if masked_features.ndim == 4:
            if self.patch_size is None:
                message = "patch_size missing from method; retrain/save with patch_size"
                raise RuntimeError(message)
            fm = feature_mask
            if fm.ndim == 4:
                fm = fm.any(dim=1)
            elif fm.ndim != 3:
                message = f"Unexpected image feature_mask shape: {feature_mask.shape}"
                raise RuntimeError(message)

            B, H, W = fm.shape
            p = self.patch_size
            if H % p != 0 or W % p != 0:
                message = (
                    f"Image size {(H, W)} not divisible by patch_size={p}"
                )
                raise RuntimeError(message)
            gh, gw = H // p, W // p
            patch_mask = fm.reshape(B, gh, p, gw, p).all(dim=(2, 4))
            counts = patch_mask.reshape(B, -1).sum(dim=1)
        else:
            counts = feature_mask.sum(dim=1)
        if not (counts == counts[0]).all():
            message = "mixed budgets in batch"
            raise RuntimeError(message)
        b = int(counts[0].item())
        if b == 0:
            # uniform prior over classes
            if masked_features.ndim == 4:
                assert self.image_size is not None
                n_classes = next(iter(self.predictors.values()))(
                    torch.zeros(
                        (1, 3, self.image_size, self.image_size),
                        device=self._device,
                    )
                ).shape[-1]
            else:
                n_classes = next(iter(self.predictors.values()))(
                    torch.zeros((1, 1), device=self._device)
                ).shape[-1]
            probs = torch.full(
                (masked_features.size(0), n_classes),
                1.0 / n_classes,
                device=self._device,
            )
            return probs
        if masked_features.ndim == 4:
            logits = self.predictors[b](masked_features.to(self._device))
        else:
            cols = self.selected_history[b]
            x_sel = masked_features[:, cols].to(self._device)
            logits = self.predictors[b](x_sel)
        return logits.softmax(dim=-1)

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        if selection_mask is not None:
            # for image datasets
            counts = selection_mask.sum(dim=1)
        else:
            # for tabular datasets
            counts = feature_mask.sum(dim=1)
        if not (counts == counts[0]).all():
            message = "mixed budgets in batch"
            raise RuntimeError(message)
        b = int(counts[0].item())
        if (b + 1) not in self.selected_history:
            return torch.zeros(
                (masked_features.size(0), 1),
                dtype=torch.long,
                device=self._device,
            )

        if selection_mask is not None:
            mask0 = selection_mask[0]
        else:
            mask0 = feature_mask[0]
        for idx in self.selected_history[b + 1]:
            if mask0[idx] == 0:
                choice = idx + 1
                return torch.full(
                    (masked_features.size(0), 1),
                    fill_value=choice,
                    dtype=torch.long,
                    device=self._device,
                )

        return torch.zeros(
            (masked_features.size(0), 1), dtype=torch.long, device=self._device
        )

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "selected_history": self.selected_history,
                "image_size": self.image_size,
                "patch_size": self.patch_size,
            },
            path / "selected.pt",
        )
        for b, mdl in self.predictors.items():
            torch.save(mdl, path / f"predictor_b{b}.pt")

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        data = torch.load(
            path / "selected.pt", weights_only=False, map_location="cpu"
        )
        hist = data["selected_history"]
        image_size = data.get("image_size", None)
        patch_size = data.get("patch_size", None)

        preds: dict[int, nn.Module] = {}
        for b in hist:
            model = torch.load(
                path / f"predictor_b{b}.pt",
                weights_only=False,
                map_location=device,
            )
            preds[b] = model.to(device)

        return cls(
            selected_history=hist,
            predictors=preds,
            image_size=image_size,
            patch_size=patch_size,
            device=device,
        )

    @override
    def to(self, device: torch.device) -> Self:
        # The predictors sit in a dict, which the base does not look into
        super().to(device)
        for b in self.predictors:
            self.predictors[b] = self.predictors[b].to(device)
        return self

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return "probabilities"

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True
