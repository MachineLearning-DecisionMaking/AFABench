from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Literal, override

import torch
from torch import nn, optim

from afabench.components.methods.static.common.utils import restore_parameters

type BatchLoader = Iterable[tuple[torch.Tensor, torch.Tensor]]
type LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseModel(nn.Module):
    """Base model, no missing features."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model: nn.Module = model

    def _to_class_indices(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim >= 2:
            return y.argmax(dim=-1).long()
        return y.long()

    def fit(  # noqa: C901, PLR0915
        self,
        train_loader: BatchLoader,
        val_loader: BatchLoader,
        lr: float,
        nepochs: int,
        loss_fn: LossFunction,
        val_loss_fn: LossFunction | None = None,
        val_loss_mode: Literal["min", "max"] | None = None,
        factor: float = 0.2,
        patience: int = 2,
        min_lr: float = 1e-6,
        early_stopping_epochs: int | None = None,
        verbose: bool = True,  # noqa: FBT002
        metric_logger: Callable[[dict[str, float]], None] | None = None,
        metric_prefix: str = "static_classifier",
    ) -> None:
        """Train model."""
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = "min"
        elif val_loss_mode is None:
            message = (
                "must specify val_loss_mode when val_loss_fn is specified"
            )
            raise ValueError(message)

        # Set up optimizer and lr scheduler.
        model = self.model
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=val_loss_mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        # For tracking best model and early stopping.
        best_model: nn.Module | None = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1

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

                # Calculate loss.
                pred = model(x)
                loss = loss_fn(pred, y)

                # Take gradient step.
                loss.backward()
                opt.step()
                model.zero_grad()
                epoch_train_loss += loss.detach()

            train_loss = epoch_train_loss.item() / train_batches

            # Calculate validation loss.
            model.eval()
            with torch.no_grad():
                # For mean loss.
                pred_list = []
                label_list = []

                for x_batch, y_batch in val_loader:
                    # Move to device.
                    x = x_batch.to(device)
                    y = self._to_class_indices(y_batch).to(device)

                    # Calculate prediction.
                    pred = model(x)
                    pred_list.append(pred)
                    label_list.append(y)

                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_loss = val_loss_fn(pred, y).item()

            # Print progress.
            if verbose:
                print(f"{'-' * 8}Epoch {epoch + 1}{'-' * 8}")
                print(f"Val loss = {val_loss:.4f}\n")

            if metric_logger is not None:
                metric_logger(
                    {
                        f"{metric_prefix}/epoch": float(epoch + 1),
                        f"{metric_prefix}/train_loss": float(train_loss),
                        f"{metric_prefix}/val_loss": float(val_loss),
                    }
                )

            # Update scheduler.
            scheduler.step(val_loss)

            # Check if best model.
            if val_loss == scheduler.best:
                best_model = deepcopy(model)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            # Early stopping.
            if num_bad_epochs > early_stopping_epochs:
                if verbose:
                    print(f"Stopping early at epoch {epoch + 1}")
                break

        # Copy parameters from best model.
        assert best_model is not None
        restore_parameters(model, best_model)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate model prediction."""
        return self.model(x)
