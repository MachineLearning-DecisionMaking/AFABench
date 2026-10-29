from typing import final

import lightning as pl
from torch import Tensor


def should_log_step(module: pl.LightningModule) -> bool:
    """
    Whether Lightning writes step-level metrics on this training step.

    Step metrics logged on other steps are discarded, so skipping them leaves
    the logger output unchanged while avoiding a host sync per call.
    """
    trainer = module._trainer  # noqa: SLF001
    if trainer is None:
        return True
    return trainer._logger_connector.should_update_logs  # noqa: SLF001


@final
class EpochMeans:
    """
    Batch-size weighted means of per-batch metrics.

    Logging these once per epoch gives the same values as calling
    `self.log` in every validation step (Lightning's default `on_epoch` mean),
    with one host sync per metric instead of one per batch.
    """

    def __init__(self) -> None:
        self.sums: dict[str, Tensor] = {}
        self.count = 0
        self.weight = 0

    def start_batch(self, batch_size: int) -> "EpochMeans":
        self.weight = batch_size
        self.count += batch_size
        return self

    def add(self, name: str, value: Tensor) -> None:
        weighted = value.detach() * self.weight
        self.sums[name] = (
            self.sums[name] + weighted if name in self.sums else weighted
        )

    def log(self, module: pl.LightningModule) -> None:
        for name, total in self.sums.items():
            module.log(name, total / self.count)
        self.sums, self.count = {}, 0


def epoch_means(module: pl.LightningModule) -> EpochMeans:
    means = module.__dict__.get("_epoch_means")
    if means is None:
        means = EpochMeans()
        module.__dict__["_epoch_means"] = means
    return means
