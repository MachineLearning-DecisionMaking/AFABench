from typing import override

import lightning as pl
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from afabench.fit.step_logging import epoch_means
from afabench.fit.tensor_batches import TensorBatchDataset, passthrough_batch


class Model(pl.LightningModule):
    def __init__(self, *, accumulate: bool):
        super().__init__()
        self.accumulate = accumulate
        self.linear = nn.Linear(3, 1)

    @override
    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        x, y = batch
        return ((self.linear(x).squeeze(-1) - y) ** 2).mean()

    @override
    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        x, y = batch
        loss = ((self.linear(x).squeeze(-1) - y) ** 2).mean()
        if self.accumulate:
            epoch_means(self).start_batch(len(x)).add("val_loss", loss)
        else:
            self.log("val_loss", loss)

    @override
    def on_validation_epoch_end(self) -> None:
        if self.accumulate:
            epoch_means(self).log(self)

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_epoch_means_match_per_step_logging() -> None:
    x, y = torch.randn(23, 3), torch.randn(23)
    results = []
    for accumulate in (False, True):
        torch.manual_seed(0)
        model = Model(accumulate=accumulate)
        trainer = pl.Trainer(
            max_epochs=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="cpu",
        )
        loader = DataLoader(
            TensorBatchDataset(x, y),
            batch_size=5,
            collate_fn=passthrough_batch,
        )
        trainer.fit(model, loader, loader)
        results.append(trainer.callback_metrics["val_loss"])
    assert torch.allclose(results[0], results[1])
