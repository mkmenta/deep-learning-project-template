import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from metrics.accuracy import Accuracy
from sklearn.metrics import confusion_matrix


class Solver(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model

        # Make hparams will available in self.hparams
        self.save_hyperparameters(hparams)

        # Metrics
        self.metrics = [
            Accuracy(self)
        ]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=False)

        # WARNING! everything returned is accumulated for the training_epoch_end function so...
        # ONLY return essential variables!
        return {
            'loss': loss  # Necessary for the training
        }

    def eval_step(self, batch, batch_idx, key: str):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        # Log
        self.log(f"{key}_loss", loss.item(), on_step=False, on_epoch=True)

        # Compute confusion matrix
        cm = confusion_matrix(y_hat.argmax(1).cpu(), y.cpu(), labels=list(range(self.trainer.datamodule.n_classes)),
                              normalize=None)

        # Update metrics
        for metric in self.metrics:
            metric.update(confusion_matrix=cm)

    def eval_epoch_end(self, outputs, key: str):
        for metric in self.metrics:
            metric.log(prefix=key)
            metric.reset()

    def validation_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, key='val')

        # WARNING! everything returned is accumulated for the validation_epoch_end function so...
        # ONLY return essential variables!
        return None

    def validation_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, key='val')

    def test_step(self, batch, batch_idx):
        self.eval_step(batch, batch_idx, key='test')

        # WARNING! everything returned is accumulated for the test_epoch_end function so...
        # ONLY return essential variables!
        return None

    def test_epoch_end(self, outputs):
        self.eval_epoch_end(outputs, key='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
        return [optimizer], [scheduler]
