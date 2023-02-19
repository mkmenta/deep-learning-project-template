from typing import Optional
import numpy as np
import pytorch_lightning as pl


class Accuracy:
    """Accuracy logger."""

    def __init__(self, pl_module: pl.LightningModule):
        """Initialize"""
        self._pl_module = pl_module
        self.reset()

    def reset(self):
        """Reset accumulators."""
        self._n_correct = 0
        self._n_samples = 0

    def update(self, confusion_matrix: np.ndarray):
        """Update values."""
        self._n_correct += np.diag(confusion_matrix).sum().item()
        self._n_samples += confusion_matrix.sum().item()

    def log(self, prefix: Optional[str] = None):
        """Log metric."""
        prefix = f"{prefix}_" if prefix else ""
        self._pl_module.log(f"{prefix}accuracy", self._n_correct / self._n_samples)
