from typing import Optional

from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_classes = 10

    def setup(self, stage: Optional[str] = None):
        # Assign Train/val split(s) for use in Dataloaders
        if stage in (None, "fit"):
            mnist_full = MNIST(self.data_dir, train=True, download=True, transform=transforms.ToTensor())
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            self.mnist_test = MNIST(self.data_dir, train=False, download=True, transform=transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
