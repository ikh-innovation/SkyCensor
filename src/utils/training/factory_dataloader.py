import os
import sys
from torch.utils.data import DataLoader

sys.path.append('./')

from src.utils.config.config import *
from src.utils.dataset.dataset import SkyCensorDataset

class DataLoaderFactory:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

    def build(self, size, batch):
        train_root = os.path.join(self.data_dir, TRAIN_PATH)
        val_root = os.path.join(self.data_dir, VALID_PATH)

        train_dataset = SkyCensorDataset(train_root, size=size)
        val_dataset = SkyCensorDataset(val_root, size=size)

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

        return train_loader, val_loader