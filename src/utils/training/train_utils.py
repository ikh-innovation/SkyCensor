import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

sys.path.append('./')

from src.utils.config.config import *
from src.utils.dataset.dataset import SkyCensorDataset

class TrainUtils:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

    def get_loaders(self, size, batch):
        """
        Returns train and validation DataLoaders.
        """
        train_root = os.path.join(self.data_dir, TRAIN_PATH)
        val_root = os.path.join(self.data_dir, VALID_PATH)

        train_dataset = SkyCensorDataset(train_root, size=size)
        val_dataset = SkyCensorDataset(val_root, size=size)

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

        return train_loader, val_loader

    def get_model(self, num_classes=1):
        """
        Returns a DeepLabV3-ResNet101 model with output layer replaced for segmentation.
        """
        model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
        model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        return model

    @staticmethod
    def dice_loss(pred, target, smooth=1.0):
        """
        Computes Dice Loss for binary segmentation.
        """
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * inter + smooth) / (union + smooth)
        return 1 - dice.mean()
