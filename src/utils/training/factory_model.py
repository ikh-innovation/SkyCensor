import os
import sys
import torch
import torch.nn as nn
from torchvision import models

sys.path.append('./')

from src.utils.config.config import *
    
class ModelFactory:
    def __init__(self, model_name=MODEL, num_classes=NUM_CLASSES):
        self.model_name = model_name
        self.num_classes = num_classes

    def build(self):
        if self.model_name == "deeplabv3_resnet101":
            model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
            model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
            return model

        raise ValueError(f"Unknown model: {self.model_name}")