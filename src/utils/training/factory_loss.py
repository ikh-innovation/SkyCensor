import sys
import torch
import torch.nn as nn

sys.path.append('./')

from src.utils.config.config import *
    
class LossFactory:
    @staticmethod
    def dice_loss(pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * inter + smooth) / (union + smooth)
        return 1 - dice.mean()

    def build(self, loss_name=LOSS):
        if loss_name == "bce":
            return nn.BCEWithLogitsLoss()

        if loss_name == "dice":
            return lambda y_pred, y_true: self.dice_loss(y_pred, y_true)

        if loss_name == "bce_dice":
            return lambda y_pred, y_true: \
                0.5 * nn.BCEWithLogitsLoss()(y_pred, y_true) + 0.5 * self.dice_loss(y_pred, y_true)

        raise ValueError(f"Unknown loss: {loss_name}")