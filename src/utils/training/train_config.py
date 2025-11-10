import sys

sys.path.append('./')

from src.utils.config.config import *
from src.utils.training.factory_model import ModelFactory
from src.utils.training.factory_loss import LossFactory
from src.utils.training.factory_dataloader import DataLoaderFactory

class TrainConfig:
    def __init__(self,
                 model_name,
                 num_classes,
                 image_size,
                 batch_size,
                 loss_name):

        self.model_name = model_name
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.loss_name = loss_name

    def build(self):
        loaders = DataLoaderFactory().build(size=self.image_size, batch=self.batch_size)
        model = ModelFactory(self.model_name, self.num_classes).build()
        loss_fn = LossFactory().build(self.loss_name)
        return loaders, model, loss_fn
    
if __name__ == '__main__':

    config = TrainConfig(
    model_name=MODEL,
    num_classes=NUM_CLASSES,
    image_size=SIZE,
    batch_size=BATCH,
    loss_name=LOSS)

    (train_loader, val_loader), model, criterion = config.build()

    [print([i for i in val_loader])]
