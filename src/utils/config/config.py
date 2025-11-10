'''
DATASET
'''
SIZE = 512
CLASS_PIXEL = 38
NUM_CLASSES = 1

'''
Training
'''
DATA_DIR = r"./data"
TRAIN_PATH = r"uavid_train/seq1"
VALID_PATH = r"uavid_val/seq16"

BATCH = 4
EPOCHS = 30
LR = 1e-4
LR_DROP_EPOCH = 8
LR_FACTOR = 0.5
LR_STEP = 3
LOSS = r"bce_dice" # can be: bce, dice, bce_dice
MODEL = r"deeplabv3_resnet101"
import torch; DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Inference
'''
#TODO: soon