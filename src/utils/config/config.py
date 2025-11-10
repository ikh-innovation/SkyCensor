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
LOSS = r"bce_dice" # can be: bce, dice, bce_dice
MODEL = r"deeplabv3_resnet101"

'''
Inference
'''