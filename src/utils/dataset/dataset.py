import os
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

sys.path.append('./')

from src.utils.config.config import *

class SkyCensorDataset(Dataset):

    def __init__(self, root_dir, size):
        '''
        Init Dataset class
        '''
        self.root_dir = root_dir
        self.size = size
        self.class_to_keep = CLASS_PIXEL

        self.image_files = None
        self.mask_files = None

        self.transform = T.Resize((size, size))

        self._prep()
        pass

    def _prep(self):
        '''
        Prepare paths for images and masks.
        '''
        image_dir = os.path.join(self.root_dir, "Images")
        masks_dir = os.path.join(self.root_dir, "Labels")

        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

    def __len__(self):
       '''
       Dataset length
       '''
       return len(self.image_files)

    def __getitem__(self, index):
        '''
        Get image & mask tensors for training.
        '''
        image = Image.open(f"{self.root_dir}/Images/{self.image_files[index]}").convert("RGB")
        mask = Image.open(f"{self.root_dir}/Labels/{self.mask_files[index]}").convert("RGB")

        mask_gray =  np.array(mask.convert("L"))
        binary_mask = (mask_gray == self.class_to_keep).astype(np.uint8) * 255
        mask_pil = Image.fromarray(binary_mask, mode="L")

        image = self.transform(image)
        mask_pil = self.transform(mask_pil)

        image_tensor = T.ToTensor()(image)
        mask_tensor = T.ToTensor()(mask_pil)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor

if __name__ == "__main__":

    root_dir = "./data/uavid_train/seq1"

    dataset = SkyCensorDataset(root_dir, size=SIZE)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f"Dataset size: {len(dataset)}")

    for i, (images, masks) in enumerate(dataloader):
        print(f"Batch {i+1}")
        print(f"Images shape: {images.shape}")  
        print(f"Masks shape: {masks.shape}")   
