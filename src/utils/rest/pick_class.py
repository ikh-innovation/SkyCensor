import os
import sys
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append('./')

class ClassPicker:

    def __init__(self, data_dir: str):
        '''
        Pick a random sample from dataset,
        browse through different class-assigned
        pixels and make sure visually we pick our
        desired class.
        '''
        self.data_dir = data_dir
        self.images_dir = os.path.join(self.data_dir, 'Images')
        self.labels_dir = os.path.join(self.data_dir, 'Labels')

        self.image_files = sorted(os.listdir(self.images_dir))
        self.label_files = sorted(os.listdir(self.labels_dir))
        pass

    def _pick_random_idx(self):
        '''
        Out of the image list, pick a random
        index to check visually later.
        '''
        return random.randint(0, len(self.images_dir)-1)

    def run(self):
        '''
        Pick a random image, show available class-assigned
        pixel values and visualize binary mask.
        '''
        idx = self._pick_random_idx()

        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.labels_dir, self.label_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        unique_classes = np.unique(np.array(mask))

        class_to_keep = int(input(f"Enter class ID to visualize from available: {unique_classes} =>"))

        binary_mask = (np.array(mask) == class_to_keep).astype(np.uint8)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(binary_mask, cmap='gray')
        plt.title(f"Class {class_to_keep} Mask")
        plt.axis('off')

        plt.show()


if __name__ == '__main__':

    data_dir = r'./data/uavid_train/seq1'
    class_picker = ClassPicker(data_dir=data_dir)
    class_picker.run() # 38 for buildings :))