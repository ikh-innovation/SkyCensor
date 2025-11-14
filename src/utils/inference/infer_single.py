import os
import cv2
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import models
import torchvision.transforms as T

sys.path.append('./')

from src.utils.config.config import DEVICE
class InferSingleCore:
    """Single-class segmentation inference."""
    def __init__(self, weights_path, num_classes=1, device=DEVICE, visualize=False):
        self.device = device
        self.visualize_enabled = visualize
        self.model = models.segmentation.deeplabv3_resnet101(weights="DEFAULT")
        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def predict(self, image, size=(512, 512)):
        """Return predicted mask."""
        tf = T.Compose([T.Resize(size), T.ToTensor()])
        x = tf(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)["out"]
        return (torch.sigmoid(out) > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255

    def visualize(self, image_cv, mask):
        """Return visualization and optionally show it."""
        overlay = image_cv.copy()
        red = np.zeros_like(image_cv)
        red[:,:,2] = mask
        cv2.addWeighted(red,0.5,overlay,0.5,0,overlay)
        blur = cv2.GaussianBlur(image_cv,(21,21),0)
        masked = image_cv.copy()
        masked[mask>0] = blur[mask>0]
        vis = cv2.hconcat([image_cv, overlay, masked])

        if self.visualize_enabled:
            cv2.imshow("vis", vis)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()

        return vis

    def run(self, image_path):
        """Infer on one image."""
        image = Image.open(image_path).convert("RGB")
        image_cv = cv2.cvtColor(np.array(image.resize((512,512))), cv2.COLOR_RGB2BGR)
        mask = self.predict(image)
        return self.visualize(image_cv, mask)

    def run_all_in_dir(self, root_dir):
        """Run inference on all images in directory."""
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith(("png","jpg","jpeg")):
                    p = os.path.join(root, f)
                    self.run(p)


if __name__ == "__main__":
    model_path = "./runs/exp_0/best_model.pth"
    test_root = "./data/uavid_test"

    infer = InferSingleCore(model_path, visualize=True)
    infer.run_all_in_dir(test_root)

    # infer.run('./image_path.png')
