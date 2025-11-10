import sys
import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.append('./')
from src.utils.config.config import *
from src.utils.training.train_config import TrainConfig

class Trainer:
    """
    Trainer class for segmentation models with mixed precision, LR scheduling,
    checkpointing, and results logging.
    """

    def __init__(self, config: TrainConfig,
                 device: torch.device = DEVICE,
                 num_epochs: int = EPOCHS,
                 lr: float = LR,
                 lr_drop_epoch: int = LR_DROP_EPOCH,
                 lr_drop_factor: float = LR_FACTOR,
                 lr_step: int = LR_STEP):
        self.device = device
        print(f"Using device: {self.device}")

        (self.train_loader, self.val_loader), self.model, self.criterion = config.build()
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr_drop_epoch = lr_drop_epoch
        self.scheduler = StepLR(self.optimizer, step_size=lr_step, gamma=lr_drop_factor)

        self.scaler = torch.cuda.amp.GradScaler()
        self.num_epochs = num_epochs
        self.best_iou = 0.0

        self.train_losses = []
        self.val_losses = []
        self.val_ious = []

        self.save_dir = self.get_exp_dir(base_dir="runs", prefix="exp_")
        os.makedirs(self.save_dir, exist_ok=True)

        self.model_path = os.path.join(self.save_dir, "best_model.pth")
        self.results_path = os.path.join(self.save_dir, "train_results.json")
        self.logfile_path = os.path.join(self.save_dir, "train_config.json")

        self.save_config(config)

    def get_exp_dir(self, base_dir: str = "runs", prefix: str = "exp_") -> str:
        """
        Create sequential experiment directory: exp_0, exp_1, etc.
        """
        os.makedirs(base_dir, exist_ok=True)
        existing = [d for d in os.listdir(base_dir)
                    if d.startswith(prefix) and os.path.isdir(os.path.join(base_dir, d))]
        idx = len(existing)
        return os.path.join(base_dir, f"{prefix}{idx}")

    def save_config(self, config: TrainConfig):
        cfg_dict = {
            "MODEL": config.model_name,
            "NUM_CLASSES": config.num_classes,
            "IMAGE_SIZE": config.image_size,
            "BATCH_SIZE": config.batch_size,
            "LOSS": config.loss_name,
            "EPOCHS": self.num_epochs,
            "LR": self.optimizer.param_groups[0]['lr'],
            "LR_DROP_EPOCH": self.lr_drop_epoch,
            "LR_FACTOR": self.scheduler.gamma,
            "LR_STEP": self.scheduler.step_size,
            "DEVICE": str(self.device)
        }
        with open(self.logfile_path, "w") as f:
            json.dump(cfg_dict, f, indent=4)

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate model on validation set.
        """
        self.model.eval()
        total_loss, total_iou = 0.0, 0.0
        for images, masks in tqdm(self.val_loader, desc="Validation", leave=False):
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)
            if isinstance(outputs, dict):
                outputs = outputs["out"]

            loss = self.criterion(outputs, masks)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            inter = (preds * masks).sum(dim=(1, 2, 3))
            union = ((preds + masks) >= 1).sum(dim=(1, 2, 3))
            iou = (inter + 1e-6) / (union + 1e-6)

            total_loss += loss.item()
            total_iou += iou.mean().item()
        return total_loss / len(self.val_loader), total_iou / len(self.val_loader)

    def save_results(self):
        results = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_ious": self.val_ious,
            "best_iou": self.best_iou
        }
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=4)

    def train_one_epoch(self, epoch: int):
        """
        Train model for a single epoch.
        """
        self.model.train()
        total_train_loss = 0.0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

        for images, masks in loop:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs["out"]
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_train_loss += loss.item()
            loop.set_postfix(train_loss=total_train_loss / (loop.n + 1))

        return total_train_loss / len(self.train_loader)

    def train(self):
        """
        Full training loop.
        """
        for epoch in range(self.num_epochs):
            if epoch >= self.lr_drop_epoch:
                self.scheduler.step()
                print(f"🔽 Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            avg_train_loss = self.train_one_epoch(epoch)

            val_loss, val_iou = self.evaluate()

            self.train_losses.append(avg_train_loss)
            self.val_losses.append(val_loss)
            self.val_ious.append(val_iou)

            print(f"Epoch [{epoch+1}/{self.num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                torch.save(self.model.state_dict(), self.model_path)
                print(f"✅ Saved new best model: {self.model_path}")

        self.save_results()
        print(f"[INFO] Training complete. Best IoU: {self.best_iou:.4f}")
        print(f"[INFO] Results saved in {self.results_path}")

    def run(self):
        self.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for segmentation models")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--batch", type=int, default=BATCH, help="Batch size")
    args = parser.parse_args()

    config = TrainConfig(
        model_name=MODEL,
        num_classes=NUM_CLASSES,
        image_size=SIZE,
        batch_size=args.batch,
        loss_name=LOSS
    )

    trainer = Trainer(config, num_epochs=args.epochs, lr=args.lr)
    trainer.run()
