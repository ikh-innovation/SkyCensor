import json
import matplotlib.pyplot as plt

def plot_training_results(json_path):
    with open(json_path, "r") as f:
        results = json.load(f)

    train_losses = results["train_losses"]
    val_losses = results["val_losses"]
    val_ious = results["val_ious"]
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    plt.plot(epochs, val_losses, label="Val Loss", marker='o')
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, val_ious, label="Val IoU", color='green', marker='o')
    plt.title("Validation IoU per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.grid(True)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    plot_training_results("runs/exp_2/train_results.json")
