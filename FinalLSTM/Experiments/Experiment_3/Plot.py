import os
import torch
import matplotlib.pyplot as plt


# Always resolve paths relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_1 = os.path.join(
    BASE_DIR,
    "Model_1",
    "checkpoints",
    "model.pt"
)

MODEL_2 = os.path.join(
    BASE_DIR,
    "Model_2",
    "checkpoints",
    "model.pt"
)

MODEL_2 = os.path.join(
    BASE_DIR,
    "Model_3",
    "checkpoints",
    "model.pt"
)

SAVE_PATH = os.path.join(
    BASE_DIR,
    "detach_vs_no_detach.png"
)


def load_checkpoint(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found:\n{path}")

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    if "train_losses" not in checkpoint:
        raise ValueError(f"train_losses missing in:\n{path}")

    if "val_losses" not in checkpoint:
        raise ValueError(f"val_losses missing in:\n{path}")

    return checkpoint


def main():

    checkpoint_1 = load_checkpoint(MODEL_1)
    checkpoint_2 = load_checkpoint(MODEL_2)

    train_1 = checkpoint_1["train_losses"]
    val_1 = checkpoint_1["val_losses"]

    train_2 = checkpoint_2["train_losses"]
    val_2 = checkpoint_2["val_losses"]

    epochs_1 = range(1, len(train_1) + 1)
    epochs_2 = range(1, len(train_2) + 1)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5)
    )

    # Training loss
    axes[0].plot(
        epochs_1,
        train_1,
        marker="o",
        label="With detach"
    )

    axes[0].plot(
        epochs_2,
        train_2,
        marker="o",
        label="Without detach"
    )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    # Validation loss
    axes[1].plot(
        epochs_1,
        val_1,
        marker="o",
        label="With detach"
    )

    axes[1].plot(
        epochs_2,
        val_2,
        marker="o",
        label="Without detach"
    )

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300)
    plt.close()

    print(f"[SUCCESS] Plot saved to:")
    print(SAVE_PATH)


if __name__ == "__main__":
    main()