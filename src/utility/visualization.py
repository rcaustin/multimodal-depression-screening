"""
Visualization utilities for training metrics and model evaluation.

Provides functions to generate and save:
- Training/validation loss curves
- Confusion matrices with annotations
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_loss_curve(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Loss Curve",
):
    """
    Plot training and optionally validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses per epoch
        save_path: Path to save the figure (if None, displays instead)
        title: Title for the plot
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss")

    if val_losses is not None:
        plt.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add minor gridlines for better readability
    plt.grid(True, which="minor", alpha=0.1)
    plt.minorticks_on()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    class_names: Optional[List[str]] = None,
):
    """
    Plot a confusion matrix with annotations.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the figure (if None, displays instead)
        title: Title for the plot
        class_names: Optional list of class names for labels
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=(8, 6))

    # Use default class names if not provided
    if class_names is None:
        class_names = ["Negative", "Positive"]

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        square=True,
    )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")

    # Add percentage annotations
    total = np.sum(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = cm[i, j] / total * 100
            plt.text(
                j + 0.5,
                i + 0.7,
                f"({percentage:.1f}%)",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_domain_loss_curve(
    train_losses: List[float],
    domain_losses: List[float],
    save_path: Optional[str] = None,
    title: str = "Training and Domain Loss Curves",
):
    """
    Plot training task loss and domain adversarial loss on separate axes.

    Args:
        train_losses: List of task training losses per epoch
        domain_losses: List of domain adversarial losses per epoch
        save_path: Path to save the figure (if None, displays instead)
        title: Title for the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    epochs = range(1, len(train_losses) + 1)

    # Task loss
    ax1.plot(epochs, train_losses, "b-", linewidth=2, label="Task Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Task Loss", fontsize=12)
    ax1.set_title("Task Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Domain loss
    ax2.plot(epochs, domain_losses, "r-", linewidth=2, label="Domain Loss")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Domain Loss", fontsize=12)
    ax2.set_title("Domain Adversarial Loss", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
