from typing import Dict

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class Tester:
    """
    Utility class for evaluating a trained multimodal model on a test dataset.

    Handles batching, device placement, evaluation metrics, and logging of results.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        test_loader (DataLoader): DataLoader providing test samples.
        device (str): Device to run evaluation on ('cpu' or 'cuda').
        metrics (dict[str, callable], optional): Dict of metric functions, each accepting
            (preds, targets) and returning a scalar.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        device: str = "cpu",
        metrics: Dict[str, callable] | None = None,
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.metrics = metrics or {}

        self.criterion = torch.nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_outputs = []
        all_targets = []
        total_loss = 0.0

        for batch in tqdm(self.test_loader, desc="[ModelTester] Evaluating"):
            # Move Modalities to Device
            text_seq = batch["text"].to(self.device)
            audio_seq = batch["audio"].to(self.device)
            visual_seq = batch["visual"].to(self.device)

            targets = batch["label"].to(self.device)
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)  # [B, 1]

            outputs = self.model(text_seq, audio_seq, visual_seq)

            # Compute Loss
            loss = self.criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)

            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

        # Concatenate All Batches
        all_outputs = torch.cat(all_outputs, dim=0).squeeze()  # [N]
        all_targets = torch.cat(all_targets, dim=0).squeeze()  # [N]

        avg_loss = total_loss / len(self.test_loader.dataset)

        # Convert Logits to Probabilities
        probs = torch.sigmoid(all_outputs)

        # Binary Predictions (Threshold 0.5)
        preds = (probs >= 0.5).long()

        # Compute Standard Metrics
        accuracy = accuracy_score(all_targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, preds, average="binary"
        )
        try:
            roc_auc = roc_auc_score(all_targets, probs)
        except ValueError:
            roc_auc = float("nan")  # Not Enough Positive/Negative Samples

        results = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "outputs": all_outputs,
            "targets": all_targets,
        }

        return results
