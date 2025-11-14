from typing import Dict

import torch
from loguru import logger
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.StaticDataset import StaticDataset
from src.datasets.TemporalDataset import TemporalDataset
from src.StaticModel import StaticModel
from src.TemporalModel import TemporalModel
from src.utility.collation import temporal_collate_fn
from src.utility.splitting import stratified_patient_split


class Tester:
    """
    Evaluates a trained multimodal model on a test dataset.

    Handles:
        - Checkpoint loading
        - Dataset and DataLoader preparation
        - Device placement
        - Loss computation
        - Standard binary classification metrics

    Args:
        model (torch.nn.Module): Model to evaluate
        device (str): Device to run evaluation on ('cpu' or 'cuda')
    """

    def __init__(self, model: torch.nn.Module, test_fraction: float = 0.2):
        self.device: str = "cpu"
        self.model = model.to(self.device)
        self.test_fraction = test_fraction

        # Determine Checkpoint Path
        if isinstance(model, StaticModel):
            self.checkpoint_path = "models/static_model.pt"
        elif isinstance(model, TemporalModel):
            self.checkpoint_path = "models/temporal_model.pt"
        else:
            raise ValueError(f"Unknown Model Type: {type(model)}")

        # Load Checkpoint
        self._load_checkpoint()

        # Prepare Dataset And Dataloader
        self._prepare_dataset_and_loader()

        # Loss Function
        self.criterion = torch.nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def evaluate(self) -> Dict:
        # Set Model To Evaluation Mode
        self.model.eval()
        all_outputs, all_targets = [], []
        total_loss = 0.0

        # Iterate Over Test Batches
        for batch in tqdm(self.test_loader, desc="[ModelTester] Evaluating"):
            # Move Data To Device
            if isinstance(self.model, TemporalModel):
                text_seq = batch["text"].to(self.device)
                audio_seq = batch["audio"].to(self.device)
                visual_seq = batch["visual"].to(self.device)
                outputs = self.model(text_seq, audio_seq, visual_seq)
            else:
                text = batch.get("text")
                audio = batch.get("audio")
                visual = batch.get("visual")

                if text is not None:
                    text = text.to(self.device)
                if audio is not None:
                    audio = audio.to(self.device)
                if visual is not None:
                    visual = visual.to(self.device)

                outputs = self.model(text, audio, visual)

            # Prepare Targets
            targets = batch["label"].to(self.device)
            if targets.ndim == 1:
                targets = targets.unsqueeze(1)

            # Compute Loss
            loss = self.criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)

            # Collect Outputs And Targets
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

        # Concatenate All Batches
        all_outputs = torch.cat(all_outputs, dim=0).squeeze()
        all_targets = torch.cat(all_targets, dim=0).squeeze()

        # Compute Average Loss
        avg_loss = total_loss / len(self.test_dataset)

        # Convert Logits To Probabilities
        probs = torch.sigmoid(all_outputs)

        # Generate Binary Predictions With Threshold 0.5
        preds = (probs >= 0.5).long()

        # Compute Standard Metrics
        accuracy = accuracy_score(all_targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, preds, average="binary", zero_division=0
        )
        try:
            roc_auc = roc_auc_score(all_targets, probs)
        except ValueError:
            roc_auc = float("nan")  # Not Enough Positive/Negative Samples

        # Aggregate Results
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

    def _load_checkpoint(self):
        # Log Checkpoint Loading
        logger.info(f"Loading Checkpoint From {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def _prepare_dataset_and_loader(self):
        # Split Dataset By Patient
        _, test_sessions = stratified_patient_split()

        # Initialize Test Dataset
        if isinstance(self.model, StaticModel):
            self.test_dataset = StaticDataset(test_sessions)
        else:
            self.test_dataset = TemporalDataset(test_sessions)

        # Initialize Test DataLoader
        if isinstance(self.model, StaticModel):
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=4,
                num_workers=0,
                shuffle=False,
            )
        else:
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=4,
                num_workers=0,
                shuffle=False,
                collate_fn=temporal_collate_fn,
            )
