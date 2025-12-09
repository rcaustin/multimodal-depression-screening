import os
from typing import Dict

import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.StaticDataset import StaticDataset
from src.datasets.TemporalDataset import TemporalDataset
from src.StaticModel import StaticModel
from src.TemporalModel import TemporalModel
from src.utility.collation import temporal_collate_fn, chunked_temporal_collate_fn
from src.utility.splitting import stratified_patient_split
from src.utility.visualization import plot_confusion_matrix


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

    def __init__(
        self,
        model: torch.nn.Module,
        test_fraction: float = 0.2,
        batch_size: int = 8,
        use_dann: bool = False,
        chunk_len: int | None = None,
        chunk_hop: int | None = None,
        ckpt_name: str | None = None,
    ):
        self.device: str = "cpu"
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.use_dann = use_dann
        self.test_fraction = test_fraction
        self.chunk_len = chunk_len
        self.chunk_hop = chunk_hop
        self.ckpt_name = ckpt_name

        # Create results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Determine Checkpoint Path
        if ckpt_name is not None:
            # Check if .pt extension is present
            if not ckpt_name.endswith(".pt"):
                ckpt_name += ".pt"
            self.checkpoint_path = (
                f"models/{ckpt_name}"  # Get the custom checkpoint path
            )
        else:
            # Fallback to default naming
            if isinstance(self.model, StaticModel):
                self.checkpoint_path = "models/static_model.pt"
            elif isinstance(self.model, TemporalModel):
                if self.use_dann:
                    self.checkpoint_path = "models/temporal_model_dann.pt"
                else:
                    self.checkpoint_path = "models/temporal_model.pt"
            else:
                raise ValueError(f"Unknown Model Type: {type(self.model)}")

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

        total_loss = 0.0

        # Detect whether we're evaluating on chunked temporal dataset
        chunked = (
            isinstance(self.model, TemporalModel)
            and hasattr(self.test_dataset, "chunk_len")
            and self.test_dataset.chunk_len is not None
        )

        if not chunked:
            all_outputs, all_targets = [], []
        else:
            session_logits: Dict[str, list[float]] = {}
            session_targets: Dict[str, int] = {}

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
            if not chunked:
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
            else:  # Chunked mode
                # Aggregate logits per session
                sessions = batch["session"]  # List of session IDs
                logits = outputs.detach().cpu().view(-1)  # [B]
                targs = targets.detach().cpu().view(-1)  # [B]

                for sid, logit, targ in zip(sessions, logits, targs):
                    sid = str(sid)
                    if sid not in session_logits:
                        session_logits[sid] = []
                    session_logits[sid].append(float(logit))

                    # Store the session-level label
                    targ_int = int(targ.item())
                    if sid in session_targets:
                        # Safety check, make sure labels match
                        if session_targets[sid] != targ_int:
                            logger.warning(
                                f"Conflicting labels for session {sid}, {session_targets[sid]}"
                                f" vs {targ_int}"
                            )
                    session_targets[sid] = targ_int

        # Concatenate All Batches or Aggregate Chunked Results
        if not chunked:
            all_outputs = torch.cat(all_outputs, dim=0).squeeze()
            all_targets = torch.cat(all_targets, dim=0).squeeze()

        else:
            # Build per-session outputs/targets
            session_ids = sorted(session_logits.keys())
            agg_outputs, agg_targets = [], []

            for sid in session_ids:
                logs = session_logits[sid]
                # Mean logit over all chunks for this session
                mean_logit = sum(logs) / len(logs)
                agg_outputs.append(mean_logit)
                agg_targets.append(session_targets[sid])

            all_outputs = torch.tensor(agg_outputs)
            all_targets = torch.tensor(agg_targets)

        # In both modes, compute loss per sample
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

        # Generate and save confusion matrix
        self._save_confusion_matrix(all_targets, preds)

        return results

    def _load_checkpoint(self):
        # Log Checkpoint Loading
        logger.info(f"Loading Checkpoint From {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load DANN if applicable
        if self.use_dann and "domain_adversary_state_dict" in checkpoint:
            if (
                hasattr(self.model, "domain_adversary")
                and self.model.domain_adversary is not None
            ):
                self.model.domain_adversary.load_state_dict(
                    checkpoint["domain_adversary_state_dict"]
                )

        # Infer chunking parameters if not explicitly set
        ckpt_chunk_len = checkpoint.get("chunk_len", None)
        ckpt_chunk_hop = checkpoint.get("chunk_hop", None)

        if self.chunk_len is None and ckpt_chunk_len is not None:
            self.chunk_len = ckpt_chunk_len

        if self.chunk_hop is None and ckpt_chunk_hop is not None:
            self.chunk_hop = ckpt_chunk_hop

        logger.info(f"Test chunk_len: {self.chunk_len}, chunk_hop: {self.chunk_hop}")

    def _prepare_dataset_and_loader(self):
        # Split Dataset By Patient
        _, test_sessions = stratified_patient_split()

        # Initialize Test Dataset
        if isinstance(self.model, StaticModel):
            self.test_dataset = StaticDataset(test_sessions)
        else:
            self.test_dataset = TemporalDataset(
                test_sessions, chunk_len=self.chunk_len, chunk_hop=self.chunk_hop
            )

        # Initialize Test DataLoader
        if isinstance(self.model, StaticModel):
            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
            )
        else:

            # Choose collate function based on chunking
            if getattr(self.test_dataset, "chunk_len", None) is None:
                collate_fn = temporal_collate_fn
            else:
                collate_fn = chunked_temporal_collate_fn

            self.test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
                collate_fn=collate_fn,
            )

    def _save_confusion_matrix(self, y_true, y_pred):
        """Generate and save confusion matrix visualization."""
        # Determine base name from checkpoint
        if self.ckpt_name:
            base_name = self.ckpt_name.replace(".pt", "")
        else:
            # Use default naming
            if isinstance(self.model, StaticModel):
                base_name = "static_model"
            elif isinstance(self.model, TemporalModel):
                if self.use_dann:
                    base_name = "temporal_model_dann"
                else:
                    base_name = "temporal_model"
            else:
                base_name = "model"

        # Convert tensors to numpy if needed
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()

        # Generate confusion matrix
        cm_path = os.path.join(self.results_dir, f"{base_name}_confusion_matrix.png")
        model_type = type(self.model).__name__

        plot_confusion_matrix(
            y_true,
            y_pred,
            save_path=cm_path,
            title=f"{model_type.replace("Model", " Model")} Confusion Matrix",
            class_names=["No Depression", "Depression"],
        )

        logger.info(f"Confusion matrix saved to {cm_path}")
