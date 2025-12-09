import os
import time

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.datasets.StaticDataset import StaticDataset
from src.datasets.TemporalDataset import TemporalDataset
from src.StaticModel import StaticModel
from src.utility.collation import temporal_collate_fn, chunked_temporal_collate_fn
from src.utility.splitting import stratified_patient_split

from src.utility.grl import grad_reverse
from src.components.DomainAdversary import DANN
from src.utility.visualization import (
    plot_loss_curve,
    plot_domain_loss_curve,
    plot_validation_metrics,
)

class Trainer:
    """
    Trainer class handles dataset loading, model training, evaluation, checkpointing, and saving.
    """

    def __init__(
        self,
        model,
        train_sessions=None,  # Optional: specify training sessions
        batch_size=1,
        epochs=50,
        lr=1e-4,
        modalities=("text", "audio", "visual"),
        save_dir="models",
        use_dann=False,  # Whether to use Domain-Adversarial Neural Network (DANN)
        dann_lambda=0.1,  # Weight for domain adversary loss
        dann_alpha=1.0,  # Gradient reversal scaling factor
        chunk_len=None,
        chunk_hop=None,
        model_name=None,
        val_sessions=None,  # Optional: validation session IDs
        early_stopping_patience=None,  # Optional: epochs without improvement
        early_stopping_metric="f1",  # Metric to monitor
        early_stopping_mode="max",  # "max" for metrics, "min" for loss
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.modalities = modalities
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.chunk_len = chunk_len
        self.chunk_hop = chunk_hop

        # ---- CUDA SPEEDUP OPTION ----
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Create Save Directory
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # DANN flags
        self.use_dann = use_dann
        self.dann_lambda = dann_lambda
        self.dann_alpha = dann_alpha

        # Validation and early stopping parameters
        self.val_sessions = val_sessions
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode

        # Apply Patient-Level Split (if train_sessions not provided)
        if train_sessions is None:
            train_sessions, _ = stratified_patient_split()

        self.train_sessions = train_sessions

        # Determine Model Type and Initialize Dataset
        if isinstance(model, StaticModel):
            train_dataset = StaticDataset(train_sessions)
            default_name = "static_model.pt"
            collate_fn = None  # Default Collate for Static Dataset
        else:  # Temporal Model
            train_dataset = TemporalDataset(
                train_sessions, chunk_len=self.chunk_len, chunk_hop=self.chunk_hop
            )
            if use_dann:
                default_name = "temporal_model_dann.pt"
            else:
                default_name = "temporal_model.pt"

            # Choose collate function based on chunking
            if self.chunk_len is None:
                collate_fn = temporal_collate_fn  # Old behavior, with padding
            else:
                collate_fn = chunked_temporal_collate_fn  # New behavior, no padding

        # Choose self.model_name
        if model_name is not None:
            # Add .pt if missing
            if not model_name.endswith(".pt"):
                model_name += ".pt"
            self.model_name = model_name
        else:
            self.model_name = default_name

        # Initialize DataLoader
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # Task loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Setup DANN if enabled
        self.domain_adversary = None
        self.domain_criterion = None

        if self.use_dann and not isinstance(
            model, StaticModel
        ):  # DANN only for Temporal Models
            feature_dim = getattr(
                self.model, "hidden_dim", 128
            )  # Uses the hidden_dim from model as input size

            # Domain adversary head
            self.domain_adversary = DANN(input_dim=feature_dim).to(self.device)
            self.domain_criterion = nn.BCEWithLogitsLoss()

            # Joint optimizer for model and domain adversary
            self.optimizer = torch.optim.Adam(
                list(self.model.parameters())
                + list(self.domain_adversary.parameters()),
                lr=self.lr,
            )
        else:
            # Original optimizer for model only
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Initialize loss tracking
        self.train_losses = []
        self.domain_losses = []

        # Initialize validation metrics tracking
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_roc_aucs = []

        # Initialize early stopping state
        if self.early_stopping_mode == "max":
            self.best_metric_value = float("-inf")
        else:
            self.best_metric_value = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stopped = False

        # Create results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize validation dataloader
        self.val_dataloader = None
        self._prepare_validation_dataloader()

        # Check for Existing Checkpoint
        self.start_epoch = 0
        self._load_checkpoint_if_available()

    def run(self):
        """Run the full training loop with validation and early stopping."""
        logger.info(f"Training model: {type(self.model).__name__}")
        if self.use_dann and self.domain_adversary is not None:
            logger.info(
                f"DANN enabled with lambda={self.dann_lambda}, alpha={self.dann_alpha}"
            )
        if self.val_sessions is not None:
            logger.info(f"Validation enabled with {len(self.val_sessions)} sessions")
        if self.early_stopping_patience is not None:
            logger.info(
                f"Early stopping enabled: patience={self.early_stopping_patience}, "
                f"metric={self.early_stopping_metric}, mode={self.early_stopping_mode}"
            )

        self.model.train()
        if self.domain_adversary is not None:
            self.domain_adversary.train()

        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.perf_counter()  # Start Timer
            epoch_loss = 0.0
            epoch_domain_loss = 0.0

            for batch in self.dataloader: 
                self.optimizer.zero_grad()

                # Move Features To Device
                text = batch.get("text")
                audio = batch.get("audio")
                visual = batch.get("visual")
                label = batch["label"].float().to(self.device)
                label = label.view(-1)

                if text is not None:
                    text = text.to(self.device)
                if audio is not None:
                    audio = audio.to(self.device)
                if visual is not None:
                    visual = visual.to(self.device)

                # Move lengths to device
                lengths = batch["lengths"].to(self.device)


                # Get gender labels for DANN if available
                gender = batch.get("gender")
                if gender is not None:
                    gender = gender.float().to(self.device).view(-1)

                # === DANN path ===
                if self.use_dann and self.domain_adversary is not None:
                    # Get the logits and features from the model
                    output, features = self.model(
                        text, audio, visual, return_features=True, lengths=lengths
                    )
                    output = output.view(-1)

                    # Main task loss
                    task_loss = self.criterion(output, label)

                    # Apply Gradient Reversal Layer before domain adversary
                    reversed_features = grad_reverse(features, alpha=self.dann_alpha)
                    dlogits = self.domain_adversary(reversed_features).view(-1)

                    # Domain adversary loss
                    domain_loss = self.domain_criterion(dlogits, gender)

                    # Total loss
                    loss = task_loss + self.dann_lambda * domain_loss

                    epoch_loss += loss.item()
                    epoch_domain_loss += domain_loss.item()

                # === Standard path ===
                else:
                    output = self.model(text, audio, visual, lengths=lengths).view(-1)
                    loss = self.criterion(output, label)
                    epoch_loss += loss.item()

                # Backward Pass
                loss.backward()
                self.optimizer.step()

            # Compute Average Loss and Elapsed Time
            avg_loss = epoch_loss / len(self.dataloader)
            avg_domain_loss = (
                epoch_domain_loss / len(self.dataloader) if self.use_dann else 0.0
            )
            elapsed = time.perf_counter() - start_time  # Seconds

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}, "
                f"Loss: {avg_loss:.4f}, "
                f"Domain Loss: {avg_domain_loss:.4f}, "
                f"Time: {elapsed:.2f}s"
            )

            # Track losses
            self.train_losses.append(avg_loss)
            if self.use_dann:
                self.domain_losses.append(avg_domain_loss)

            # Validation after each epoch
            if self.val_dataloader is not None:
                val_results = self._validate()

                # Track validation metrics
                self.val_losses.append(val_results["loss"])
                self.val_accuracies.append(val_results["accuracy"])
                self.val_f1_scores.append(val_results["f1"])
                self.val_precisions.append(val_results["precision"])
                self.val_recalls.append(val_results["recall"])
                self.val_roc_aucs.append(val_results["roc_auc"])

                # Log validation metrics
                logger.info(
                    f"Validation - Loss: {val_results['loss']:.4f}, "
                    f"Acc: {val_results['accuracy']:.4f}, "
                    f"F1: {val_results['f1']:.4f}, "
                    f"AUC: {val_results['roc_auc']:.4f}"
                )

                # Early stopping check
                if self.early_stopping_patience is not None:
                    should_stop = self._check_early_stopping(val_results, epoch)
                    if should_stop:
                        logger.info(f"Early stopping triggered at epoch {epoch+1}")
                        self.early_stopped = True
                        # Save final checkpoint before stopping
                        self._save_checkpoint(epoch + 1)
                        self._save_loss_curves()
                        break

            # Save Checkpoint Each Epoch
            self._save_checkpoint(epoch + 1)

            # Generate and save loss curves
            self._save_loss_curves()

        # Load best model if early stopping was used
        if self.early_stopping_patience is not None and self.best_epoch > 0:
            logger.info(f"Loading best model from epoch {self.best_epoch}")
            self._load_best_checkpoint()

        logger.info("Training Complete.")

    def _prepare_validation_dataloader(self):
        """Create validation dataloader if val_sessions provided."""
        if self.val_sessions is None:
            return

        # Initialize validation dataset (same logic as training)
        if isinstance(self.model, StaticModel):
            val_dataset = StaticDataset(self.val_sessions)
            collate_fn = None
        else:  # Temporal Model
            val_dataset = TemporalDataset(
                self.val_sessions, chunk_len=self.chunk_len, chunk_hop=self.chunk_hop
            )
            if self.chunk_len is None:
                collate_fn = temporal_collate_fn
            else:
                collate_fn = chunked_temporal_collate_fn

        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # No shuffling for validation
            num_workers=0,
            collate_fn=collate_fn,
        )

    @torch.no_grad()
    def _validate(self):
        """
        Run validation loop and compute metrics.

        Returns:
            Dict with keys: loss, accuracy, precision, recall, f1, roc_auc, outputs, targets
        """
        if self.val_dataloader is None:
            return None

        from sklearn.metrics import (
            accuracy_score,
            precision_recall_fscore_support,
            roc_auc_score,
        )

        self.model.eval()
        if self.domain_adversary is not None:
            self.domain_adversary.eval()

        total_loss = 0.0

        # Determine if we're in chunked mode
        chunked = not isinstance(self.model, StaticModel) and self.chunk_len is not None

        if not chunked:
            all_outputs, all_targets = [], []
        else:
            session_logits = {}
            session_targets = {}

        # Validation loop
        for batch in self.val_dataloader:
            # Move features to device
            text = batch.get("text")
            audio = batch.get("audio")
            visual = batch.get("visual")
            label = batch["label"].float().to(self.device)
            label = label.view(-1)

            if text is not None:
                text = text.to(self.device)
            if audio is not None:
                audio = audio.to(self.device)
            if visual is not None:
                visual = visual.to(self.device)

            # Move lengths to device
            lengths = batch.get("lengths")
            if lengths is not None:
                lengths = lengths.to(self.device)

            # Forward pass
            if isinstance(self.model, StaticModel):
                output = self.model(text, audio, visual).view(-1)
            else:
                output = self.model(text, audio, visual, lengths=lengths).view(-1)

            # Compute loss
            loss = self.criterion(output, label)
            total_loss += loss.item() * label.size(0)

            # Store outputs and targets
            if not chunked:
                all_outputs.append(output.cpu())
                all_targets.append(label.cpu())
            else:
                # Chunked mode: aggregate by session
                sessions = batch["session"]
                for sid, logit, targ in zip(sessions, output.cpu(), label.cpu()):
                    if sid not in session_logits:
                        session_logits[sid] = []
                        session_targets[sid] = targ.item()
                    session_logits[sid].append(logit.item())

        # Aggregate results
        if not chunked:
            all_outputs = torch.cat(all_outputs)
            all_targets = torch.cat(all_targets)
        else:
            # Average logits per session
            all_outputs = []
            all_targets = []
            for sid in session_logits.keys():
                mean_logit = sum(session_logits[sid]) / len(session_logits[sid])
                all_outputs.append(mean_logit)
                all_targets.append(session_targets[sid])
            all_outputs = torch.tensor(all_outputs)
            all_targets = torch.tensor(all_targets)

        # Convert to probabilities and predictions
        probs = torch.sigmoid(all_outputs)
        preds = (probs >= 0.5).long()

        # Compute metrics
        accuracy = accuracy_score(all_targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, preds, average="binary", zero_division=0
        )
        try:
            roc_auc = roc_auc_score(all_targets, probs)
        except ValueError:
            roc_auc = float("nan")

        # Compute average loss
        if chunked:
            avg_loss = total_loss / len(session_logits)
        else:
            avg_loss = total_loss / len(self.val_dataloader.dataset)

        # Restore training mode
        self.model.train()
        if self.domain_adversary is not None:
            self.domain_adversary.train()

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "outputs": all_outputs,
            "targets": all_targets,
        }

    def _check_early_stopping(self, val_results, epoch):
        """
        Check if early stopping criteria is met.

        Returns:
            True if training should stop, False otherwise
        """
        # Get metric value
        metric_value = val_results[self.early_stopping_metric]

        # Check if this is the best metric so far
        is_better = (
            self.early_stopping_mode == "max"
            and metric_value > self.best_metric_value
        ) or (
            self.early_stopping_mode == "min"
            and metric_value < self.best_metric_value
        )

        if is_better:
            self.best_metric_value = metric_value
            self.best_epoch = epoch + 1
            self.patience_counter = 0
            # Save best model checkpoint
            self._save_best_checkpoint()
            logger.info(f"New best {self.early_stopping_metric}: {metric_value:.4f}")
            return False
        else:
            self.patience_counter += 1
            logger.info(
                f"No improvement in {self.early_stopping_metric} "
                f"({self.patience_counter}/{self.early_stopping_patience})"
            )
            return self.patience_counter >= self.early_stopping_patience

    def _best_checkpoint_path(self):
        """Path for best model checkpoint."""
        base_name = self.model_name.replace(".pt", "")
        return os.path.join(self.save_dir, f"{base_name}_best.pt")

    def _save_best_checkpoint(self):
        """Save best model checkpoint (separate from regular checkpoint)."""
        save_path = self._best_checkpoint_path()
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs_trained": self.best_epoch,
            "modalities": self.modalities,
            "use_dann": self.use_dann,
            "dann_lambda": self.dann_lambda,
            "dann_alpha": self.dann_alpha,
            "chunk_len": self.chunk_len,
            "chunk_hop": self.chunk_hop,
            "train_losses": self.train_losses,
            "domain_losses": self.domain_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "val_f1_scores": self.val_f1_scores,
            "val_precisions": self.val_precisions,
            "val_recalls": self.val_recalls,
            "val_roc_aucs": self.val_roc_aucs,
            "best_metric": self.early_stopping_metric,
            "best_metric_value": self.best_metric_value,
        }
        if self.domain_adversary is not None:
            checkpoint["domain_adversary_state_dict"] = (
                self.domain_adversary.state_dict()
            )
        torch.save(checkpoint, save_path)

    def _load_best_checkpoint(self):
        """Load best model checkpoint."""
        load_path = self._best_checkpoint_path()
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.use_dann and "domain_adversary_state_dict" in checkpoint:
                if self.domain_adversary is not None:
                    self.domain_adversary.load_state_dict(
                        checkpoint["domain_adversary_state_dict"]
                    )
            logger.info(f"Loaded best model from {load_path}")

    def _save_validation_metrics_plot(self):
        """Plot validation metrics (accuracy, F1, AUC) over epochs."""
        base_name = self.model_name.replace(".pt", "")
        metrics_path = os.path.join(self.results_dir, f"{base_name}_val_metrics.png")

        plot_validation_metrics(
            val_accuracies=self.val_accuracies,
            val_f1_scores=self.val_f1_scores,
            val_roc_aucs=self.val_roc_aucs,
            save_path=metrics_path,
            title=f"{type(self.model).__name__} Validation Metrics",
        )

    def _checkpoint_path(self):
        return os.path.join(self.save_dir, self.model_name)

    def _save_checkpoint(self, epoch):
        """Save model and optimizer state with validation metrics."""
        save_path = self._checkpoint_path()

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs_trained": epoch,
            "modalities": self.modalities,
            "use_dann": self.use_dann,
            "dann_lambda": self.dann_lambda,
            "dann_alpha": self.dann_alpha,
            "chunk_len": self.chunk_len,
            "chunk_hop": self.chunk_hop,
            "train_losses": self.train_losses,
            "domain_losses": self.domain_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "val_f1_scores": self.val_f1_scores,
            "val_precisions": self.val_precisions,
            "val_recalls": self.val_recalls,
            "val_roc_aucs": self.val_roc_aucs,
        }

        if self.domain_adversary is not None:
            checkpoint["domain_adversary_state_dict"] = (
                self.domain_adversary.state_dict()
            )

        torch.save(checkpoint, save_path)

    def _load_checkpoint_if_available(self):
        """Load model and optimizer state if a checkpoint exists."""
        load_path = self._checkpoint_path()
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Restore DANN if applicable
            if self.use_dann and "domain_adversary_state_dict" in checkpoint:
                if self.domain_adversary is not None:
                    self.domain_adversary.load_state_dict(
                        checkpoint["domain_adversary_state_dict"]
                    )

            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Check for mismatched chunking settings
            ckpt_chunk_len = checkpoint.get("chunk_len", None)
            ckpt_chunk_hop = checkpoint.get("chunk_hop", None)

            if ckpt_chunk_len != self.chunk_len or ckpt_chunk_hop != self.chunk_hop:
                raise RuntimeError(
                    f"\nChunk configuration mismatch when resuming training:\n"
                    f"Checkpoint chunk_len: {ckpt_chunk_len}, Current chunk_len: {self.chunk_len}\n"
                    f"Checkpoint chunk_hop: {ckpt_chunk_hop}, Current chunk_hop: {self.chunk_hop}\n"
                    "Training will not proceed. Re-run with matching flags.\n"
                )

            self.start_epoch = checkpoint.get("epochs_trained", 0)

            # Restore loss history if available
            self.train_losses = checkpoint.get("train_losses", [])
            self.domain_losses = checkpoint.get("domain_losses", [])

            # Restore validation metrics history if available
            self.val_losses = checkpoint.get("val_losses", [])
            self.val_accuracies = checkpoint.get("val_accuracies", [])
            self.val_f1_scores = checkpoint.get("val_f1_scores", [])
            self.val_precisions = checkpoint.get("val_precisions", [])
            self.val_recalls = checkpoint.get("val_recalls", [])
            self.val_roc_aucs = checkpoint.get("val_roc_aucs", [])
        else:
            logger.info("No existing checkpoint found — starting fresh.")
            self.start_epoch = 0

    def _save_loss_curves(self):
        """Generate and save loss curve visualizations with validation."""
        if len(self.train_losses) == 0:
            return

        # Determine model type for filename
        model_type = type(self.model).__name__.lower()
        base_name = self.model_name.replace(".pt", "")

        # Save training/validation loss curve
        loss_path = os.path.join(self.results_dir, f"{base_name}_loss_curve.png")

        # Use val_losses if available
        val_losses = self.val_losses if len(self.val_losses) > 0 else None

        if self.use_dann and len(self.domain_losses) > 0:
            # Save combined task and domain loss curves
            plot_domain_loss_curve(
                self.train_losses,
                self.domain_losses,
                save_path=loss_path,
                title=f"{model_type.title().replace('model', ' Model')} Training Loss Curves",
            )
        else:
            # Save task loss curve with validation
            plot_loss_curve(
                self.train_losses,
                val_losses=val_losses,
                save_path=loss_path,
                title=f"{model_type.title().replace('model', ' Model')} Loss Curves",
            )

        # If validation metrics exist, plot them
        if len(self.val_f1_scores) > 0:
            self._save_validation_metrics_plot()
