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


class Trainer:
    """
    Trainer class handles dataset loading, model training, evaluation, checkpointing, and saving.
    """

    def __init__(
        self,
        model,
        batch_size=8,
        epochs=50,
        lr=1e-4,
        modalities=("text", "audio", "visual"),
        save_dir="models",
        use_dann=False,  # Whether to use Domain-Adversarial Neural Network (DANN)
        dann_lambda=0.1,  # Weight for domain adversary loss
        dann_alpha=1.0,  # Gradient reversal scaling factor
        chunk_len = None,
        chunk_hop = None,
        model_name = None
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

        # Apply Patient-Level Split
        train_sessions, _ = stratified_patient_split()

        # Determine Model Type and Initialize Dataset
        if isinstance(model, StaticModel):
            train_dataset = StaticDataset(train_sessions)
            default_name = "static_model.pt"
            collate_fn = None  # Default Collate for Static Dataset
        else: # Temporal Model
            train_dataset = TemporalDataset(train_sessions, chunk_len=self.chunk_len, chunk_hop=self.chunk_hop)
            if use_dann:
                default_name = "temporal_model_dann.pt"
            else:
                default_name = "temporal_model.pt"
            
            # Choose collate function based on chunking
            if self.chunk_len is None:
                collate_fn = temporal_collate_fn # Old behavior, with padding
            else:
                collate_fn = chunked_temporal_collate_fn # New behavior, no padding

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

        # Check for Existing Checkpoint
        self.start_epoch = 0
        self._load_checkpoint_if_available()

    def run(self):
        """Run the full training loop with checkpointing and timing."""
        logger.info(f"Training model: {type(self.model).__name__}")
        if self.use_dann and self.domain_adversary is not None:
            logger.info(
                f"DANN enabled with lambda={self.dann_lambda}, alpha={self.dann_alpha}"
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

                # Get gender labels for DANN if available
                gender = batch.get("gender")
                if gender is not None:
                    gender = gender.float().to(self.device).view(-1)

                # === DANN path ===
                if self.use_dann and self.domain_adversary is not None:
                    # Get the logits and features from the model
                    output, features = self.model(
                        text, audio, visual, return_features=True
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
                    output = self.model(text, audio, visual).view(-1)
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

            # Save Checkpoint Each Epoch
            self._save_checkpoint(epoch + 1)

        logger.info("Training Complete.")

    def _checkpoint_path(self):
        return os.path.join(self.save_dir, self.model_name)

    def _save_checkpoint(self, epoch):
        """Save model and optimizer state."""
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
                    self.domain_adversary.load_state_dict(checkpoint["domain_adversary_state_dict"])

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
        else:
            logger.info("No existing checkpoint found — starting fresh.")
            self.start_epoch = 0