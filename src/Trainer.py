import os
import time

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.datasets.StaticDataset import StaticDataset
from src.datasets.TemporalDataset import TemporalDataset
from src.StaticModel import StaticModel
from src.utility.collation import temporal_collate_fn
from src.utility.splitting import stratified_patient_split


class Trainer:
    """
    Trainer class handles dataset loading, model training, evaluation, checkpointing, and saving.
    """

    def __init__(
        self,
        model,
        batch_size=4,
        epochs=10,
        lr=1e-3,
        modalities=("text", "audio", "visual"),
        save_dir="models",
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.modalities = modalities
        self.device = torch.device("cpu")
        self.model = model.to(self.device)

        # Create Save Directory
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        # Apply Patient-Level Split
        train_sessions, _ = stratified_patient_split()

        # Determine Model Type and Initialize Dataset
        if isinstance(model, StaticModel):
            train_dataset = StaticDataset(train_sessions)
            self.model_name = "static_model.pt"
            collate_fn = None  # Default Collate for Static Dataset
        else:
            train_dataset = TemporalDataset(train_sessions)
            self.model_name = "temporal_model.pt"
            collate_fn = temporal_collate_fn  # Use Custom Temporal Collate

        # Initialize DataLoader
        self.dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # Loss and Optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Check for Existing Checkpoint
        self.start_epoch = 0
        self._load_checkpoint_if_available()

    def run(self):
        """Run the full training loop with checkpointing and timing."""
        logger.info(f"Training model: {type(self.model).__name__}")
        self.model.train()

        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.perf_counter()  # Start Timer
            epoch_loss = 0.0

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

                # Forward Pass
                output = self.model(text, audio, visual).view(-1)
                loss = self.criterion(output, label)

                # Backward Pass
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # Compute Average Loss and Elapsed Time
            avg_loss = epoch_loss / len(self.dataloader)
            elapsed = time.perf_counter() - start_time  # Seconds

            logger.info(
                f"Epoch {epoch+1}/{self.epochs}, "
                f"Loss: {avg_loss:.4f}, "
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
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epochs_trained": epoch,
                "modalities": self.modalities,
            },
            save_path,
        )

    def _load_checkpoint_if_available(self):
        """Load model and optimizer state if a checkpoint exists."""
        load_path = self._checkpoint_path()
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.start_epoch = checkpoint.get("epochs_trained", 0)
            logger.info(
                f"Resumed training from checkpoint at epoch {self.start_epoch} ({load_path})"
            )
        else:
            logger.info("No existing checkpoint found — starting fresh.")
