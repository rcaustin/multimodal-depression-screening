import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.DepressionDataset import DepressionDataset


class Trainer:
    """
    Trainer class handles dataset loading, model training, and evaluation.
    """

    def __init__(
            self,
            model,
            batch_size=4,
            epochs=10,
            lr=1e-3,
            modalities=("text", "audio", "visual")
    ):
        # Initialize trainer with model, dataset, and training parameters
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.modalities = modalities
        self.device = torch.device("cpu")
        self.model.to(self.device)

        # Dataset & dataloader
        self.dataset = DepressionDataset(modalities=self.modalities, cache=True)
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        # Loss and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def run(self):
        """Run the full training loop."""
        logger.info(f"Training model: {type(self.model).__name__}")
        self.model.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch in self.dataloader:
                self.optimizer.zero_grad()

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

                # Forward pass
                output = self.model(text, audio, visual).view(-1)
                loss = self.criterion(output, label)

                # Backward pass
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        logger.info("Training complete.")
