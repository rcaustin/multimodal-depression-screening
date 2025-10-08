#!/usr/bin/env python3
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from src.DepressionDataset import DepressionDataset
from src.SimpleMultimodalModel import SimpleMultimodalModel


def main():
    # ---- Config ----
    modalities = ("text", "audio", "visual")
    batch_size = 4
    num_workers = 2
    lr = 1e-3
    epochs = 10
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # ---- Initialize model, optimizer, loss ----
    logger.info("Initializing model...")
    model = SimpleMultimodalModel()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()  # Binary classification

    # ---- Load dataset ----
    logger.info("Loading dataset...")
    dataset = DepressionDataset(modalities=modalities, cache=True)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    logger.info(f"Dataset size: {len(dataset)} samples")

    # ---- Verify DataLoader ----
    logger.info("Verifying DataLoader...")
    for batch in dataloader:
        text = batch.get("text")
        audio = batch.get("audio")
        visual = batch.get("visual")
        label = batch["label"]

        logger.info(
            f"Text: {text.shape if text is not None else 'N/A'}, "
            f"Audio: {audio.shape if audio is not None else 'N/A'}, "
            f"Visual: {visual.shape if visual is not None else 'N/A'}, "
            f"Label: {label.shape}"
        )
        break

    # ---- Training Loop ----
    logger.info("Starting training...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()

            text = batch.get("text")
            audio = batch.get("audio")
            visual = batch.get("visual")
            label = batch["label"].float().to(device)
            label = label.view(-1)  # ensure shape [batch_size]

            # Move features to device
            if text is not None:
                text = text.to(device)
            if audio is not None:
                audio = audio.to(device)
            if visual is not None:
                visual = visual.to(device)

            # Forward pass
            output = model(text, audio, visual).view(-1)  # ensure shape
            loss = criterion(output, label.float())

            # Backward pass
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
