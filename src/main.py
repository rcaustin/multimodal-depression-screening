#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.DepressionDataset import DepressionDataset
from src.SimpleMultimodalModel import SimpleMultimodalModel


def main():
    # Initialize Model, Optimizer, and Loss Function
    print("Initializing model...")
    model = SimpleMultimodalModel()
    print("Initializing optimizer and loss function...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # or BCEWithLogitsLoss for classification

    # DataLoader
    print("Loading dataset...")
    dataset = DepressionDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    print(f"Dataset size: {len(dataset)} samples")

    # Verify DataLoader
    print("Verifying DataLoader...")
    for text, audio, visual, label in dataloader:
        print(f"Text batch shape: {text.shape}")
        print(f"Audio batch shape: {audio.shape}")
        print(f"Visual batch shape: {visual.shape}")
        print(f"Label batch shape: {label.shape}")
        break  # Just to verify one batch
    print("DataLoader verification complete.")

    # Training Loop
    print("Starting training...")
    epochs = 10
    for epoch in range(epochs):
        epoch_loss = 0.0
        for text, audio, visual, label in dataloader:
            optimizer.zero_grad()
            output = model(text, audio, visual)
            loss = criterion(output.squeeze(), label.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader):.4f}")


if __name__ == "__main__":
    main()
