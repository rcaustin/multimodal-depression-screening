#!/usr/bin/env python3
import torch
import torch.nn as nn

from src.DataLoader import dataloader
from src.SimpleMultimodalModel import SimpleMultimodalModel


def main():
    model = SimpleMultimodalModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()  # or BCEWithLogitsLoss for classification

    # Training Loop
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
