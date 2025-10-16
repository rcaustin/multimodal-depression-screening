#!/usr/bin/env python3
import argparse

from loguru import logger

from src.StaticModel import StaticModel
from src.TemporalModel import TemporalModel
from src.Trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train multimodal depression models"
    )
    parser.add_argument(
        "mode",
        choices=["static", "temporal"],
        help=("Choose which model to train: 'static' for "
              "StaticModel or 'temporal' for "
              "TemporalModel")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Selected mode: {args.mode}")

    if args.mode == "static":
        model = StaticModel()
    elif args.mode == "temporal":
        model = TemporalModel(
            text_dim=768,
            audio_dim=88,
            visual_dim=17,
            hidden_dim=128,
            encoder_type="lstm",
            pooling="mean",
            dropout=0.3,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    trainer = Trainer(model)
    trainer.run()


if __name__ == "__main__":
    main()
