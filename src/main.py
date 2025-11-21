#!/usr/bin/env python3
import argparse
from pprint import pprint

from loguru import logger

from src.StaticModel import StaticModel
from src.TemporalModel import TemporalModel
from src.Tester import Tester
from src.Trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or test multimodal depression models"
    )

    parser.add_argument(
        "operation",
        choices=["train", "test"],
        help="Operation mode: 'train' or 'test'",
    )

    parser.add_argument(
        "model",
        choices=["static", "temporal", "DANN"],
        help="Model type to use: 'static', 'temporal', or 'DANN'",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved checkpoint (required for test mode)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Operation: {args.operation}")
    logger.info(f"Model: {args.model}")

    BATCH_SIZE = 1
    EPOCHS = 50
    LR = 1e-4
    USE_DANN = args.model == "DANN"

    # Initialize Model
    model = StaticModel() if args.model == "static" else TemporalModel()

    # Training Branch
    if args.operation == "train":
        trainer = Trainer(model, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR, use_dann=USE_DANN)
        trainer.run()

    # Testing Branch
    elif args.operation == "test":
        try:
            tester = Tester(model, batch_size=BATCH_SIZE, use_dann=USE_DANN)
            results = tester.evaluate()
            logger.info("Test Results:")
            pprint(
                {k: v for k, v in results.items() if k not in ["outputs", "targets"]}
            )
        except FileNotFoundError:
            logger.warning(
                f"Checkpoint not found for {args.model} model. "
                f"Try running '{args.model} train' first."
            )

    else:
        raise ValueError(f"Unknown operation: {args.operation}")


if __name__ == "__main__":
    main()
