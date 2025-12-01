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

    parser.add_argument(
        "--chunk",
        action='store_true',
        help="Enable temporal chunking (4s windows, 2s hop) for temporal models."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Operation: {args.operation}")
    logger.info(f"Model: {args.model}")

    BATCH_SIZE = 8
    EPOCHS = args.epochs
    LR = 1e-4
    USE_DANN = args.model == "DANN"

    # Chunk config for temporal models
    if args.chunk and args.model != "static":
        CHUNK_LEN = 120 # 4 seconds at 30Hz 
        CHUNK_HOP = 60  # 2 seconds at 30Hz
        logger.info("Temporal chunking enabled: 4s windows with 2s hop.")
    else:
        CHUNK_LEN = None
        CHUNK_HOP = None
        if args.model == "static":
            logger.warning("Chunking option ignored for static model.")

    # Initialize Model
    model = StaticModel() if args.model == "static" else TemporalModel()

    # Training Branch
    if args.operation == "train":
        trainer = Trainer(model,
                          batch_size=BATCH_SIZE, 
                          epochs=EPOCHS, lr=LR, 
                          use_dann=USE_DANN, 
                          chunk_len=CHUNK_LEN, 
                          chunk_hop=CHUNK_HOP
        )
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
