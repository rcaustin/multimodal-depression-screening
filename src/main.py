#!/usr/bin/env python3
import argparse
from pprint import pprint

import torch
from loguru import logger
from torch.utils.data import DataLoader

from src.datasets.StaticDataset import StaticDataset
from src.datasets.TemporalDataset import TemporalDataset
from src.StaticModel import StaticModel
from src.TemporalModel import TemporalModel
from src.Tester import Tester
from src.Trainer import Trainer
from src.utility.collation import temporal_collate_fn


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
        choices=["static", "temporal"],
        help="Model type to use: 'static' or 'temporal'",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a saved checkpoint (required for test mode)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for testing or training",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader worker threads",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Operation: {args.operation}")
    logger.info(f"Model: {args.model}")

    # Initialize model
    if args.model == "static":
        model = StaticModel()
    elif args.model == "temporal":
        model = TemporalModel()
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    if args.operation == "train":
        trainer = Trainer(model)
        trainer.run()

    elif args.operation == "test":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for test mode.")

        # Load Checkpoint
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        # Prepare Dataset
        data_dir = "data/processed/sessions"
        metadata_path = "data/processed/metadata_mapped.csv"
        caching = True
        if isinstance(model, StaticModel):
            test_dataset = StaticDataset(
                data_dir=data_dir,
                metadata_path=metadata_path,
                cache=caching
            )
        else:
            test_dataset = TemporalDataset(
                data_dir=data_dir,
                metadata_path=metadata_path,
                cache=caching
            )

        # Prepare Dataloader
        if isinstance(model, StaticModel):
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=0,  # Avoid Shared Memory Issues by Using Single Worker
                shuffle=False
            )
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=0,  # Avoid Shared Memory Issues by Using Single Worker
                shuffle=False,
                collate_fn=temporal_collate_fn
            )

        # Run Evaluation using Tester
        tester = Tester(model, test_loader)
        results = tester.evaluate()
        logger.info("Test Results:")
        pprint({k: v for k, v in results.items() if k not in ["outputs", "targets"]})

    else:
        raise ValueError(f"Unknown operation: {args.operation}")


if __name__ == "__main__":
    main()
