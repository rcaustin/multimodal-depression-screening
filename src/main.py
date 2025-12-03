#!/usr/bin/env python3
import argparse
from pprint import pprint

from loguru import logger

from src.StaticModel import StaticModel
from src.TemporalModel import TemporalModel
from src.Tester import Tester
from src.Trainer import Trainer
from src.Evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or test multimodal depression models"
    )

    parser.add_argument(
        "operation",
        choices=["train", "test", "eval"],
        help="Operation mode: 'train', 'test', or 'eval'",
    )

    parser.add_argument(
        "model",
        choices=["static", "temporal", "DANN"],
        help="Model type to use: 'static', 'temporal', or 'DANN'",
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of a saved checkpoint. \
              Path assumes 'models/{name}.pt', .pt automatically added if missing. \
              If not provided, defaults used based on model type. \
              For training, this names the saved model. For testing, this loads the model.",
    )

    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Enable temporal chunking (4s windows, 2s hop) for temporal models.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )

    parser.add_argument(
        "--sessions",
        type=str,
        nargs="+",
        help="Session ID(s) to evaluate (for 'eval' operation only). \
              Accepts one or more session IDs separated by spaces.",
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
        CHUNK_LEN = 120  # 4 seconds at 30Hz
        CHUNK_HOP = 60  # 2 seconds at 30Hz
        logger.info("Temporal chunking enabled: 4s windows with 2s hop.")
    else:
        CHUNK_LEN = None
        CHUNK_HOP = None
        if args.model == "static" and args.chunk:
            logger.warning("Chunking option ignored for static model.")

    # Initialize Model
    model = StaticModel() if args.model == "static" else TemporalModel()

    # Training Branch
    if args.operation == "train":
        trainer = Trainer(
            model,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            lr=LR,
            use_dann=USE_DANN,
            chunk_len=CHUNK_LEN,
            chunk_hop=CHUNK_HOP,
            model_name=args.name,
        )
        trainer.run()

    # Testing Branch
    elif args.operation == "test":
        try:
            tester = Tester(
                model,
                batch_size=BATCH_SIZE,
                use_dann=USE_DANN,
                chunk_len=CHUNK_LEN,
                chunk_hop=CHUNK_HOP,
                ckpt_name=args.name,
            )
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

    # Evaluation Branch (specific sessions)
    elif args.operation == "eval":
        if not args.sessions:
            logger.error(
                "No session IDs provided. Use --sessions to specify one or more session IDs."
            )
            return

        try:
            evaluator = Evaluator(
                model,
                session_ids=args.sessions,
                ckpt_name=args.name,
                use_dann=USE_DANN,
                chunk_len=CHUNK_LEN,
                chunk_hop=CHUNK_HOP,
            )
            results = evaluator.evaluate()
            evaluator.print_results(results)
        except FileNotFoundError:
            logger.warning(
                f"Checkpoint not found for {args.model} model. "
                f"Try running '{args.model} train' first."
            )
        except ValueError as e:
            logger.error(f"Evaluation error: {e}")

    else:
        raise ValueError(f"Unknown operation: {args.operation}")


if __name__ == "__main__":
    main()
