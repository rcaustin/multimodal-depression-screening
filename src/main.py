#!/usr/bin/env python3
"""
Main entry point for multimodal depression screening model training, testing, and evaluation.

Supports three operations:
    - train: Train a model from scratch or resume from checkpoint
    - test: Evaluate a trained model on the test set
    - eval: Evaluate specific sessions with a trained model
"""
import argparse
from pprint import pprint
from typing import Tuple

from loguru import logger

from src.Evaluator import Evaluator
from src.StaticModel import StaticModel
from src.TemporalModel import TemporalModel
from src.Tester import Tester
from src.Trainer import Trainer

# ============================================================================
# CONSTANTS
# ============================================================================

BATCH_SIZE = 8
LEARNING_RATE = 1e-4

# Chunking configuration (for temporal models)
CHUNK_LENGTH_FRAMES = 120  # 4 seconds at 30Hz
CHUNK_HOP_FRAMES = 60  # 2 seconds at 30Hz


# ============================================================================
# ARGUMENT PARSING
# ============================================================================


def parse_args():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train, test, or evaluate multimodal depression screening models"
    )

    # Required arguments
    parser.add_argument(
        "operation",
        choices=["train", "test", "eval"],
        help="Operation mode: 'train', 'test', or 'eval'",
    )

    parser.add_argument(
        "model",
        choices=["static", "temporal", "DANN"],
        help="Model type: 'static' (session-level), 'temporal' (sequence), or 'DANN' (domain-adversarial)",
    )

    # Optional arguments
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Checkpoint name (without .pt extension). "
        "For training: names the saved model. "
        "For testing/eval: loads the specified checkpoint. "
        "If not provided, uses default naming based on model type.",
    )

    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Enable temporal chunking (4s windows, 2s hop) for temporal models. "
        "Ignored for static models.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50). Only used for training.",
    )

    parser.add_argument(
        "--sessions",
        type=str,
        nargs="+",
        help="Session ID(s) to evaluate (for 'eval' operation only). "
        "Accepts one or more session IDs separated by spaces.",
    )

    return parser.parse_args()


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================


def configure_chunking(args) -> Tuple[int | None, int | None]:
    """
    Configure chunking parameters based on arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Tuple of (chunk_len, chunk_hop) or (None, None) if chunking disabled
    """
    # Chunking only applies to temporal models
    if args.chunk and args.model != "static":
        logger.info(
            f"Temporal chunking enabled: {CHUNK_LENGTH_FRAMES} frames "
            f"({CHUNK_LENGTH_FRAMES/30:.1f}s) windows with "
            f"{CHUNK_HOP_FRAMES} frames ({CHUNK_HOP_FRAMES/30:.1f}s) hop."
        )
        return CHUNK_LENGTH_FRAMES, CHUNK_HOP_FRAMES

    # Warn if user tried to enable chunking for static model
    if args.model == "static" and args.chunk:
        logger.warning("Chunking option ignored for static model.")

    return None, None


def initialize_model(args):
    """
    Initialize the appropriate model based on arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Initialized model instance (StaticModel or TemporalModel)
    """
    if args.model == "static":
        logger.info("Initializing StaticModel (session-level features)")
        return StaticModel()
    else:
        # Both 'temporal' and 'DANN' use TemporalModel
        # DANN is a training mode, not a different architecture
        logger.info("Initializing TemporalModel (sequence-to-sequence)")
        return TemporalModel()


# ============================================================================
# OPERATION HANDLERS
# ============================================================================


def run_training(model, args, use_dann: bool, chunk_len: int | None, chunk_hop: int | None):
    """
    Execute model training.

    Args:
        model: Model instance to train
        args: Parsed command-line arguments
        use_dann: Whether to use domain-adversarial training
        chunk_len: Chunk length in frames (None if disabled)
        chunk_hop: Chunk hop in frames (None if disabled)
    """
    logger.info("Starting training...")

    trainer = Trainer(
        model,
        batch_size=BATCH_SIZE,
        epochs=args.epochs,
        lr=LEARNING_RATE,
        use_dann=use_dann,
        chunk_len=chunk_len,
        chunk_hop=chunk_hop,
        model_name=args.name,
    )
    trainer.run()

    logger.info("Training complete.")


def run_testing(model, args, use_dann: bool, chunk_len: int | None, chunk_hop: int | None):
    """
    Execute model testing on the test set.

    Args:
        model: Model instance to test
        args: Parsed command-line arguments
        use_dann: Whether model was trained with DANN
        chunk_len: Chunk length in frames (None if disabled)
        chunk_hop: Chunk hop in frames (None if disabled)
    """
    logger.info("Starting testing on test set...")

    try:
        tester = Tester(
            model,
            batch_size=BATCH_SIZE,
            use_dann=use_dann,
            chunk_len=chunk_len,
            chunk_hop=chunk_hop,
            ckpt_name=args.name,
        )
        results = tester.evaluate()

        # Display results (excluding raw outputs and targets)
        logger.info("Test Results:")
        pprint({k: v for k, v in results.items() if k not in ["outputs", "targets"]})

    except FileNotFoundError:
        logger.error(
            f"Checkpoint not found for {args.model} model. "
            f"Train the model first using: python -m src.main train {args.model}"
        )


def run_evaluation(model, args, use_dann: bool, chunk_len: int | None, chunk_hop: int | None):
    """
    Execute evaluation on specific sessions.

    Args:
        model: Model instance to evaluate with
        args: Parsed command-line arguments
        use_dann: Whether model was trained with DANN
        chunk_len: Chunk length in frames (None if disabled)
        chunk_hop: Chunk hop in frames (None if disabled)
    """
    # Validate that session IDs were provided
    if not args.sessions:
        logger.error(
            "No session IDs provided. "
            "Use --sessions followed by one or more session IDs."
        )
        return

    logger.info(f"Evaluating {len(args.sessions)} session(s): {args.sessions}")

    try:
        evaluator = Evaluator(
            model,
            session_ids=args.sessions,
            ckpt_name=args.name,
            use_dann=use_dann,
            chunk_len=chunk_len,
            chunk_hop=chunk_hop,
        )
        results = evaluator.evaluate()
        evaluator.print_results(results)

    except FileNotFoundError:
        logger.error(
            f"Checkpoint not found for {args.model} model. "
            f"Train the model first using: python -m src.main train {args.model}"
        )
    except ValueError as e:
        logger.error(f"Evaluation error: {e}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    args = parse_args()

    logger.info("=" * 70)
    logger.info(f"Operation: {args.operation.upper()}")
    logger.info(f"Model Type: {args.model}")
    logger.info("=" * 70)

    # Determine if using Domain-Adversarial Neural Network (DANN)
    # DANN is a training technique, not a separate model architecture
    use_dann = args.model == "DANN"
    if use_dann:
        logger.info("Domain-Adversarial Neural Network (DANN) mode enabled")

    # Configure chunking for temporal models
    chunk_len, chunk_hop = configure_chunking(args)

    # Initialize the appropriate model
    model = initialize_model(args)

    # Dispatch to the appropriate operation handler
    if args.operation == "train":
        run_training(model, args, use_dann, chunk_len, chunk_hop)

    elif args.operation == "test":
        run_testing(model, args, use_dann, chunk_len, chunk_hop)

    elif args.operation == "eval":
        run_evaluation(model, args, use_dann, chunk_len, chunk_hop)

    else:
        raise ValueError(f"Unknown operation: {args.operation}")


if __name__ == "__main__":
    main()
