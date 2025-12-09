"""
K-Fold Cross-Validation Trainer.

Orchestrates K independent training runs (one per fold) and aggregates results.
"""

import os
import json
import numpy as np
import pandas as pd
from loguru import logger

from src.Trainer import Trainer
from src.utility.splitting import stratified_patient_kfold


class KFoldTrainer:
    """
    Orchestrates K-fold cross-validation training.

    Runs K independent training loops, one per fold, and aggregates results.
    """

    def __init__(
        self,
        model_class,  # StaticModel or TemporalModel class (not instance)
        k_folds=5,
        batch_size=8,
        epochs=50,
        lr=1e-4,
        modalities=("text", "audio", "visual"),
        save_dir="models/kfold",
        results_dir="results/kfold",
        use_dann=False,
        dann_lambda=0.1,
        dann_alpha=1.0,
        chunk_len=None,
        chunk_hop=None,
        early_stopping_patience=10,
        early_stopping_metric="f1",
        early_stopping_mode="max",
        seed=42,
    ):
        self.model_class = model_class
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.modalities = modalities
        self.save_dir = save_dir
        self.results_dir = results_dir
        self.use_dann = use_dann
        self.dann_lambda = dann_lambda
        self.dann_alpha = dann_alpha
        self.chunk_len = chunk_len
        self.chunk_hop = chunk_hop
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self.seed = seed

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Results storage
        self.fold_results = []

    def run(self):
        """Run K-fold cross-validation training."""
        logger.info("=" * 70)
        logger.info(f"Starting {self.k_folds}-Fold Cross-Validation")
        logger.info("=" * 70)

        # Generate K folds
        fold_generator = stratified_patient_kfold(
            n_splits=self.k_folds,
            seed=self.seed,
        )

        # Train on each fold
        for fold_idx, train_sessions, val_sessions in fold_generator:
            logger.info(f"\n{'=' * 70}")
            logger.info(f"FOLD {fold_idx + 1}/{self.k_folds}")
            logger.info(f"{'=' * 70}")
            logger.info(f"Train sessions: {len(train_sessions)}")
            logger.info(f"Val sessions: {len(val_sessions)}")

            # Train this fold
            fold_result = self._train_fold(fold_idx, train_sessions, val_sessions)
            self.fold_results.append(fold_result)

        # Aggregate results
        logger.info("\n" + "=" * 70)
        logger.info("K-FOLD CROSS-VALIDATION COMPLETE")
        logger.info("=" * 70)

        self._aggregate_and_save_results()

    def _train_fold(self, fold_idx, train_sessions, val_sessions):
        """Train a single fold."""
        # Initialize fresh model
        model = self.model_class()

        # Fold-specific checkpoint name
        model_name = f"fold_{fold_idx}_model.pt"

        # Create trainer with validation
        trainer = Trainer(
            model,
            train_sessions=train_sessions,
            batch_size=self.batch_size,
            epochs=self.epochs,
            lr=self.lr,
            modalities=self.modalities,
            save_dir=self.save_dir,
            use_dann=self.use_dann,
            dann_lambda=self.dann_lambda,
            dann_alpha=self.dann_alpha,
            chunk_len=self.chunk_len,
            chunk_hop=self.chunk_hop,
            model_name=model_name,
            val_sessions=val_sessions,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_metric=self.early_stopping_metric,
            early_stopping_mode=self.early_stopping_mode,
        )

        # Run training
        trainer.run()

        # Extract final metrics
        fold_result = {
            "fold": fold_idx,
            "train_loss": trainer.train_losses[-1] if trainer.train_losses else None,
            "val_loss": trainer.val_losses[-1] if trainer.val_losses else None,
            "val_accuracy": (
                trainer.val_accuracies[-1] if trainer.val_accuracies else None
            ),
            "val_f1": trainer.val_f1_scores[-1] if trainer.val_f1_scores else None,
            "val_precision": (
                trainer.val_precisions[-1] if trainer.val_precisions else None
            ),
            "val_recall": trainer.val_recalls[-1] if trainer.val_recalls else None,
            "val_roc_auc": trainer.val_roc_aucs[-1] if trainer.val_roc_aucs else None,
            "best_epoch": trainer.best_epoch,
            "early_stopped": trainer.early_stopped,
            "epochs_trained": len(trainer.train_losses),
        }

        return fold_result

    def _aggregate_and_save_results(self):
        """Aggregate results across folds and save."""
        # Compute mean and std for each metric
        metrics = [
            "val_loss",
            "val_accuracy",
            "val_f1",
            "val_precision",
            "val_recall",
            "val_roc_auc",
            "epochs_trained",
        ]

        aggregated = {}
        for metric in metrics:
            values = [r[metric] for r in self.fold_results if r[metric] is not None]
            if values:
                aggregated[f"{metric}_mean"] = float(np.mean(values))
                aggregated[f"{metric}_std"] = float(np.std(values))

        # Log results
        logger.info("\nAggregated Results Across Folds:")
        logger.info("-" * 70)
        for key, value in aggregated.items():
            logger.info(f"{key:30s}: {value:.4f}")

        # Save to JSON
        results_path = os.path.join(self.results_dir, "kfold_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "fold_results": self.fold_results,
                    "aggregated": aggregated,
                    "config": {
                        "k_folds": self.k_folds,
                        "epochs": self.epochs,
                        "batch_size": self.batch_size,
                        "lr": self.lr,
                        "early_stopping_patience": self.early_stopping_patience,
                        "early_stopping_metric": self.early_stopping_metric,
                        "use_dann": self.use_dann,
                        "chunk_len": self.chunk_len,
                        "chunk_hop": self.chunk_hop,
                    },
                },
                f,
                indent=2,
            )

        logger.info(f"\nResults saved to {results_path}")

        # Save to CSV for easy viewing
        df = pd.DataFrame(self.fold_results)
        csv_path = os.path.join(self.results_dir, "kfold_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
