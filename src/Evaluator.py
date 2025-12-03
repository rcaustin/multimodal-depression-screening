import os
from typing import List, Dict

import pandas as pd
import torch
from loguru import logger

from src.datasets.StaticDataset import StaticDataset
from src.datasets.TemporalDataset import TemporalDataset
from src.StaticModel import StaticModel
from src.TemporalModel import TemporalModel


class Evaluator:
    """
    Evaluates specific sessions using a trained multimodal model.

    Handles:
        - Checkpoint loading
        - Single or multiple session evaluation
        - Device placement
        - Pretty-printed result output

    Args:
        model (torch.nn.Module): Model to evaluate
        session_ids (List[str]): List of session IDs to evaluate
        ckpt_name (str | None): Name of checkpoint to load
        use_dann (bool): Whether model was trained with DANN
        chunk_len (int | None): Chunk length if using chunking
        chunk_hop (int | None): Chunk hop if using chunking
    """

    def __init__(
        self,
        model: torch.nn.Module,
        session_ids: List[str],
        ckpt_name: str | None = None,
        use_dann: bool = False,
        chunk_len: int | None = None,
        chunk_hop: int | None = None,
        data_dir: str = "data/processed/sessions",
        metadata_path: str = "data/processed/metadata_mapped.csv",
    ):
        self.device: str = "cpu"
        self.model = model.to(self.device)
        self.session_ids = [str(sid) for sid in session_ids]
        self.use_dann = use_dann
        self.chunk_len = chunk_len
        self.chunk_hop = chunk_hop
        self.data_dir = data_dir
        self.metadata_path = metadata_path

        # Load metadata
        self.metadata = pd.read_csv(metadata_path)

        # Validate sessions exist
        self._validate_sessions()

        # Determine Checkpoint Path
        if ckpt_name is not None:
            if not ckpt_name.endswith(".pt"):
                ckpt_name += ".pt"
            self.checkpoint_path = f"models/{ckpt_name}"
        else:
            # Fallback to default naming
            if isinstance(self.model, StaticModel):
                self.checkpoint_path = "models/static_model.pt"
            elif isinstance(self.model, TemporalModel):
                if self.use_dann:
                    self.checkpoint_path = "models/temporal_model_dann.pt"
                else:
                    self.checkpoint_path = "models/temporal_model.pt"
            else:
                raise ValueError(f"Unknown Model Type: {type(self.model)}")

        # Load Checkpoint
        self._load_checkpoint()

    def _validate_sessions(self):
        """Validate that all requested sessions exist in the dataset."""
        for session_id in self.session_ids:
            session_dir = os.path.join(self.data_dir, session_id)
            if not os.path.isdir(session_dir):
                raise ValueError(
                    f"Session {session_id} not found in {self.data_dir}. "
                    f"Please ensure the session directory exists."
                )

            # Check if session is in metadata
            matching_rows = self.metadata.loc[
                self.metadata["Participant_ID"].astype(str) == session_id
            ]
            if len(matching_rows) == 0:
                logger.warning(
                    f"Session {session_id} not found in metadata. "
                    f"Ground truth label will not be available."
                )

    def _load_checkpoint(self):
        """Load model weights from checkpoint."""
        logger.info(f"Loading Checkpoint From {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Infer chunking parameters if not explicitly set
        ckpt_chunk_len = checkpoint.get("chunk_len", None)
        ckpt_chunk_hop = checkpoint.get("chunk_hop", None)

        if self.chunk_len is None and ckpt_chunk_len is not None:
            self.chunk_len = ckpt_chunk_len

        if self.chunk_hop is None and ckpt_chunk_hop is not None:
            self.chunk_hop = ckpt_chunk_hop

        logger.info(
            f"Evaluator chunk_len: {self.chunk_len}, chunk_hop: {self.chunk_hop}"
        )

    @torch.no_grad()
    def evaluate(self) -> List[Dict]:
        """
        Evaluate all requested sessions.

        Returns:
            List of dictionaries containing results for each session:
            {
                'session_id': str,
                'probability': float,
                'prediction': str ('Depressed' or 'Not Depressed'),
                'ground_truth': str or None,
                'logit': float
            }
        """
        self.model.eval()

        results = []

        for session_id in self.session_ids:
            session_result = self._evaluate_session(session_id)
            results.append(session_result)

        return results

    def _evaluate_session(self, session_id: str) -> Dict:
        """Evaluate a single session."""
        # Load features based on model type
        if isinstance(self.model, StaticModel):
            # Use StaticDataset for single session
            dataset = StaticDataset(
                [session_id], data_dir=self.data_dir, metadata_path=self.metadata_path
            )
            sample = dataset[0]

            # Move to device
            text = sample.get("text")
            audio = sample.get("audio")
            visual = sample.get("visual")

            if text is not None:
                text = text.unsqueeze(0).to(self.device)
            if audio is not None:
                audio = audio.unsqueeze(0).to(self.device)
            if visual is not None:
                visual = visual.unsqueeze(0).to(self.device)

            # Run inference
            output = self.model(text, audio, visual)
            logit = output.squeeze().item()

        else:  # TemporalModel
            # Use TemporalDataset
            dataset = TemporalDataset(
                [session_id],
                data_dir=self.data_dir,
                metadata_path=self.metadata_path,
                chunk_len=self.chunk_len,
                chunk_hop=self.chunk_hop,
            )

            # Handle chunked vs full session
            if self.chunk_len is not None:
                # Aggregate predictions across chunks
                logits = []
                for i in range(len(dataset)):
                    sample = dataset[i]
                    text_seq = sample["text"].unsqueeze(0).to(self.device)
                    audio_seq = sample["audio"].unsqueeze(0).to(self.device)
                    visual_seq = sample["visual"].unsqueeze(0).to(self.device)

                    output = self.model(text_seq, audio_seq, visual_seq)
                    logits.append(output.squeeze().item())

                # Mean logit across chunks
                logit = sum(logits) / len(logits) if logits else 0.0
            else:
                # Full session
                sample = dataset[0]
                text_seq = sample["text"].unsqueeze(0).to(self.device)
                audio_seq = sample["audio"].unsqueeze(0).to(self.device)
                visual_seq = sample["visual"].unsqueeze(0).to(self.device)

                output = self.model(text_seq, audio_seq, visual_seq)
                logit = output.squeeze().item()

        # Convert logit to probability
        probability = torch.sigmoid(torch.tensor(logit)).item()

        # Binary prediction
        prediction = "Depressed" if probability >= 0.5 else "Not Depressed"

        # Get ground truth label if available
        matching_rows = self.metadata.loc[
            self.metadata["Participant_ID"].astype(str) == session_id
        ]
        if len(matching_rows) > 0:
            phq_binary = matching_rows.iloc[0]["PHQ_Binary"]
            ground_truth = "Depressed" if phq_binary == 1.0 else "Not Depressed"
        else:
            ground_truth = None

        return {
            "session_id": session_id,
            "probability": probability,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "logit": logit,
        }

    def print_results(self, results: List[Dict]):
        """Pretty-print evaluation results."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)

        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Session: {result['session_id']}")
            print("-" * 70)
            print(f"  Prediction:    {result['prediction']}")
            print(
                f"  Probability:   {result['probability']:.4f} ({result['probability']*100:.2f}%)"
            )
            print(f"  Logit:         {result['logit']:.4f}")

            if result["ground_truth"] is not None:
                print(f"  Ground Truth:  {result['ground_truth']}")
                correct = "✓" if result["prediction"] == result["ground_truth"] else "✗"
                print(f"  Correct:       {correct}")
            else:
                print("  Ground Truth:  Not Available")

        print("\n" + "=" * 70)
        print(f"Total Sessions Evaluated: {len(results)}")

        # Summary statistics if ground truth available
        with_gt = [r for r in results if r["ground_truth"] is not None]
        if with_gt:
            correct_count = sum(
                1 for r in with_gt if r["prediction"] == r["ground_truth"]
            )
            accuracy = correct_count / len(with_gt)
            print(f"Accuracy: {correct_count}/{len(with_gt)} ({accuracy*100:.2f}%)")

        print("=" * 70 + "\n")
