import torch


def temporal_collate_fn(batch):
    """
    Pads sequences per modality in the batch to the maximum length in that batch.

    Args:
        batch: List of samples from TemporalDataset.
               Each sample is a dict:
                {"text": tensor, "audio": tensor, "visual": tensor, "label": tensor, "session": int}

    Returns:
        collated dict with same keys:
            - modalities: [B, T_max, feature_dim]
            - label: [B]
            - session: [B] (list or tensor)
    """
    collated = {}

    # Process modalities
    for k in ["text", "audio", "visual"]:
        max_len = max(b[k].shape[0] for b in batch)
        feat_dim = batch[0][k].shape[1]

        padded = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
        for i, b in enumerate(batch):
            seq_len = b[k].shape[0]
            padded[i, :seq_len] = b[k]
        collated[k] = padded

    # Stack labels
    collated["label"] = torch.stack([b["label"] for b in batch])

    # Stack gender if present
    if "gender" in batch[0]:
        collated["gender"] = torch.stack([b["gender"] for b in batch])

    # Keep session IDs as a list (or convert to tensor if needed)
    collated["session"] = [b["session"] for b in batch]

    return collated
