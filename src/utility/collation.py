import torch


def temporal_collate_fn(batch):
    """
    Pads sequences per modality in the batch to the maximum length in that batch.

    Args:
        batch: List of samples from TemporalDataset.
               Each sample is a dict:
                {"text": tensor, "audio": tensor, "visual": tensor, "label": tensor}

    Returns:
        collated dict with same keys:
            - modalities: [B, T_max, feature_dim]
            - label: [B]
    """
    keys = batch[0].keys()
    collated = {}

    for k in keys:
        if k == "label":
            collated[k] = torch.stack([b[k] for b in batch])
        else:
            # max sequence length in this batch for this modality
            max_len = max(b[k].shape[0] for b in batch)
            feat_dim = batch[0][k].shape[1]

            padded = torch.zeros((len(batch), max_len, feat_dim), dtype=torch.float32)
            for i, b in enumerate(batch):
                seq_len = b[k].shape[0]
                padded[i, :seq_len] = b[k]
            collated[k] = padded

    return collated
