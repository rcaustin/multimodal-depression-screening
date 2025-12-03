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

    # Get the lengths before padding
    lengths = torch.tensor(
        [b["visual"].shape[0] for b in batch], dtype=torch.long)
    collated["lengths"] = lengths

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

def chunked_temporal_collate_fn(batch):
    """
    Collate function for chunked temporal data.

    Assumes each sample in batch is a dict, Temporal modalities are 2D tensors [T, D], label is scalar tensor.
    
    Returns:
        dict with:
            - temporal tensors stacked to [B, T, D]
            - label/gender stacked to [B]
            -session/start kept as lists
    """
    collated = {}

    # Get the keys from the first sample
    keys = batch[0].keys()

    for k in keys:
        v0 = batch[0][k]

        # Handle 2D temporal tensors
        if isinstance(v0, torch.Tensor) and v0.dim() == 2:
            # In chunked mode, T is the same for all items. Just stack.
            collated[k] = torch.stack([b[k] for b in batch])  # [B, T, D]

        # Handle scalar tensors
        elif isinstance(v0, torch.Tensor):
            collated[k] = torch.stack([b[k] for b in batch])  # [B]

        # Handle non-tensor items (e.g., session IDs, start indices)
        else:
            collated[k] = [b[k] for b in batch]

    return collated
