import torch


def align_to_grid(modality_list, timestamp_list, step_hz=30):
    """
    Align multiple temporal sequences to a common grid with zero-padding.

    Args:
        modality_list: list of torch.Tensor [seq_len, feat_dim] per modality
        timestamp_list: list of torch.Tensor [seq_len] with absolute timestamps (seconds)
        grid_hz: temporal grid frequency (Hz)

    Returns:
        aligned_list: list of torch.Tensor [grid_len, feat_dim] per modality
    """
    # Determine overall grid range: from t=0 to max timestamp across all modalities
    max_time = max([ts[-1].item() if ts.numel() > 0 else 0.0 for ts in timestamp_list])
    grid_times = torch.arange(0, max_time + 1e-8, 1 / step_hz)
    grid_len = len(grid_times)

    aligned_list = []

    for seq, ts in zip(modality_list, timestamp_list):
        feat_dim = seq.size(1)
        aligned = torch.zeros((grid_len, feat_dim), dtype=seq.dtype, device=seq.device)

        if ts.numel() == 0:
            # modality has no data at all, keep zeros
            aligned_list.append(aligned)
            continue

        # Index into original sequence
        idx = 0
        for i, t in enumerate(grid_times):
            # Advance idx while next timestamp <= current grid time
            while idx + 1 < len(ts) and ts[idx + 1] <= t:
                idx += 1

            # If current timestamp <= grid time, copy value; else leave zeros
            if ts[idx] <= t:
                aligned[i] = seq[idx]

        aligned_list.append(aligned)

    return aligned_list
