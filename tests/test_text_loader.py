import os
import numpy as np
import torch
import tempfile
from src.loaders.temporal.TextLoader import TextLoader

def _mk_session(tmpdir: str, name: str) -> str:
    path = os.path.join(tmpdir, name)
    os.makedirs(path, exist_ok=True)
    return path

def test_text_loader_npy_and_cache():
    with tempfile.TemporaryDirectory() as tmp:
        sess = _mk_session(tmp, "S001")
        arr = np.random.randn(10, 32).astype(np.float32)
        np.save(os.path.join(sess, "text_embeddings.npy"), arr)

        loader = TextLoader(cache=True)
        t1 = loader.load(sess)
        assert isinstance(t1, torch.Tensor)
        assert tuple(t1.shape) == (10, 32)

        # mutate file to check cache
        arr2 = np.random.randn(8, 32).astype(np.float32)
        np.save(os.path.join(sess, "text_embeddings.npy"), arr2)
        t2 = loader.load(sess)
        assert tuple(t2.shape) == (10, 32)  # still cached

def test_text_loader_csv_norm_and_hop():
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        sess = _mk_session(tmp, "S002")
        arr = np.random.randn(25, 16).astype(np.float32)
        pd.DataFrame(arr).to_csv(os.path.join(sess, "text_embeddings.csv"), header=False, index=False)

        loader = TextLoader(embedding_glob="text_embeddings.csv", normalize=True, frame_hop=3, cache=False)
        t = loader.load(sess)
        assert isinstance(t, torch.Tensor)
        assert t.shape[1] == 16
        assert t.shape[0] == 9  # 25 / 3
        assert torch.allclose(t.mean(0), torch.zeros(16), atol=1e-1)

def test_missing_files_raise():
    with tempfile.TemporaryDirectory() as tmp:
        sess = _mk_session(tmp, "S003")
        loader = TextLoader(embedding_glob="does_not_exist.npy", transcript_glob="nope.txt")
        import pytest
        with pytest.raises(FileNotFoundError):
            loader.load(sess)
