import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DepressionDataset(Dataset):
    def __init__(self, sessions_list, root_dir, transform=None):
        """
        Args:
            sessions_list (list): list of session IDs (like ['300', '301'])
            root_dir (string): base directory for processed sessions
        """
        self.sessions_list = sessions_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.sessions_list)

    def __getitem__(self, idx):
        session_id = self.sessions_list[idx]
        session_path = f"{self.root_dir}/{session_id}"

        # Text Features
        transcript_path = f"{session_path}/{session_id}_Transcript.csv"
        text_df = pd.read_csv(transcript_path)
        text_data = " ".join(text_df['Text'].tolist())

        # Audio Features
        audio_path = (
            f"{session_path}/features/{session_id}"
            f"_BoAW_openSMILE_2.3.0_eGeMAPS.csv"
        )
        audio_features = pd.read_csv(audio_path).values.astype(np.float32)

        # Visual Features
        visual_path = f"{session_path}/features/{session_id}_CNN_VGG.mat"
        import scipy.io
        mat = scipy.io.loadmat(visual_path)
        visual_features = mat['features']  # Assuming 'features' is the key

        # Convert to PyTorch Tensors
        text_tensor = torch.tensor(text_data)  # Placeholder for text features
        audio_tensor = torch.tensor(audio_features)
        visual_tensor = torch.tensor(visual_features)

        # Label
        # Example: depression label (0/1) or PHQ score
        label = torch.tensor(0.0)  # Placeholder for label

        return text_tensor, audio_tensor, visual_tensor, label
