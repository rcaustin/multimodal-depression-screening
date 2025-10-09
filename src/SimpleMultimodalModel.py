import torch
import torch.nn as nn


class SimpleMultimodalModel(nn.Module):
    def __init__(
            self, text_dim=384, audio_dim=88, visual_dim=17, hidden_dim=256
    ):
        super().__init__()
        self.text_fc = nn.Linear(text_dim, hidden_dim)
        self.audio_fc = nn.Linear(audio_dim, hidden_dim)
        self.visual_fc = nn.Linear(visual_dim, hidden_dim)

        self.fusion_fc = nn.Linear(hidden_dim*3, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1)  # regr. or 1-class sigmoid

    def forward(self, text, audio, visual):
        t = torch.relu(self.text_fc(text))
        a = torch.relu(self.audio_fc(audio))
        v = torch.relu(self.visual_fc(visual))

        fused = torch.cat([t, a, v], dim=1)
        x = torch.relu(self.fusion_fc(fused))
        out = self.output_fc(x)
        return out
