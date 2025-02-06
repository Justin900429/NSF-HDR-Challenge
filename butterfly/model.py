import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model


class ClassifyingModel(nn.Module):
    def __init__(self, load_pretrained: bool = True, num_classes: int = 14):
        super().__init__()

        self.model = create_model(
            "hf-hub:imageomics/bioclip", require_pretrained=load_pretrained
        ).visual
        out_dim = self.model.output_dim
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(
                out_dim,
                out_dim * 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(
                out_dim * 2,
                out_dim,
                bias=False,
            ),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, x):
        features = self.model(x)
        features = F.normalize(features, dim=-1)
        out = self.fc(features)
        return out
