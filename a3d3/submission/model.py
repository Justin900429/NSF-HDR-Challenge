import os

import torch
import torch.nn as nn


class ADClassifier(nn.Module):
    def __init__(self):
        super(ADClassifier, self).__init__()

        self.dim = 256
        self.n_head = 4
        self.model = nn.Sequential(
            nn.Conv1d(2, self.dim, 3, stride=2, padding=1, bias=False),  # 100
            nn.BatchNorm1d(self.dim),
            nn.ReLU(),
            nn.Conv1d(self.dim, self.dim, 3, stride=2, padding=1, bias=False),  # 50
            nn.BatchNorm1d(self.dim),
            nn.ReLU(),
            nn.Conv1d(self.dim, self.dim, 3, stride=2, padding=1, bias=False),  # 25
            nn.BatchNorm1d(self.dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.dim, 128, bias=False), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.model(x)
        x = self.fc(x.mean(dim=-1))

        return x


class Model:
    def __init__(self):
        super().__init__()
        self.model = ADClassifier()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.inference_mode()
    def predict(self, x):
        batch_size = 16
        total_res = []
        for start_idx in range(0, len(x), batch_size):
            end_idx = min(start_idx + batch_size, len(x))
            tensor_x = torch.tensor(x[start_idx:end_idx], dtype=torch.float32, device=self.device)
            temp_res = (1 - self.model(tensor_x).sigmoid()).cpu().squeeze().tolist()
            total_res.extend(temp_res)
        return total_res

    def load(self):
        model_file = os.path.join(os.path.dirname(__file__), "best_model.pth")
        self.model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
        self.model.to(self.device)
