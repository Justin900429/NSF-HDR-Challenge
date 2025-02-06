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
