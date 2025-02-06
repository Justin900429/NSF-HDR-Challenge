import torch
from torch.utils.data import Dataset


class TorchADDataset(Dataset):
    def __init__(self, features, labels, device, add_noise=False):
        self.add_noise = add_noise
        self.device = device
        self.features = torch.from_numpy(features).to(dtype=torch.float32, device=self.device)
        self.labels = torch.from_numpy(labels).to(dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        X = self.features[index]
        if self.add_noise:
            random_noise = torch.randn_like(X) * 0.3
            X = X + random_noise

        y = self.labels[index]
        return X, y
