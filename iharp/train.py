import argparse
import glob
import os
from random import seed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import convnext_base
from tqdm import tqdm


class TideAnomalyDataset(Dataset):
    def __init__(self, image_folder, csv_files):
        csv_files.sort()
        self.image_folder = image_folder
        self.data = []
        self.locations = {}
        self.anomalies = {}
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        added = set()
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            dates = df["t"].unique()

            for date in dates:
                image_filename = f"{date}.png"
                image_path = os.path.join(self.image_folder, image_filename)
                if not os.path.exists(image_path):
                    continue
                daily_data = df[df["t"] == date]
                locations = daily_data[["latitude", "longitude"]].values
                anomalies = daily_data["anomaly"].values

                if date not in added:
                    added.add(date)
                    self.data.append(image_path)
                if date not in self.locations.keys():
                    self.locations[date] = []
                    self.anomalies[date] = []

                self.locations[date].append(locations)
                self.anomalies[date].append(anomalies)

        self.date_keys = sorted(list(self.locations.keys()))

    def get_weights(self):
        from collections import defaultdict

        count = defaultdict(int)
        for idx in range(len(self.date_keys)):
            date = self.date_keys[idx]
            anomalies = np.array(self.anomalies[date])
            anomaly = (anomalies > 0).any()
            count[anomaly] += 1

        max_count = max(count.values())
        weight = []
        for idx in range(len(self.date_keys)):
            date = self.date_keys[idx]
            anomalies = np.array(self.anomalies[date])
            anomaly = (anomalies > 0).any()
            weight.append(max_count / count[anomaly])

        return weight

    def __len__(self):
        return len(self.date_keys)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        date = self.date_keys[idx]
        locations = torch.FloatTensor(np.array(self.locations[date])).squeeze()
        anomalies = torch.FloatTensor(np.array(self.anomalies[date])).squeeze()

        return image, locations, anomalies


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--csv_folder", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    csv_files = sorted(list(glob.glob(os.path.join(args.csv_folder, "*.csv"))))
    dataset = TideAnomalyDataset(args.image_folder, csv_files)
    get_weights = dataset.get_weights()

    train_dataset = torch.utils.data.Subset(dataset, range(5593))
    test_dataset = torch.utils.data.Subset(dataset, range(5593, len(dataset)))
    leaderboard_dataset = torch.utils.data.Subset(dataset, range(3867, len(dataset)))

    sampler = torch.utils.data.WeightedRandomSampler(
        get_weights[:5593], len(train_dataset), replacement=True, generator=None
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    leaderboard_loader = DataLoader(leaderboard_dataset, batch_size=16)

    model = convnext_base(weights=True)
    feature_num = model.classifier[-1].in_features
    model.classifier = nn.Sequential(model.classifier[0], nn.Flatten(1), nn.Linear(feature_num, 12))
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        for images, _, anomalies in tqdm(train_loader, total=len(train_loader)):
            images, anomalies = (
                images.to(device),
                anomalies.to(device),
            )
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, anomalies)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            temp_output = []
            gt = []
            for images, locations, anomalies in test_loader:
                images, anomalies = (
                    images.to(device),
                    anomalies.to(device),
                )
                outputs = model(images)
                loss = criterion(outputs, anomalies)
                val_loss += loss.item()
                temp_output.append((outputs.sigmoid().cpu().numpy() > 0.3).any(axis=1))
                gt.append(anomalies.cpu().numpy().any(axis=1))

            temp_output = np.concatenate(temp_output, axis=0)
            gt = np.concatenate(gt, axis=0)
            f1 = f1_score(gt, temp_output)

            leaderboard_temp_output = []
            leaderboard_gt = []
            for images, locations, anomalies in leaderboard_loader:
                images, anomalies = (
                    images.to(device),
                    anomalies.to(device),
                )
                outputs = model(images)
                loss = criterion(outputs, anomalies)
                val_loss += loss.item()
                leaderboard_temp_output.append((outputs.sigmoid().cpu().numpy() > 0.3).any(axis=1))
                leaderboard_gt.append(anomalies.cpu().numpy().any(axis=1))

            leaderboard_temp_output = np.concatenate(leaderboard_temp_output, axis=0)
            leaderboard_gt = np.concatenate(leaderboard_gt, axis=0)
            leaderboard_f1 = f1_score(leaderboard_gt, leaderboard_temp_output)

        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val f1 score: {f1:.4f}, Leaderboard f1 score: {leaderboard_f1:.4f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "best_model.pth")
