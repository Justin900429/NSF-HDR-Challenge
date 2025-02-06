import os
from collections import defaultdict

import torch.utils.data
from PIL import Image


class ButterflyDataset(torch.utils.data.Dataset):
    def __init__(self, root, data, unique_list, transforms):
        self.root = root
        self.unique_list = unique_list
        self.data = data

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def get_weights(self):
        count = defaultdict(int)

        for idx in range(len(self.data)):
            cur_data = self.data.iloc[idx]
            if cur_data["hybrid_stat"] == "non-hybrid":
                species = int(cur_data["subspecies"])
                count[species] += 1
            else:
                parent1 = int(cur_data["parent_subspecies_1"])
                parent2 = int(cur_data["parent_subspecies_2"])
                count[tuple(sorted([parent1, parent2]))] += 1

        max_num = max(count.values())
        weights = []

        for idx in range(len(self.data)):
            cur_data = self.data.iloc[idx]
            if cur_data["hybrid_stat"] == "non-hybrid":
                species = int(cur_data["subspecies"])
                weights.append(max_num / count[species])
            else:
                parent1 = int(cur_data["parent_subspecies_1"])
                parent2 = int(cur_data["parent_subspecies_2"])
                weights.append(max_num / count[tuple(sorted([parent1, parent2]))])

        return weights

    def get_file_path(self, data):
        return os.path.join(self.root, data["hybrid_stat"], data["filename"])

    def __getitem__(self, idx):
        cur_data = self.data.iloc[idx]

        prob = torch.tensor([0.0] * len(self.unique_list))
        if cur_data["hybrid_stat"] == "non-hybrid":
            species = int(cur_data["subspecies"])
            prob[species] = 1.0
        else:
            parent1 = int(cur_data["parent_subspecies_1"])
            parent2 = int(cur_data["parent_subspecies_2"])
            prob[parent1] = 0.5
            prob[parent2] = 0.5

        img = Image.open(self.get_file_path(cur_data)).convert("RGB")
        img = self.transforms(img)

        return img, prob
