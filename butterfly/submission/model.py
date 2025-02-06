import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip import create_model
from torchvision import transforms


class ClassifyingModel(nn.Module):
    def __init__(self, load_pretrained: bool = True, num_classes: int = 14):
        super().__init__()

        self.model = create_model(
            "hf-hub:imageomics/bioclip", require_pretrained=load_pretrained
        ).visual
        out_dim = self.model.output_dim
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
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

    def forward(self, x) -> torch.Tensor:
        features = self.model(x)
        features = F.normalize(features, dim=-1)
        out = self.fc(features)
        return out


class Model:
    def __init__(self):
        self.model = None
        self.preprocess_img = None

    def load(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model = ClassifyingModel(num_classes=14)
        model.load_state_dict(
            torch.load(
                os.path.join(os.path.dirname(__file__), "best_model.pth"),
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        self.model = model.to(self.device)

        self.preprocess_img = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.crop((450, 350, 4850, 2850))),
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def process_score(self, prob, thresh=0.75):
        scores = prob.topk(2)[0]
        max_score = scores[0].item()
        second_max_score = scores[1].item()

        if max_score > thresh:
            return 1.0 - max_score
        else:
            return max_score + second_max_score

    def predict(self, data, thresh=0.75, save_folder=None, idx=None):
        single = not isinstance(data, list)

        if single:
            image = self.preprocess_img(data).to(self.device).unsqueeze(0)
        else:
            image = torch.stack([self.preprocess_img(img).to(self.device) for img in data])
        with torch.inference_mode():
            prob = self.model(image).softmax(dim=-1).cpu()

        if save_folder is not None and idx is not None:
            torch.save(prob, os.path.join(save_folder, f"{idx}.pt"))

        if single:
            return self.process_score(prob[0], thresh=thresh)
        else:
            return [self.process_score(p, thresh=thresh) for p in prob]
