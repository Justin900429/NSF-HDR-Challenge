import io
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_base


def to_numpy_image(sla):
    lon, lat = np.meshgrid(range(sla.shape[2]), range(sla.shape[1]))
    sla_clipped = np.clip(sla[0], -1, 1)
    fig, ax = plt.subplots(figsize=(1, 1), dpi=512)
    ax = plt.gca()
    ax.axis("off")
    plt.contourf(lon, lat, sla_clipped, cmap="bwr", vmin=-1, vmax=1)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    buf = io.BytesIO()
    fig.savefig(buf, format="raw", dpi=512, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(buf.getvalue(), dtype=np.uint8),
        (int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    buf.close()
    plt.close(fig)
    return img_arr


class Model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = convnext_base(weights=None)
        feature_num = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            model.classifier[0], nn.Flatten(1), nn.Linear(feature_num, 1)
        )
        self.model = model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(
            torch.load(
                os.path.join(os.path.dirname(__file__), "best_model.pth"), map_location=self.device
            )
        )
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, x):
        img = Image.fromarray(to_numpy_image(x)).convert("RGB")
        img = self.transforms(img)
        with torch.inference_mode():
            output = self.model(img.unsqueeze(0).to(self.device)).sigmoid().cpu().numpy()
        output = (output > 0.3).astype(int)
        return output
