from collections.abc import Iterator, Sequence

import accelerate
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.data
import tyro
from dataset import ButterflyDataset
from model import ClassifyingModel
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm


class DistributedWeightedRandomSampler(torch.utils.data.Sampler):
    weights: torch.Tensor
    num_samples: int
    replacement: bool

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        replacement: bool = True,
        generator: torch.Generator = None,
    ) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError(
                f"num_samples should be a positive integer value, but got num_samples={num_samples}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )

        self.indices = torch.randperm(num_samples, generator=generator)
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)[self.indices]
        if len(weights_tensor.shape) != 1:
            raise ValueError(
                "weights should be a 1d sequence but given "
                f"weights have shape {tuple(weights_tensor.shape)}"
            )
        self.mask = torch.ones_like(weights_tensor).bool()

        if dist.is_initialized():
            num_processes = dist.get_world_size()
            if num_processes > 1:
                assert generator is not None, "A generator should be set when num_processes > 1"
                self.mask = torch.zeros_like(weights_tensor)
                rank_indices = [
                    i for i in range(len(self.mask)) if i % num_processes == dist.get_rank()
                ]
                self.mask[rank_indices] = 1
                self.mask = self.mask.bool()
        else:
            num_processes = 1

        self.weights = weights_tensor
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(
            self.weights[self.mask],
            self.num_samples,
            self.replacement,
            generator=self.generator,
        )
        rank_indices = self.indices[self.mask]
        rand_indices = rank_indices[rand_tensor]
        yield from iter(rand_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def main(
    root: str,
    csv_file: str,
    save_place: str = "checkpoints",
    accum_steps: int = 1,
    test_size: float = 0.2,
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 3e-4,
    num_workers: int = 8,
    seed: int = 0,
):
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=accum_steps,
    )
    accelerate.utils.set_seed(seed, device_specific=True)

    train_transforms = transforms.Compose(
        [
            transforms.Lambda(
                lambda x: x.crop((500, 400, 4800, 2800))
            ),  # Remove the unused information in a hurestics way
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=(-5, 5),
                        translate=(0.02, 0.02),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    )
                ]
            ),
            transforms.RandomCrop((224, 224)),
            transforms.RandomApply(
                [transforms.ColorJitter(contrast=0.2, hue=0.25, brightness=0.3)]
            ),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.crop((500, 400, 4800, 2800))),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    with accelerator.main_process_first():
        data = pd.read_csv(csv_file)
    data = data.fillna(0)
    data["subspecies"] = data["subspecies"].astype("int").astype("category")
    data["parent_subspecies_1"] = data["parent_subspecies_1"].astype("int").astype("category")
    data["parent_subspecies_2"] = data["parent_subspecies_2"].astype("int").astype("category")

    hybrid_data = data[data["hybrid_stat"] == "hybrid"]
    non_hybrid_data = data[data["hybrid_stat"] == "non-hybrid"]

    unique_list = data["subspecies"].unique()

    train_hybrid, test_hybrid = train_test_split(hybrid_data, test_size=test_size)

    # non_hybrid_data = non_hybrid_data.sample(n=train_hybrid * 3).reset_index(drop=True)
    train_data = pd.concat([train_hybrid, non_hybrid_data]).reset_index(drop=True)
    test_data = test_hybrid.reset_index(drop=True)  # If hybrid data is great, we're all good

    train_dataset = ButterflyDataset(root, train_data, unique_list, train_transforms)
    test_dataset = ButterflyDataset(root, test_data, unique_list, test_transforms)

    generator = torch.Generator()
    train_sampler = DistributedWeightedRandomSampler(
        train_dataset.get_weights(), len(train_dataset), generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=train_sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=False,
    )

    with accelerator.main_process_first():
        model = ClassifyingModel(num_classes=len(train_dataset.unique_list))

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": model.fc.parameters(), "lr": lr},
            {"params": model.model.parameters(), "lr": lr * 0.01},
        ],
        weight_decay=1e-5,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(30, epochs, 30)), 0.7)
    model, optimizer, lr_scheduler, train_loader, test_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader, test_loader
    )

    best_test_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        step_loss = 0
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_main_process,
            leave=False,
        ) as pbar:
            for images, labels in train_loader:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    model_output = model(images)
                    loss = torch.nn.functional.cross_entropy(model_output, labels)
                    accelerator.backward(loss)
                    optimizer.step()
                    step_loss += accelerator.gather_for_metrics(loss).detach().mean().item()

                if accelerator.sync_gradients:
                    step_loss /= accum_steps
                    pbar.set_postfix({"loss": step_loss})
                    accelerator.log({"train_loss": step_loss})
                    step_loss = 0

                pbar.update(1)
        lr_scheduler.step()

        model.eval()
        test_loss = 0
        with tqdm(
            total=len(test_loader),
            desc=f"Validation {epoch}",
            disable=not accelerator.is_main_process,
            leave=False,
        ) as pbar:
            for images, labels in test_loader:
                with torch.inference_mode():
                    model_output = model(images)
                loss = torch.nn.functional.cross_entropy(model_output, labels, reduction="sum")
                test_loss += accelerator.gather_for_metrics(loss).sum().item()
                pbar.update(1)

        if accelerator.is_main_process:
            test_loss /= len(test_dataset)
            accelerator.log({"test_loss": test_loss})
            print("Epoch:", epoch, "test loss:", test_loss)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                accelerator.save_state(save_place)
                unwrap_model = accelerator.unwrap_model(model)
                torch.save(unwrap_model.state_dict(), f"{save_place}/best_model.pth")

    accelerator.end_training()


if __name__ == "__main__":
    tyro.cli(main)
