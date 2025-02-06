import argparse
import os

import numpy as np
import torch
from dataset import TorchADDataset
from model import ADClassifier
from sklearn.model_selection import train_test_split


def main(args):
    background = np.load(os.path.join(args.data_path, "background.npz"))["data"]
    stds = np.std(background, axis=-1)[:, :, np.newaxis]
    background = background / stds
    background = np.swapaxes(background, 1, 2)

    sglf = np.load(os.path.join(args.data_path, "sglf_for_challenge.npy"))
    stds = np.std(sglf, axis=-1)[:, :, np.newaxis]
    sglf = sglf / stds
    sglf = np.swapaxes(sglf, 1, 2)

    bbh = np.load(os.path.join(args.data_path, "bbh_for_challenge.npy"))
    stds = np.std(bbh, axis=-1)[:, :, np.newaxis]
    bbh = bbh / stds
    bbh = np.swapaxes(bbh, 1, 2)

    signal = np.concatenate((sglf, bbh), axis=0)

    anomaly_class = {"background": 0, "signal": 1}

    background_ids = np.full(background.shape[0], anomaly_class["background"], dtype=int)
    signal_ids = np.full(signal.shape[0], anomaly_class["signal"], dtype=int)

    x = np.concatenate((background, signal), axis=0).reshape((-1, 200, 2))
    y = np.concatenate((background_ids, signal_ids), axis=0)

    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]

    y = y.reshape((-1, 1))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42)

    print(f"x train/test shapes: {x_train.shape} {x_val.shape}")
    print(f"y train/test shapes: {y_train.shape} {y_val.shape}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # Create data loaders for our datasets; shuffle for training, not for validation
    training_loader = torch.utils.data.DataLoader(
        TorchADDataset(x_train, y_train, device, add_noise=True), batch_size=1280, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        TorchADDataset(x_val, y_val, device), batch_size=1280, shuffle=False
    )

    model = ADClassifier().to(device)
    print(model)
    print(
        f"The number of trainable parameters of the model is {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def train_one_epoch():
        running_loss = 0.0
        last_loss = 0.0
        for i, data in enumerate(training_loader):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 10  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                running_loss = 0.0

        return last_loss

    epoch_number = 0

    EPOCHS = 100

    best_vloss = float("inf")
    train_losses = []
    val_losses = []
    for epoch in range(EPOCHS):
        print("EPOCH {}:".format(epoch_number + 1))

        model.train(True)
        avg_loss = train_one_epoch()

        running_vloss = 0.0
        model.eval()

        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))
        train_losses.append(avg_loss)
        val_losses.append(avg_vloss.item())

        if avg_vloss.item() < best_vloss:
            best_vloss = avg_vloss.item()
            torch.save(model.state_dict(), args.model_path)

        epoch_number += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the input dataset")
    parser.add_argument("--model_path", type=str, help="Where to save the model")
    args = parser.parse_args()
    main(args)
