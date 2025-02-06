# Detecting Sea Level Rise Anomalies

## Data Preparing

Please download the data with

```shell
wget -O public_data.zip https://www.codabench.org/datasets/download/e703ab84-4444-4972-9ef7-1ebd0fc09c88/
mkdir data && unzip public_data.zip -d data && rm public_data.zip
```

and ensure it as follows:

```shell
project_root
|_ data
|  |_ background.npz
|  |_ bbh_for_challenge.npy
|  |_ sglf_for_challenge.npy
|  ...
```

## Training

To train the model, run the following command:

```shell
python train_classifier.py --data_path data --model_path best_model.pth 
```

## Submission

Move the trained model from `best_model.pth` to `submission/model.pth`.