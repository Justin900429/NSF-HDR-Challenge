# Butterfly Hybrid Detection

## Data preparing

Please download the data with

```shell
python download_data.py --csv_file https://raw.githubusercontent.com/Imageomics/HDR-anomaly-challenge/refs/heads/main/files/butterfly_anomaly_train.csv
```

and ensure it as follows:

```shell
project_root
|_ data
|  |_ images
|  |  |_ hybrid
|  |  |_ non-hybrid
|  |  |_ ...
|  |_ butterfly_anomaly_train.csv
|  ...
```

## Training

To train the model, run the following command:

```shell
# add --multi_gpu --num_processes {} if using multi-gpu
accelerate launch train.py --root data/images --csv_file data/butterfly_anomaly_train.csv --accum_steps 8 --batch_size 8 --epochs 100
```

## Submission

Move the trained model from `checkpoints/model.pth` to `submission/model.pth`.