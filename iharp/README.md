# Detecting Anomalous Gravitational Wave Signals

## Data Preparing

Please download the data with

```shell
# This might fail if more than 50 files. If so, please download manually.
gdown https://drive.google.com/drive/folders/14ylyfXXiBunScZiVbdzhYxh2bTdhfVxO --folder -O maps_data
gdown https://drive.google.com/drive/folders/1Uawcu_ocPE69Mx0nokOQI56KZ-l7pjbz --folder -O anomalies_data

# Data filtering
python nc_to_image.py --nc_folder maps_data --output_folder images
python data_filter.py --csv_folder anomalies_data --output_folder new_anomalies_data --img_folder images
```

and ensure it as follows:

```shell
project_root
|_ images
|  |_ *.png
|_ new_anomalies_data
|  |_ *.csv
|_ maps_data
|  |_ *.nc
|_ anomalies_data
|  |_ *.csv
```

## Training

To train the model, run the following command:

```shell
python train.py --image_folder images --csv_files anomalies_data
```

## Submission

Move the trained model from `best_model.pth` to `submission/model.pth`.