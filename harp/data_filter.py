import argparse
import glob
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_folder", type=str, required=True)
    parser.add_argument("--img_folder", type=str, required=True)
    parser.add_argument("--new_csv_folder", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.new_save_path, exist_ok=True)
    file_paths = sorted(list(glob.glob(os.path.join(args.csv_folder, "*.csv"))))

    csv_data = [pd.read_csv(file) for file in file_paths]

    common_dates = set(csv_data[0]["t"])
    for data in csv_data[1:]:
        common_dates &= set(data["t"])

    image_files = set(
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(args.img_folder, "*.png"))
    )
    common_dates &= image_files

    for i, data in enumerate(csv_data):
        basename = os.path.basename(file_paths[i])
        filtered_data = data[data["t"].isin(common_dates)]
        filtered_data.to_csv(os.path.join(args.new_save_path, basename), index=False)
