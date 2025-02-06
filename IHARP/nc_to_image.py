import argparse
import glob
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xarray
from tqdm import tqdm


def to_numpy_image(data_path):
    dataset = xarray.open_dataset(data_path)
    sla = dataset.variables["sla"][:]
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
        shape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
    )
    buf.close()
    return img_arr


def main(nc_folder, output_folder):
    nc_files = glob.glob(os.path.join(nc_folder, "*.nc"))
    for nc_file in tqdm(nc_files):
        img_name = os.path.basename(nc_file).replace(".nc", ".png")
        img = to_numpy_image(nc_file)
        cv2.imwrite(os.path.join(output_folder, img_name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.nc_folder, args.output_folder)
