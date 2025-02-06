import argparse
import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image


def calculate_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_file(download_folder, file_url, hybrid_stat, expected_md5, filename):
    hybrid_folder = os.path.join(download_folder, hybrid_stat)
    os.makedirs(hybrid_folder, exist_ok=True)

    file_path = os.path.join(hybrid_folder, filename)

    if os.path.exists(file_path):
        try:
            Image.open(file_path).load()
            return
        except Exception as e:
            print(str(e))

    try:
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        # Validate MD5
        if calculate_md5(file_path) != expected_md5:
            os.remove(file_path)
            print(f"MD5 mismatch for {filename}, file removed.")

    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)


def main(csv_file, download_folder, max_workers):
    tasks = []

    with open(csv_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row in reader:
                file_url = row["file_url"]
                hybrid_stat = row["hybrid_stat"]
                md5 = row["md5"]
                filename = row["filename"]

                if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                tasks.append(
                    executor.submit(
                        download_file, download_folder, file_url, hybrid_stat, md5, filename
                    )
                )

            for future in as_completed(tasks):
                future.result()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--download_folder", type=str, default="data")
    parser.add_argument("--max_workers", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.download_folder, exist_ok=True)

    if args.csv_file.startswith("http"):
        response = requests.get(args.csv_file)
        basename = os.path.basename(args.csv_file)
        with open(os.path.join(args.download_folder, basename), "wb") as f:
            f.write(response.content)
        args.csv_file = os.path.join(args.download_folder, basename)

    main(
        csv_file=args.csv_file,
        download_folder=args.download_folder,
        max_workers=args.max_workers,
    )
