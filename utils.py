import os
from zipfile import ZipFile

import cv2
import requests


# --- Download and Extract ---
def download_zip(zip_filename, data_url):
    try:
        response = requests.get(data_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded '{zip_filename}' from '{data_url}'")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")


def extract_zip(zip_filepath, extract_to_path):
    """Extracts the contents of a ZIP file to a specified directory."""
    try:
        with ZipFile(zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to_path)
        print(f"Successfully extracted '{zip_filepath}' to '{extract_to_path}'")
    except FileNotFoundError:
        print(f"Error: ZIP file '{zip_filepath}' not found.")
    except Exception as e:
        print(f"Error extracting the ZIP file: {e}")


def extract_gold_standard(extract_dir, dataset_dir, gold_standard_dir, gold_extract_dir):
    gold_dir = os.path.join(f"./{extract_dir}", f"{dataset_dir}/{gold_standard_dir}")
    gold_extract_dir = os.path.join(extract_dir, gold_extract_dir)
    os.makedirs(gold_extract_dir, exist_ok=True)

    if not os.path.exists(gold_dir):
        print(f"[WARN] Gold-standard folder not found: {gold_dir}")
        return None

    for fname in os.listdir(gold_dir):
        if not fname.endswith(".zip"):
            continue
        zip_path = os.path.join(gold_dir, fname)
        subfolder = os.path.splitext(fname)[0]
        target_dir = os.path.join(gold_extract_dir, subfolder)
        os.makedirs(target_dir, exist_ok=True)

        with ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_dir)
        print(f"[INFO] Extracted {fname} -> {target_dir}")

    print(f"✅ All gold-standard ROIs extracted to: {gold_extract_dir}\n")
    return gold_extract_dir


def visualize_save(img, lane_centers, lane_bands, save_path, proj=None):
    vis = img.copy()
    h, w = img.shape[:2]
    for cx in lane_centers:
        cv2.line(vis, (int(cx), 0), (int(cx), h - 1), (0, 255, 0), 1)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]
    for i, (cx, bands) in enumerate(zip(lane_centers, lane_bands)):
        c = colors[i % len(colors)]
        for y in bands:
            cv2.line(vis, (int(cx) - 40, int(y)), (int(cx) + 40, int(y)), c, 1)

    # save image
    cv2.imwrite(save_path, vis)

    # plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    # plt.show()
