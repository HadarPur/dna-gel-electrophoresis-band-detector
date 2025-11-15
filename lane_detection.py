import csv
import os

import cv2
import numpy as np
from read_roi import read_roi_file
from scipy.signal import find_peaks

from evaluate import evaluate_lanes
from utils import visualize_save

LANE_WIDTH_EST = 20  # width in pixels to extract each lane
BAND_PROMINENCE = 0.03
BAND_DISTANCE = 5


def preprocess(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return blurred


def detect_lanes(img_gray, min_prominence=0.03):
    proj = np.mean(img_gray, axis=0)
    proj_inv = proj.max() - proj
    p = (proj_inv - proj_inv.min()) / (np.ptp(proj_inv) + 1e-8)
    distance = max(10, img_gray.shape[1] // 50)
    peaks, props = find_peaks(p, prominence=min_prominence, distance=distance)
    return peaks, p


def extract_lane_strip(img_gray, center_x, width=LANE_WIDTH_EST):
    h, w = img_gray.shape
    x1 = max(0, center_x - width // 2)
    x2 = min(w, center_x + width // 2)
    return img_gray[:, x1:x2]


def detect_bands_in_lane(lane_strip, prominence=BAND_PROMINENCE, distance=BAND_DISTANCE, invert_dark_bands=True):
    profile = np.mean(lane_strip, axis=1)
    if invert_dark_bands:
        prof = profile.max() - profile
    else:
        prof = profile
    prof_norm = (prof - prof.min()) / (np.ptp(prof) + 1e-8)
    peaks, props = find_peaks(prof_norm, prominence=prominence, distance=distance)
    return peaks, prof_norm


def process_dataset(extract_dir, extract_dataset_dir, gel_images_dir, extract_gold_images_dir):
    gels_dir = os.path.join(f"./{extract_dir}", f"{extract_dataset_dir}/{gel_images_dir}")
    output_vis_dir = os.path.join(extract_dir, "results_vis")
    output_csv = os.path.join(extract_dir, "results.csv")
    os.makedirs(output_vis_dir, exist_ok=True)

    if not os.path.exists(gels_dir):
        print(f"❌ Gel images directory not found: {gels_dir}")
        print("Make sure the dataset was extracted correctly.")
        return

    lane_prec_list, lane_rec_list, lane_f1_list = [], [], []
    gold_annotations = load_lane_annotations(f"{extract_dir}/{extract_gold_images_dir}")

    with open(output_csv, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["image", "lane_index", "band_y_position"])

        for fname in os.listdir(gels_dir):
            if not fname.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(gels_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                print("Failed to load:", img_path)
                continue

            img_proc = preprocess(img)
            lane_centers, proj = detect_lanes(img_proc)

            # 🧩 --- Lane evaluation ---
            base = os.path.splitext(fname)[0]
            gt_lane_centers = gold_annotations.get(base, [])

            detected_count = len(lane_centers)
            actual_count = len(gt_lane_centers)

            lane_p, lane_r, lane_f = evaluate_lanes(lane_centers, gt_lane_centers, tolerance=30)
            print(f"[{fname:<12}]  Lanes Detected: {detected_count:<2}  |  Actual: {actual_count:<2}  |  "
                  f"Precision: {lane_p:.2f}  |  Recall: {lane_r:.2f}  |  F1: {lane_f:.2f}")

            lane_prec_list.append(lane_p)
            lane_rec_list.append(lane_r)
            lane_f1_list.append(lane_f)

            lane_bands = []
            for i, cx in enumerate(lane_centers):
                strip = extract_lane_strip(img_proc, cx)
                peaks, prof = detect_bands_in_lane(strip)
                lane_bands.append(peaks)
                for py in peaks:
                    csvwriter.writerow([fname, i, py])

            vis_path = os.path.join(output_vis_dir, fname.replace('.tif', '.png'))
            visualize_save(img, lane_centers, lane_bands, vis_path, proj=proj)

    # 🧾 Summary
    if lane_prec_list:
        print("\n=== Overall Lane Detection Performance ===")
        print(f"Precision: {np.mean(lane_prec_list):.3f}")
        print(f"Recall:    {np.mean(lane_rec_list):.3f}")
        print(f"F1-score:  {np.mean(lane_f1_list):.3f}")


def load_lane_annotations(roi_dir):
    """
    Parses ROI lane regions and returns dict[image_name] = list of x-centers.
    Filters out band ROIs (small height) and keeps lane ROIs (large height).
    """
    annotations = {}
    for subdir in sorted(os.listdir(roi_dir)):
        subpath = os.path.join(roi_dir, subdir)
        if not os.path.isdir(subpath):
            continue

        lanes = []
        roi_files = [f for f in os.listdir(subpath) if f.endswith(".roi")]
        if not roi_files:
            print(f"⚠️ No ROI files found in {subdir}")
            continue

        # Debug: track what we're filtering
        all_rois = []

        for fname in roi_files:
            roi_path = os.path.join(subpath, fname)
            try:
                roi = read_roi_file(roi_path)
                for roi_name, data in roi.items():
                    # Ensure it's a rectangle
                    if all(k in data for k in ("left", "width", "top", "height")):
                        height = data["height"]
                        width = data["width"]
                        all_rois.append((roi_name, width, height))

                        # Lane ROIs should be tall and narrow
                        # Adjust thresholds based on your actual data
                        if height > width and height > 40:  # Changed logic
                            cx = data["left"] + data["width"] / 2
                            lanes.append(cx)
            except Exception as e:
                print(f"❌ Error reading ROI {roi_path}: {e}")

        # Debug output for images with 0 lanes
        if len(lanes) == 0 and len(all_rois) > 0:
            print(f"⚠️ {subdir}: Found {len(all_rois)} ROIs but 0 lanes. Sample ROIs:")
            for name, w, h in all_rois[:3]:
                print(f"    - {name}: width={w}, height={h}")

        annotations[subdir] = lanes
        print(f"[INFO] {subdir}: {len(lanes)} valid lane(s) loaded from {len(roi_files)} ROI files")

    print(f"✅ Loaded annotations for {len(annotations)} gels\n")
    return annotations
