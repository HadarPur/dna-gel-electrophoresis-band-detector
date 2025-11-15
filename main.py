from lane_detection import process_dataset
from utils import download_zip, extract_zip, extract_gold_standard

DATA_URL = "https://www.unirioja.es/cu/joheras/surveying/dataset-lane-segmentation.zip"
ZIP_FILENAME = "dataset_lane_segmentation.zip"
EXTRACT_DIR = "gel-band-data"
EXTRACT_DATASET_DIR = "dataset-lane-segmentation"
GEL_IMAGES_DIR = "gel-images"
GOLD_STANDARD_DIR = "gold-standard"
EXTRACT_GOLD_IMAGES_DIR = "gold-extracted"

if __name__ == "__main__":
    download_zip(ZIP_FILENAME, DATA_URL)
    extract_zip(ZIP_FILENAME, EXTRACT_DIR)
    extract_gold_standard(EXTRACT_DIR, EXTRACT_DATASET_DIR, GOLD_STANDARD_DIR, EXTRACT_GOLD_IMAGES_DIR)
    process_dataset(EXTRACT_DIR, EXTRACT_DATASET_DIR, GEL_IMAGES_DIR, EXTRACT_GOLD_IMAGES_DIR)
