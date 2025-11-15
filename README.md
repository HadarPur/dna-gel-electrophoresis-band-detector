# dna-gel-electrophoresis-band-detector

Small toolkit to detect lanes/bands in gel electrophoresis images and evaluate detections against ROI annotations.

## Features
- Load ImageJ `.roi` files and `.zip` collections of ROIs.
- Convert rectangular and polygon ROIs to bounding boxes for lane detection.
- One\-to\-one greedy matching evaluation with precision / recall / F1 computation.
- Debugging helpers to inspect ROIs and matching results.

## Requirements
- Python 3.8+
- See `requirements.txt` for exact dependencies.

## Installation
- Clone the repository:
  ```
  git clone https://github.com/HadarPur/dna-gel-electrophoresis-band-detector.git
  ```
- Install requirements:
  ```
  pip install -r requirements.txt
  ```

## Quick usage
Run detection and evaluation:
  ```
  python main.py
  ```
  
## Dataset
- Dataset reference: https://www.unirioja.es/cu/joheras/surveying/

## ROI handling notes
- ROI files may contain rectangle fields (`left`, `top`, `width`, `height`) or polygon point lists (`x`, `y`).
- The code computes a bounding box for polygons so both shapes are handled uniformly.
- Some ROI collections contain short bands or non\-lane ROIs; the loader filters by bounding box height and aspect ratio.
