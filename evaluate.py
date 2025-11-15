# python
def evaluate_lanes(detected_centers, gt_centers, tolerance=30, debug=False):
    """
    One-to-one greedy matching between detected lane x-centers and ground-truth centers.
    Returns (precision, recall, f1) as floats in [0,1].

    Algorithm:
    - Sort detected and gt lists.
    - For each detected center, find nearest unmatched gt within tolerance.
    - Count TP as number of successful unique matches.
    - FP = len(detected) - TP
    - FN = len(gt) - TP
    """
    det = sorted([float(x) for x in detected_centers])
    gt = sorted([float(x) for x in gt_centers])

    matched_gt = [False] * len(gt)
    tp = 0

    # For each detected, scan gt for nearest unmatched within tolerance
    for d in det:
        best_idx = -1
        best_dist = tolerance + 1
        # linear scan; lists are small (<= ~30), simple and safe
        for i, g in enumerate(gt):
            if matched_gt[i]:
                continue
            dist = abs(d - g)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0:
            matched_gt[best_idx] = True
            tp += 1

    fp = len(det) - tp
    fn = len(gt) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 if (tp + fn) == 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 if (tp + fp) == 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if debug:
        print(f"DEBUG evaluate_lanes: detected={len(det)}, gt={len(gt)}, tp={tp}, fp={fp}, fn={fn}")
    return precision, recall, f1