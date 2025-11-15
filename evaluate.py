def evaluate_lanes(detected, ground_truth, tolerance=5):
    TP = sum(any(abs(g - d) <= tolerance for d in detected) for g in ground_truth)
    FP = len(detected) - TP
    FN = len(ground_truth) - TP

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0

    # Clamp to [0, 1]
    precision = min(max(precision, 0), 1)
    recall = min(max(recall, 0), 1)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1
