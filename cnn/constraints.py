"""Optional hard label rules for image-level predictions.

Some training images contain multiple animals and violate these rules. Use
them only after comparing constrained and unconstrained validation scores.
"""

import numpy as np


def apply_label_constraints(probabilities, thresholds):
    """Threshold probabilities, then apply the requested exclusion rules.

    Label order: Cat, Dog, Moris, Motya, Biatrix. For simultaneous Moris and
    Motya predictions, retain the class farther above its own threshold in
    log-odds space. This makes the comparison fair with different thresholds.
    """
    probs = np.asarray(probabilities, dtype=float)
    limits = np.asarray(thresholds, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 5:
        raise ValueError("probabilities must have shape (n_images, 5)")
    if limits.shape != (5,) or np.any((limits <= 0) | (limits >= 1)):
        raise ValueError("thresholds must contain five values strictly between 0 and 1")

    result = (probs >= limits).astype(np.int64)
    result[result[:, 0] == 1, 4] = 0  # Cat excludes Biatrix.
    result[result[:, 1] == 1, 2:4] = 0  # Dog excludes Moris and Motya.

    both = np.flatnonzero((result[:, 2] == 1) & (result[:, 3] == 1))
    if len(both):
        clipped = np.clip(probs[both, 2:4], 1e-7, 1 - 1e-7)
        log_odds = np.log(clipped / (1 - clipped))
        threshold_odds = np.log(limits[2:4] / (1 - limits[2:4]))
        winner = (log_odds - threshold_odds).argmax(axis=1)
        result[both, 3 - winner] = 0
    return result
