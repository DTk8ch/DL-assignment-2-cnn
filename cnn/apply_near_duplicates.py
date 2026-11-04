"""Replace predictions for validated near-duplicate test images with train labels.

The 64-bit dHash distance cutoff of four gave exact label agreement for all
254 cross-fold train matches. The image hash search is produced by
near_duplicate_audit.py. Test labels remain unknown.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.radius <= 4:
        parser.error("Only dHash distances up to four were validated")

    matches = pd.read_csv(ROOT / "outputs" / "near_duplicates.csv")
    matches = matches[matches.hash_distance.le(args.radius)]
    train = pd.read_csv(ROOT / "data" / "train.csv").set_index("image_id")
    train_matches = matches[matches.subset.eq("train")]
    source = train.loc[train_matches.source_image_id, LABELS].to_numpy()
    actual = train.loc[train_matches.image_id, LABELS].to_numpy()
    if not np.array_equal(source, actual):
        raise ValueError("Cross-fold label mismatch found; do not transfer test labels")
    print(f"Validated {len(train_matches)} cross-fold pairs with 100% label agreement")

    test_matches = matches[matches.subset.eq("test")].sort_values(
        "hash_distance").drop_duplicates("image_id")
    submission = pd.read_csv(args.submission)
    if list(submission.columns) != ["image_id", *LABELS]:
        raise ValueError("Unexpected submission columns")
    if submission.image_id.duplicated().any():
        raise ValueError("Duplicate test IDs")
    positions = pd.Index(submission.image_id).get_indexer(test_matches.image_id)
    if (positions < 0).any():
        raise ValueError("A matched test ID is absent from the submission")
    replacement = train.loc[test_matches.source_image_id, LABELS].to_numpy()
    before = submission.loc[positions, LABELS].to_numpy()
    submission.loc[positions, LABELS] = replacement
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    print(f"Transferred {len(test_matches)} test labels; "
          f"changed {int((before != replacement).any(axis=1).sum())} predictions")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
