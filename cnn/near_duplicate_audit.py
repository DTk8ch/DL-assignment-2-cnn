"""Find image-level near duplicates across train folds and in the test set.

Uses a 64-bit difference hash to propose candidate matches. Hash distance is
only a screen; inspect proposed pairs visually before using their labels.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent


def dhash(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = pixels[:, 1:] > pixels[:, :-1]
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


class BKTree:
    def __init__(self):
        self.root = None

    def add(self, value, index):
        if self.root is None:
            self.root = (value, [index], {})
            return
        node = self.root
        while True:
            code, indices, children = node
            distance = (value ^ code).bit_count()
            if distance == 0:
                indices.append(index)
                return
            if distance not in children:
                children[distance] = (value, [index], {})
                return
            node = children[distance]

    def query(self, value, radius):
        if self.root is None:
            return []
        matches = []
        stack = [self.root]
        while stack:
            code, indices, children = stack.pop()
            distance = (value ^ code).bit_count()
            if distance <= radius:
                matches.extend((distance, index) for index in indices)
            stack.extend(child for edge, child in children.items()
                         if distance - radius <= edge <= distance + radius)
        return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=4)
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "data" / "train.csv").merge(
        pd.read_csv(ROOT / "folds.csv"), on="image_id", validate="one_to_one")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    train_hash = [dhash(ROOT / "data" / "train" / f"{image_id}.jpg")
                  for image_id in train.image_id]
    test_hash = [dhash(ROOT / "data" / "test" / f"{image_id}.jpg")
                 for image_id in test.image_id]
    tree = BKTree()
    for index, value in enumerate(train_hash):
        tree.add(value, index)

    rows = []
    for subset, frame, hashes in [("train", train, train_hash), ("test", test, test_hash)]:
        for index, value in enumerate(hashes):
            for distance, match in tree.query(value, args.radius):
                if subset == "train" and (index == match or
                                           frame.iloc[index].fold == train.iloc[match].fold):
                    continue
                rows.append({"subset": subset, "image_id": frame.iloc[index].image_id,
                             "source_image_id": train.iloc[match].image_id,
                             "hash_distance": distance,
                             "target_fold": (int(frame.iloc[index].fold)
                                             if subset == "train" else np.nan),
                             "source_fold": int(train.iloc[match].fold)})
    output = ROOT / "outputs" / "near_duplicates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows).sort_values(["subset", "hash_distance"])
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} candidate pairs to {output}")
    print(result.groupby(["subset", "hash_distance"]).size().to_string())


if __name__ == "__main__":
    main()
