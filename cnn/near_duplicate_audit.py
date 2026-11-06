import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
LABELS = ["Cat", "Dog", "Moris", "Motya", "Biatrix"]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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


def find(parent, index):
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def audit(train, radius=4):
    if not 0 <= radius <= 4:
        raise ValueError("dHash radius must be between 0 and 4")
    exact_hashes, image_hashes = [], []
    exact_groups = {}
    for index, image_id in enumerate(train.image_id):
        path = ROOT / "data" / "train" / f"{image_id}.jpg"
        code = sha256(path)
        exact_hashes.append(code)
        image_hashes.append(dhash(path))
        exact_groups.setdefault(code, []).append(index)

    pairs = {}
    for indices in exact_groups.values():
        for i, j in combinations(indices, 2):
            pairs[i, j] = 0

    tree = BKTree()
    for index, value in enumerate(image_hashes):
        for distance, match in tree.query(value, radius):
            pairs[match, index] = distance
        tree.add(value, index)

    parent = list(range(len(train)))
    for i, j in pairs:
        parent[find(parent, j)] = find(parent, i)
    roots = sorted({find(parent, i) for i, j in pairs})
    group_ids = {root: number for number, root in enumerate(roots, 1)}
    labels = train[LABELS].to_numpy()
    label_names = ["|".join(name for name in LABELS if row[name]) or "Background"
                   for _, row in train.iterrows()]
    rows = []
    for (i, j), distance in sorted(pairs.items()):
        a, b = train.iloc[i], train.iloc[j]
        rows.append({
            "group_id": group_ids[find(parent, i)],
            "image_id_a": a.image_id, "image_id_b": b.image_id,
            "labels_a": label_names[i], "labels_b": label_names[j],
            "fold_a": int(a.fold), "fold_b": int(b.fold),
            "sha256_equal": exact_hashes[i] == exact_hashes[j],
            "hash_distance": distance,
            "cross_fold": a.fold != b.fold,
            "label_conflict": bool((labels[i] != labels[j]).any()),
        })
    columns = ["group_id", "image_id_a", "image_id_b", "labels_a", "labels_b",
               "fold_a", "fold_b", "sha256_equal", "hash_distance",
               "cross_fold", "label_conflict"]
    result = pd.DataFrame(rows, columns=columns).astype({
        "sha256_equal": bool, "cross_fold": bool, "label_conflict": bool,
    })
    for column in ["cross_fold", "label_conflict"]:
        result[f"group_{column}"] = result.groupby("group_id")[column].transform("any")
    result["needs_review"] = ~result.sha256_equal
    return result.sort_values(["group_id", "hash_distance", "image_id_a", "image_id_b"])


def show_group(result, group_id):
    pairs = result[result.group_id == group_id]
    if pairs.empty:
        raise ValueError(f"Group {group_id} not found")
    print(pairs.to_string(index=False))
    images = {}
    for row in pairs.itertuples(index=False):
        images[row.image_id_a] = (row.labels_a, row.fold_a)
        images[row.image_id_b] = (row.labels_b, row.fold_b)
    columns = min(3, len(images))
    height = 300 * ((len(images) + columns - 1) // columns)
    sheet = Image.new("RGB", (300 * columns, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image_id, (labels, fold)) in enumerate(images.items()):
        x, y = (index % columns) * 300, (index // columns) * 300
        with Image.open(ROOT / "data" / "train" / f"{image_id}.jpg") as image:
            image = image.convert("RGB")
            image.thumbnail((290, 240))
            sheet.paste(image, (x, y))
        draw.text((x, y + 242), f"{image_id}\n{labels}; fold {fold}", fill="black")
    sheet.show(title=f"Duplicate candidates: group {group_id}")


def main():
    parser = argparse.ArgumentParser(description="Find SHA-256 duplicates and dHash candidates")
    parser.add_argument("--radius", type=int, choices=range(5), default=4)
    parser.add_argument("--show-group", type=int)
    args = parser.parse_args()
    output = ROOT / "outputs" / "train_duplicate_audit.csv"
    if args.show_group is not None:
        show_group(pd.read_csv(output), args.show_group)
        return

    train = pd.read_csv(ROOT / "data" / "train.csv")
    folds = pd.read_csv(ROOT / "folds.csv")
    train = train.merge(folds, on="image_id", how="outer", validate="one_to_one", indicator=True)
    if not train["_merge"].eq("both").all() or train.fold.isna().any():
        raise ValueError("Every train image must have exactly one current fold")
    result = audit(train, args.radius)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    exact = result[result.sha256_equal]
    candidates = result[result.needs_review]
    groups = result.drop_duplicates("group_id")
    print(f"Scanned {len(train)} train images; saved {output}")
    print(f"SHA-256: {len(exact)} exact pairs, "
          f"{int(exact.cross_fold.sum())} cross-fold pairs")
    print(f"dHash <= {args.radius}: {len(result)} unique pairs (including SHA-256); "
          f"{len(candidates)} non-exact candidates need visual review")
    print(f"Connected groups: {len(groups)}; across folds: {int(groups.group_cross_fold.sum())}")
    print(f"Cross-fold pairs: {int(result.cross_fold.sum())}; "
          f"within-fold pairs: {int((~result.cross_fold).sum())}")
    print(f"Label conflicts: {int(result.label_conflict.sum())} pairs, "
          f"{int(groups.group_label_conflict.sum())} groups "
          f"({int(exact.label_conflict.sum())} exact pairs)")
    print("Manual review: CSV rows with needs_review=True; start with distance 4 or label conflicts")
    print("Preview: python cnn/near_duplicate_audit.py --show-group ID")


if __name__ == "__main__":
    main()
