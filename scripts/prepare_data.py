"""
Prepare the 100k JetClass subset for the JEPA vs MAE comparison demo.

Reads one ROOT file per class from the val_5M directory, takes the first 10k events
per class (100k total, balanced), and splits into train / val / test sets.

Split per class:
  - 8,000 train
  - 1,000 val
  - 1,000 test

Output: numpy .npy files in data/{train,val,test}/
  - particles.npy  : (N, 4, 128) float32
  - labels.npy     : (N, 10)     int32

Usage:
    python scripts/prepare_data.py --data-dir /path/to/val_5M --output-dir ./data --seed 42
"""

import os
import argparse
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data.dataloader import read_file


# Maps class index → source ROOT filename (one file per class from val_5M)
CLASS_FILES = {
    0: "ZJetsToNuNu_120.root",   # label_QCD  (background, q/g jets)
    1: "HToBB_120.root",          # label_Hbb
    2: "HToCC_120.root",          # label_Hcc
    3: "HToGG_120.root",          # label_Hgg
    4: "HToWW4Q_120.root",        # label_H4q
    5: "HToWW2Q1L_120.root",      # label_Hqql
    6: "ZToQQ_120.root",          # label_Zqq
    7: "WToQQ_120.root",          # label_Wqq
    8: "TTBar_120.root",          # label_Tbqq
    9: "TTBarLep_120.root",       # label_Tbl
}

EVENTS_PER_CLASS = 10_000
TRAIN_PER_CLASS = 8_000
VAL_PER_CLASS = 1_000
TEST_PER_CLASS = 1_000


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare 100k JetClass subset")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to val_5M directory containing ROOT files")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Output directory for processed numpy files")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits")
    parser.add_argument("--max-particles", type=int, default=128,
                        help="Maximum number of particles per jet")
    return parser.parse_args()


def load_class(filepath: str, n_events: int, max_particles: int):
    """Load first n_events from a ROOT file, return (particles, label_onehot)."""
    print(f"  Reading {os.path.basename(filepath)} ...", flush=True)
    x_particles, _, y = read_file(filepath, max_num_particles=max_particles)
    # x_particles: (N, 4, max_particles)
    # y: (N, 10) one-hot
    x_particles = x_particles[:n_events].astype(np.float32)
    y = y[:n_events].astype(np.int32)
    print(f"    Loaded {len(x_particles)} events, shape {x_particles.shape}")
    return x_particles, y


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    os.makedirs(os.path.join(args.output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)

    all_train_x, all_train_y = [], []
    all_val_x, all_val_y = [], []
    all_test_x, all_test_y = [], []

    print(f"\nLoading {EVENTS_PER_CLASS} events per class from {args.data_dir}")
    print(f"Split: {TRAIN_PER_CLASS} train / {VAL_PER_CLASS} val / {TEST_PER_CLASS} test\n")

    for class_idx, filename in CLASS_FILES.items():
        filepath = os.path.join(args.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Expected file not found: {filepath}")

        x, y = load_class(filepath, EVENTS_PER_CLASS, args.max_particles)

        # Shuffle within class for random split
        perm = rng.permutation(len(x))
        x, y = x[perm], y[perm]

        all_train_x.append(x[:TRAIN_PER_CLASS])
        all_train_y.append(y[:TRAIN_PER_CLASS])

        all_val_x.append(x[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS])
        all_val_y.append(y[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS])

        all_test_x.append(x[TRAIN_PER_CLASS + VAL_PER_CLASS:])
        all_test_y.append(y[TRAIN_PER_CLASS + VAL_PER_CLASS:])

    # Concatenate and shuffle each split globally
    for split_name, x_list, y_list in [
        ("train", all_train_x, all_train_y),
        ("val", all_val_x, all_val_y),
        ("test", all_test_x, all_test_y),
    ]:
        X = np.concatenate(x_list, axis=0)
        Y = np.concatenate(y_list, axis=0)
        perm = rng.permutation(len(X))
        X, Y = X[perm], Y[perm]

        out_dir = os.path.join(args.output_dir, split_name)
        np.save(os.path.join(out_dir, "particles.npy"), X)
        np.save(os.path.join(out_dir, "labels.npy"), Y)

        print(f"\n{split_name:5s}: {len(X):6d} jets  "
              f"| particles {X.shape}  | labels {Y.shape}")
        print(f"       Saved to {out_dir}/")

    print("\nDone. Class distribution (train set):")
    Y_train = np.concatenate(all_train_y, axis=0)
    class_names = [
        "QCD/ZNuNu", "H→bb", "H→cc", "H→gg", "H→4q",
        "H→ℓνqq", "Z→qq", "W→qq", "t→bqq", "t→bℓν"
    ]
    for i, name in enumerate(class_names):
        count = Y_train[:, i].sum()
        print(f"  Class {i:2d} ({name:10s}): {count:5d}")


if __name__ == "__main__":
    main()
