"""Write the Kaggle submission file.

The test set is a folder of numbered sub-folders. Each one has a query motion
(query.npy, or motion.npy in some folders) and a texts.txt file with the
candidate descriptions, one per line, as "id#text".

    python predict.py --checkpoint checkpoints/model.pt --out submission.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from motion_retrieval.data import prepare_motion
from motion_retrieval.model import RetrievalModel
from motion_retrieval.utils import KAGGLE_DATA, data_subdir, get_device

TOP_K = 10


def read_candidates(path):
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "#" in line:
                text_id, text = line.strip().split("#", 1)
                ids.append(int(text_id))
                texts.append(text)
    return ids, texts


def find_query(folder):
    for name in ("query.npy", "motion.npy"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


@torch.no_grad()
def rank_folder(model, folder, mean, std, device):
    motion_path = find_query(folder)
    text_path = os.path.join(folder, "texts.txt")
    if motion_path is None or not os.path.exists(text_path):
        print(f"skipping {folder}: missing query or texts")
        return None

    ids, texts = read_candidates(text_path)
    if not texts:
        return None

    motion = prepare_motion(np.load(motion_path), mean, std)
    motion = torch.from_numpy(motion).unsqueeze(0).to(device)
    scores = model(motion, texts).squeeze(0)

    order = torch.argsort(scores, descending=True).cpu().numpy()
    return [ids[i] for i in order[:TOP_K]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/model.pt")
    p.add_argument("--test-dir", default=None, help="defaults to <data-dir>/test")
    p.add_argument("--data-dir", default=KAGGLE_DATA)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = RetrievalModel(ckpt["text_model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    mean, std = ckpt["mean"].numpy(), ckpt["std"].numpy()

    test_dir = args.test_dir or data_subdir(args.data_dir, "test")
    folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
    folders.sort(key=lambda f: int(f) if f.isdigit() else float("inf"))

    rows = []
    for folder in tqdm(folders, desc="ranking"):
        top = rank_folder(model, os.path.join(test_dir, folder), mean, std, device)
        if top is not None:
            rows.append([folder] + top)

    columns = ["query_id"] + [f"candidate_{i}" for i in range(1, TOP_K + 1)]
    pd.DataFrame(rows, columns=columns).to_csv(args.out, index=False)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
