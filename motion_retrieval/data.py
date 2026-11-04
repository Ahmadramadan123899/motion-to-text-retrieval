"""Loading the motions and their text descriptions."""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Each frame has 384 values: 192 for person 1, then 192 for person 2.
# Inside one person, the first 66 values are the 3D positions of the 22 joints.
PERSON_DIM = 192
N_JOINTS = 22
POS_DIM = N_JOINTS * 3

# Most sequences are shorter than this. Longer ones are cut.
MAX_LEN = 256


def scan_dataset(motions_dir, texts_dir):
    """List every motion file that has a matching text file.

    One motion can have several descriptions (one per line). We keep all of
    them and pick one at random each time the motion is used in training.
    """
    rows = []
    for fname in sorted(os.listdir(motions_dir)):
        if not fname.endswith(".npy"):
            continue
        motion_id = fname[:-4]
        text_path = os.path.join(texts_dir, motion_id + ".txt")
        if not os.path.exists(text_path):
            continue
        with open(text_path, encoding="utf-8") as f:
            descriptions = [line.strip() for line in f if line.strip()]
        if descriptions:
            rows.append({
                "id": motion_id,
                "motion_path": os.path.join(motions_dir, fname),
                "descriptions": descriptions,
            })
    return pd.DataFrame(rows)


def compute_stats(df, max_files=2000):
    """Mean and std of each feature, for normalization.

    Computed on the training rows only, otherwise the validation set leaks
    into the preprocessing. 2000 files is plenty to get stable values and
    keeps it fast.
    """
    frames = [np.load(p) for p in df["motion_path"].iloc[:max_files]]
    all_frames = np.concatenate(frames, axis=0)
    # The small epsilon avoids dividing by zero for features that never move.
    return all_frames.mean(axis=0), all_frames.std(axis=0) + 1e-6


def pad_or_cut(motion, max_len=MAX_LEN):
    """Give every sequence the same length so they can be batched."""
    if len(motion) < max_len:
        return np.pad(motion, ((0, max_len - len(motion)), (0, 0)))
    return motion[:max_len]


def prepare_motion(motion, mean, std, max_len=MAX_LEN):
    """Same preprocessing for training, validation and the test set."""
    motion = (motion - mean) / std
    return pad_or_cut(motion, max_len).astype(np.float32)


def rotate_around_vertical(motion):
    """Rotate the whole scene around the Y (up) axis by a random angle.

    A handshake is still a handshake if you look at it from another side,
    so this is a free way to get more training data. Both people get the
    same rotation, so their relative position does not change.

    Only the joint positions are rotated. The other values stay as they are,
    which is not perfectly consistent, but it helped anyway.
    """
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    motion = motion.copy()
    for start in (0, PERSON_DIM):
        pos = motion[:, start:start + POS_DIM].reshape(-1, N_JOINTS, 3)
        motion[:, start:start + POS_DIM] = (pos @ rot).reshape(-1, POS_DIM)
    return motion


def swap_people(motion):
    """Put person 2 first and person 1 second.

    The descriptions talk about "one person" and "the other", so the order
    of the two people should not matter to the model.
    """
    return np.concatenate([motion[:, PERSON_DIM:], motion[:, :PERSON_DIM]], axis=-1)


class MotionTextDataset(Dataset):
    """Returns (motion tensor, one of its descriptions)."""

    def __init__(self, df, mean, std, augment=False):
        self.df = df.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        motion = np.load(row["motion_path"])

        if self.augment:
            if random.random() < 0.5:
                motion = rotate_around_vertical(motion)
            if random.random() < 0.5:
                motion = swap_people(motion)

        motion = prepare_motion(motion, self.mean, self.std)
        text = random.choice(row["descriptions"])
        return torch.from_numpy(motion), text
