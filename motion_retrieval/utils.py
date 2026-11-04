import os
import random

import numpy as np
import torch

KAGGLE_DATA = "/kaggle/input/datasets/valentingauthi/datasetmotiontotext"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_subdir(data_dir, name):
    """The Kaggle upload has folders like motions/motions, a local copy may not."""
    nested = os.path.join(data_dir, name, name)
    return nested if os.path.isdir(nested) else os.path.join(data_dir, name)
