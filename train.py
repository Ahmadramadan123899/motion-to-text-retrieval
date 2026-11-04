"""Train the motion and text encoders.

Two ways to run it:

    # 1. Find the best number of epochs, with an honest validation score.
    python train.py --data-dir /path/to/data

    # 2. Final model for the submission: same settings, but trained on all
    #    the labelled data (no validation left, so no early stopping).
    #    Use the best epoch found in step 1.
    python train.py --data-dir /path/to/data --full --epochs 19
"""

import argparse
import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from motion_retrieval.data import MotionTextDataset, compute_stats, scan_dataset
from motion_retrieval.metrics import ranks_from_similarity, summary
from motion_retrieval.model import RetrievalModel, contrastive_loss
from motion_retrieval.utils import KAGGLE_DATA, data_subdir, get_device, set_seed

# The test set gives 32 candidate texts per query motion, so we evaluate the
# same way: split the validation set into groups of 32 and rank inside each.
GROUP_SIZE = 32


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=KAGGLE_DATA)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--full", action="store_true", help="train on all the data, no validation")
    p.add_argument("--text-model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--limit", type=int, default=None, help="use only N motions (quick tests)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, df_val, mean, std, device):
    """Competition-style score on the validation set."""
    model.eval()
    # No augmentation, and always the first description, so the score
    # is the same every time we run it.
    dataset = MotionTextDataset(df_val, mean, std, augment=False)
    # The few motions left after the last full group are skipped, so that
    # every query has the same number of candidates as on the test set.
    group = min(GROUP_SIZE, len(dataset))
    all_ranks = []
    for start in range(0, len(dataset) - group + 1, group):
        idx = range(start, start + group)
        motions = torch.stack([dataset[i][0] for i in idx]).to(device)
        texts = [df_val.iloc[i]["descriptions"][0] for i in idx]
        sim = model(motions, texts).cpu().numpy()
        all_ranks.extend(ranks_from_similarity(sim))
    return summary(all_ranks)


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    losses = []
    bar = tqdm(loader, desc=f"epoch {epoch}")
    for motions, texts in bar:
        optimizer.zero_grad()
        loss = contrastive_loss(model(motions.to(device), texts))
        loss.backward()
        # The recurrent layers can get big gradients at the start; clipping
        # keeps the first epochs from blowing up.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        bar.set_postfix(loss=f"{np.mean(losses[-20:]):.3f}",
                        temp=f"{model.logit_scale.exp().item():.1f}")
    return float(np.mean(losses))


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)

    df = scan_dataset(data_subdir(args.data_dir, "motions"), data_subdir(args.data_dir, "texts"))
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed)
    print(f"{len(df)} motions with descriptions")

    if args.full:
        df_train, df_val = df, None
    else:
        df_train, df_val = train_test_split(df, test_size=args.val_size, random_state=args.seed)
        print(f"train: {len(df_train)}  validation: {len(df_val)}")

    # Normalization stats come from the training rows only.
    mean, std = compute_stats(df_train)

    loader = DataLoader(
        MotionTextDataset(df_train, mean, std, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        # A last small batch would have very few negatives and a noisy loss.
        drop_last=True,
        num_workers=args.num_workers,
    )

    model = RetrievalModel(args.text_model).to(device)
    # AdamW + strong weight decay: the model overfits fast on 6k motions.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    ckpt_path = os.path.join(args.out_dir, "model.pt")
    best_score, best_epoch, bad_epochs = -1.0, 0, 0
    history = []

    def save():
        torch.save({
            "model": model.state_dict(),
            # Saved as tensors so torch.load works with weights_only=True.
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "text_model": args.text_model,
        }, ckpt_path)

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device, epoch)
        scheduler.step()

        if args.full:
            history.append({"epoch": epoch, "loss": loss})
            continue

        metrics = evaluate(model, df_val, mean, std, device)
        history.append({"epoch": epoch, "loss": loss, **metrics})
        print(f"  val R@1 {metrics['R@1']:.3f}  R@5 {metrics['R@5']:.3f}  score {metrics['score']:.4f}")

        if metrics["score"] > best_score:
            best_score, best_epoch, bad_epochs = metrics["score"], epoch, 0
            save()
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("no improvement for a while, stopping")
                break

    if args.full:
        save()
        print(f"saved final model to {ckpt_path}")
    else:
        print(f"best validation score {best_score:.4f} at epoch {best_epoch}, saved to {ckpt_path}")

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
