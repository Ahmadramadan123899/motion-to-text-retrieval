"""The two encoders and the contrastive loss.

The idea is the same as CLIP (and TMR for motion): one network turns a motion
into a vector, another turns a sentence into a vector, and we train both so
that a motion and its description end up close together, and far from the
other descriptions in the batch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .data import N_JOINTS, PERSON_DIM, POS_DIM

EMBED_DIM = 256

# Joints used for the distance features. We meant to use the hands, but if the
# data follows the usual SMPL order, 15 is the head and 16 the left shoulder
# (the wrists are 20 and 21). We kept these because they are what the winning
# submission used. Testing the wrists is on the to-do list in the README.
PELVIS = 0
JOINT_A = 15
JOINT_B = 16


def time_diff(x):
    """Frame-to-frame difference, with a zero for the first frame."""
    d = torch.zeros_like(x)
    d[:, 1:] = x[:, 1:] - x[:, :-1]
    return d


def joint_distance(pos1, pos2, j1, j2):
    """Distance between joint j1 of person 1 and joint j2 of person 2, per frame."""
    return torch.norm(pos1[:, :, j1] - pos2[:, :, j2], dim=-1, keepdim=True)


class MotionEncoder(nn.Module):
    """Turns a two-person motion of shape (batch, frames, 384) into one vector.

    Before the recurrent layers we add hand-made features:
    - velocity and acceleration of every value, because many actions differ
      more by how fast people move than by their pose (a push vs a touch);
    - four distances between the two people, because the descriptions are
      all about interactions and "how close are they" is the key signal.
    """

    def __init__(self, embed_dim=EMBED_DIM, hidden=256, dropout=0.3):
        super().__init__()
        # 2 people x (pose + velocity + acceleration) + 4 distances
        in_dim = 2 * PERSON_DIM * 3 + 4

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),  # the dataset is small (~6k motions), so we regularize a lot
        )

        # A GRU and an LSTM read the same sequence and their outputs are joined,
        # a bit like a small ensemble of two recurrent models. Bidirectional,
        # because the end of a clip helps understand the start (and vice versa).
        self.gru = nn.GRU(512, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.lstm = nn.LSTM(512, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)

        # Attention pooling: the model learns which frames matter (the moment
        # of the hug) instead of averaging over the whole clip.
        self.attn = nn.Linear(2 * hidden, 1)

        self.head = nn.Sequential(
            nn.Linear(4 * hidden, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

    def features(self, x):
        b, t, _ = x.shape
        p1, p2 = x[:, :, :PERSON_DIM], x[:, :, PERSON_DIM:]

        v1, v2 = time_diff(p1), time_diff(p2)
        a1, a2 = time_diff(v1), time_diff(v2)

        pos1 = p1[:, :, :POS_DIM].reshape(b, t, N_JOINTS, 3)
        pos2 = p2[:, :, :POS_DIM].reshape(b, t, N_JOINTS, 3)
        distances = [
            joint_distance(pos1, pos2, PELVIS, PELVIS),    # how far apart the two bodies are
            joint_distance(pos1, pos2, JOINT_A, JOINT_A),
            joint_distance(pos1, pos2, JOINT_B, JOINT_B),
            joint_distance(pos1, pos2, JOINT_B, PELVIS),   # person 1 reaching towards person 2
        ]
        return torch.cat([p1, p2, v1, v2, a1, a2, *distances], dim=-1)

    def attention_pool(self, seq):
        weights = torch.softmax(self.attn(seq), dim=1)
        return (seq * weights).sum(dim=1)

    def forward(self, x):
        h = self.input_proj(self.features(x))
        gru_out, _ = self.gru(h)
        lstm_out, _ = self.lstm(h)
        pooled = torch.cat([self.attention_pool(gru_out), self.attention_pool(lstm_out)], dim=-1)
        return F.normalize(self.head(pooled), dim=-1)


class TextEncoder(nn.Module):
    """A pretrained sentence model (MPNet) plus a linear layer to EMBED_DIM.

    With only ~6k motions there is not enough text to learn language from
    scratch, so we start from a model that already knows what sentences mean.
    The whole model is fine-tuned, not only the last layer.
    """

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", embed_dim=EMBED_DIM):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.model.config.hidden_size, embed_dim)

    def forward(self, texts):
        device = self.proj.weight.device
        tokens = self.tokenizer(list(texts), padding=True, truncation=True,
                                return_tensors="pt").to(device)
        out = self.model(**tokens).last_hidden_state

        # Mean over the real tokens only (padding tokens are masked out).
        # This is how sentence-transformers builds its sentence vectors.
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        sentence = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(self.proj(sentence), dim=-1)


class RetrievalModel(nn.Module):
    def __init__(self, text_model_name="sentence-transformers/all-mpnet-base-v2"):
        super().__init__()
        self.motion_encoder = MotionEncoder()
        self.text_encoder = TextEncoder(text_model_name)
        # Temperature of the softmax, learned instead of fixed. Starts at
        # 1 / 0.07 like in CLIP. Stored as a log so it stays positive.
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def forward(self, motions, texts):
        """Similarity matrix between every motion and every text of the batch."""
        z_motion = self.motion_encoder(motions)
        z_text = self.text_encoder(texts)
        # Clamped so the temperature cannot explode and make training unstable.
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * z_motion @ z_text.t()


def contrastive_loss(logits):
    """Symmetric InfoNCE loss (the CLIP loss).

    Row i should pick column i (motion i matches text i), and the same for
    columns. All the other texts in the batch act as negatives, which is why
    a bigger batch usually helps.
    """
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
