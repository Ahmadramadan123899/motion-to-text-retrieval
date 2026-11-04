import numpy as np
import pytest

torch = pytest.importorskip("torch")

from motion_retrieval.data import (MAX_LEN, N_JOINTS, PERSON_DIM, POS_DIM,  # noqa: E402
                                   pad_or_cut, rotate_around_vertical, swap_people)


def fake_motion(frames=40, seed=0):
    return np.random.default_rng(seed).normal(size=(frames, 2 * PERSON_DIM))


def positions(motion, person):
    start = person * PERSON_DIM
    return motion[:, start:start + POS_DIM].reshape(-1, N_JOINTS, 3)


def test_rotation_keeps_distance_between_people():
    motion = fake_motion()
    rotated = rotate_around_vertical(motion)
    before = np.linalg.norm(positions(motion, 0) - positions(motion, 1), axis=-1)
    after = np.linalg.norm(positions(rotated, 0) - positions(rotated, 1), axis=-1)
    assert np.allclose(before, after)


def test_rotation_keeps_height():
    motion = fake_motion()
    rotated = rotate_around_vertical(motion)
    assert np.allclose(positions(motion, 0)[..., 1], positions(rotated, 0)[..., 1])


def test_swap_twice_gives_back_the_original():
    motion = fake_motion()
    assert np.array_equal(swap_people(swap_people(motion)), motion)


def test_pad_and_cut():
    assert pad_or_cut(fake_motion(10)).shape == (MAX_LEN, 2 * PERSON_DIM)
    assert pad_or_cut(fake_motion(MAX_LEN + 50)).shape == (MAX_LEN, 2 * PERSON_DIM)


def test_motion_encoder_output():
    from motion_retrieval.model import EMBED_DIM, MotionEncoder

    encoder = MotionEncoder().eval()
    x = torch.randn(3, 50, 2 * PERSON_DIM)
    z = encoder(x)
    assert z.shape == (3, EMBED_DIM)
    # Vectors are normalized, so the dot product is a cosine similarity.
    assert torch.allclose(z.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_loss_is_lower_when_pairs_match():
    from motion_retrieval.model import contrastive_loss

    good = torch.eye(4) * 10
    bad = torch.ones(4, 4) * 10 - torch.eye(4) * 10
    assert contrastive_loss(good) < contrastive_loss(bad)
