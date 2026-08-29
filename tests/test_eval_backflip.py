"""Unit checks for hierarchical backflip evaluation context handling."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_backflip import _specialist_observation


def test_airborne_specialist_preserves_global_phase_and_clears_assist():
    obs = torch.zeros((2, 61))
    obs[:, 55] = torch.tensor([0.8, 0.9])
    obs[:, 56] = 1.0

    specialized = _specialist_observation(
        obs,
        torch.tensor([0, 5]),
        0.02,
        local_phase=False,
    )

    assert torch.equal(specialized[:, 55], obs[:, 55])
    assert torch.count_nonzero(specialized[:, 56]) == 0


def test_touchdown_specialist_uses_local_elapsed_phase():
    obs = torch.ones((2, 61))
    specialized = _specialist_observation(
        obs,
        torch.tensor([0, 15]),
        0.02,
        local_phase=True,
    )

    assert specialized[0, 55] == 0.0
    assert float(specialized[1, 55]) == pytest.approx(0.5)
    assert torch.count_nonzero(specialized[:, 56]) == 0
