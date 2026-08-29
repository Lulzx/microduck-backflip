import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.retarget_checkpoint_std import STD_KEY, retarget_checkpoint


def test_retarget_checkpoint_preserves_policy_and_resets_optimizer(tmp_path: Path):
    source = tmp_path / "source.pt"
    destination = tmp_path / "destination.pt"
    torch.save(
        {
            "actor_state_dict": {
                "mlp.weight": torch.tensor([3.0]),
                STD_KEY: torch.tensor([1.2, 0.9]),
            },
            "critic_state_dict": {"mlp.weight": torch.tensor([4.0])},
            "optimizer_state_dict": {
                "state": {0: {"step": torch.tensor(9.0), "exp_avg": torch.ones(2)}},
                "param_groups": [{"params": [0]}],
            },
            "iter": 640,
            "infos": {"env_state": {"common_step_counter": 15360}},
        },
        source,
    )

    retarget_checkpoint(source, destination, 0.2)
    result = torch.load(destination, map_location="cpu", weights_only=False)
    assert torch.equal(result["actor_state_dict"]["mlp.weight"], torch.tensor([3.0]))
    assert torch.equal(result["critic_state_dict"]["mlp.weight"], torch.tensor([4.0]))
    assert torch.equal(result["actor_state_dict"][STD_KEY], torch.full((2,), 0.2))
    assert result["optimizer_state_dict"]["state"][0]["exp_avg"].count_nonzero() == 0
    assert result["iter"] == 640
    assert result["infos"]["env_state"]["common_step_counter"] == 15360
