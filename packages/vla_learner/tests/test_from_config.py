"""Smoke de ``from_config`` sobre los configs tiny de ejemplo (el camino de la CLI)."""

from pathlib import Path

import pytest
import torch
import yaml

from vla_learner.learner import Learner
from vla_learner.policy.composed import ComposedPolicy
from vla_learner.policy.heads import GaussianActionHead

pytestmark = pytest.mark.fast

_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


@pytest.mark.parametrize("nombre", ["bc_tiny.yaml", "ppo_tiny.yaml"])
def test_from_config_ejemplos(tmp_path, nombre):
    config = yaml.safe_load((_CONFIGS / nombre).read_text(encoding="utf-8"))
    config["checkpoint"]["directory"] = str(tmp_path / "ckpt")
    learner = Learner.from_config(config)
    assert isinstance(learner.policy, ComposedPolicy)
    head = learner.policy.action_head
    assert isinstance(head, GaussianActionHead)
    antes = head.mean_proj.weight.detach().clone()
    learner.train(1)
    assert not torch.equal(antes, head.mean_proj.weight.detach())


def test_from_config_flow_matching(tmp_path):
    # Flow matching (mecánica de π0) de punta a punta: se arma desde el YAML y
    # entrena por BC. El π0 pesado real (policy type `pi0`) va en el cluster.
    from vla_learner.policy.heads import FlowMatchingActionHead

    config = yaml.safe_load((_CONFIGS / "flow_matching_bc.yaml").read_text(encoding="utf-8"))
    config["checkpoint"]["directory"] = str(tmp_path / "ckpt")
    learner = Learner.from_config(config)
    assert isinstance(learner.policy, ComposedPolicy)
    head = learner.policy.action_head
    assert isinstance(head, FlowMatchingActionHead)
    assert head.capabilities == set()  # sin log-prob: solo BC
    antes = [p.detach().clone() for p in head.parameters()]
    learner.train(1)
    assert any(not torch.equal(a, p) for a, p in zip(antes, head.parameters(), strict=True))
