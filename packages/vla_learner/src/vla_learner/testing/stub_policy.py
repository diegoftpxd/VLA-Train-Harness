"""``StubPolicy``: política mínima para probar algoritmos sin un modelo real.

Definida en ``docs/Test Learners.md``. Es una Gaussiana diagonal con media y
desviación aprendibles, **independiente de la entrada**: solo usa el tamaño de
batch para dar la forma. No aprende tareas dependientes del estado —para eso
está la política tiny de la Fase 3—, pero basta para ejercitar la mecánica de
BC, PPO y GRPO (gradientes, recorte, dirección del paso). Sus capacidades son
configurables.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn

from vla_common import Observation
from vla_learner.policy.interfaces import Capability, ObservationProcessor, VLAPolicy
from vla_learner.types import ActionEvaluation, LossOutput, ModelInput, PolicyOutput


class StubProcessor(ObservationProcessor):
    """Procesador mínimo: apila la propiocepción en un tensor ``[B, P]``."""

    def __init__(self, cameras: list[str] | None = None) -> None:
        self._cameras = list(cameras or [])

    def __call__(self, observations: list[Observation]) -> ModelInput:
        proprio = torch.stack(
            [torch.as_tensor(o.proprio, dtype=torch.float32) for o in observations]
        )
        return ModelInput(tensors={"proprio": proprio}, batch_size=len(observations))

    @property
    def required_cameras(self) -> list[str]:
        return list(self._cameras)


class StubPolicy(VLAPolicy):
    def __init__(
        self,
        action_dim: int = 2,
        chunk_size: int = 1,
        capabilities: set[Capability] | None = None,
        init_std: float = 1.0,
    ) -> None:
        super().__init__()
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._capabilities = (
            set(capabilities)
            if capabilities is not None
            else {Capability.LOG_PROB, Capability.VALUE, Capability.ENTROPY}
        )
        self.mean = nn.Parameter(torch.zeros(chunk_size, action_dim))
        self.log_std = nn.Parameter(torch.full((chunk_size, action_dim), math.log(init_std)))
        self.value_bias = nn.Parameter(torch.zeros(()))
        self.processor: ObservationProcessor = StubProcessor()

    @property
    def capabilities(self) -> set[Capability]:
        return set(self._capabilities)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _distribution(self) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.mean, self.log_std.exp())

    def act(self, inputs: ModelInput, deterministic: bool = False) -> PolicyOutput:
        batch_size = inputs.batch_size
        dist = self._distribution()
        if deterministic:
            actions = self.mean.unsqueeze(0).expand(batch_size, -1, -1).clone()
        else:
            actions = dist.sample((batch_size,))
        log_probs = (
            dist.log_prob(actions).sum(dim=(1, 2))
            if Capability.LOG_PROB in self._capabilities
            else None
        )
        values = (
            self.value_bias.expand(batch_size).clone()
            if Capability.VALUE in self._capabilities
            else None
        )
        return PolicyOutput(actions=actions, log_probs=log_probs, values=values, extras={})

    def evaluate_actions(
        self,
        inputs: ModelInput,
        actions: Tensor,
        extras: dict[str, Tensor] | None = None,
    ) -> ActionEvaluation:
        if Capability.LOG_PROB not in self._capabilities:
            raise NotImplementedError("StubPolicy sin LOG_PROB no evalúa acciones.")
        batch_size = actions.shape[0]
        dist = self._distribution()
        log_probs = dist.log_prob(actions).sum(dim=(1, 2))
        entropy = (
            dist.entropy().sum().expand(batch_size)
            if Capability.ENTROPY in self._capabilities
            else None
        )
        values = (
            self.value_bias.expand(batch_size).clone()
            if Capability.VALUE in self._capabilities
            else None
        )
        return ActionEvaluation(log_probs=log_probs, entropy=entropy, values=values)

    def imitation_loss(self, inputs: ModelInput, actions: Tensor) -> LossOutput:
        dist = self._distribution()
        nll = -dist.log_prob(actions).sum(dim=(1, 2)).mean()
        return LossOutput(loss=nll, metrics={"nll": float(nll.detach())})

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state": self.state_dict(),
                "action_dim": self._action_dim,
                "chunk_size": self._chunk_size,
                "capabilities": [c.value for c in self._capabilities],
            },
            directory / "stub.pt",
        )

    @classmethod
    def from_pretrained(cls, directory: Path) -> StubPolicy:
        data = torch.load(Path(directory) / "stub.pt", weights_only=False)
        policy = cls(
            action_dim=data["action_dim"],
            chunk_size=data["chunk_size"],
            capabilities={Capability(v) for v in data["capabilities"]},
        )
        policy.load_state_dict(data["state"])
        return policy
