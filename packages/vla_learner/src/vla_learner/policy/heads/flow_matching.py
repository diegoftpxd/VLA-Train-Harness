"""``FlowMatchingActionHead``: cabeza de acción de *flow matching*, estilo π0.

Es la cabeza que ``docs/learner/Learner Design.md`` anticipa para la *mecánica* de
modelos como **π0 / π0.5** (y, con la variante de *denoising*, Octo): un VLA cuya
cabeza no predice una distribución explícita sino un **campo de velocidad** que
transporta ruido gaussiano hasta la acción. Es un **bloque de construcción
reutilizable y liviano** (corre en CPU) para montar políticas de flow matching
como ``ComposedPolicy`` = backbone + esta cabeza, sin tocar el resto del Learner.

**No es el π0 pesado.** Los pesos reales de π0 descargados de Hugging Face entran
por el adaptador ``Pi0Policy`` (``policy/adapters/pi0.py``, grupo opcional ``pi0``),
que delega el flow matching en LeRobot. Esta cabeza sirve para experimentar con el
objetivo en CPU y para modelos de flow matching propios.

**Objetivo (rectified flow).** Para una acción objetivo normalizada ``a1`` se
sortea ``t ~ U(0, 1)`` y ruido ``eps ~ N(0, I)``, se arma el punto intermedio
``x_t = t·a1 + (1 − t)·eps`` y se entrena la red a predecir la velocidad
constante del camino recto ``u = a1 − eps``. La loss de BC es el MSE entre la
velocidad predicha y ``u`` (``docs/learner/Learner Design.md``: "flow matching (MSE
sobre ruido)").

**Muestreo.** Integra el ODE ``dx/dt = v_θ(cond, x, t)`` con Euler desde ``t=0``
hasta ``t=1`` en ``num_sampling_steps`` pasos. ``deterministic=True`` arranca del
origen (lectura reproducible, POL-04); ``deterministic=False`` arranca de ruido
``N(0, I)`` del RNG global (reproducible al fijar la semilla, POL-05).

**Capacidades: ninguna.** El flow matching no da una log-prob tratable, así que
no declara ``LOG_PROB`` ni ``ENTROPY`` (``log_prob``/``entropy`` lanzan
``NotImplementedError``) y no tiene cabeza de valor. Es compatible con
**Behavior Cloning** (y un futuro ``RewardWeightedBC``), no con PPO/GRPO, que el
Learner descarta al chequear capacidades contra el algoritmo.

**Tamaño.** La red es un MLP condicionado en las features del backbone, el punto
``x_t`` y un *embedding* del tiempo; corre en CPU. El π0 real se obtiene usando
un backbone PaliGemma en lugar del de prueba, sin cambiar esta cabeza.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vla_learner.policy.interfaces import ActionHead, Capability
from vla_learner.types import LossOutput


class FlowMatchingActionHead(ActionHead):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
        num_sampling_steps: int = 10,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self._action_dim = action_dim
        self._chunk_size = chunk_size
        self._num_sampling_steps = num_sampling_steps
        self._flat = chunk_size * action_dim

        # Embedding del tiempo escalar t -> vector, para condicionar la velocidad.
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # Campo de velocidad v_θ(cond, x_t, t): [features ⊕ x_t ⊕ emb(t)] -> velocidad.
        in_dim = feature_dim + self._flat + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self._flat),
        )

    @property
    def capabilities(self) -> set[Capability]:
        return set()  # flow matching: sin log-prob tratable

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _velocity(self, cond: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Velocidad predicha ``[B, H, D]`` en el punto ``x`` y tiempo ``t``.

        ``cond`` ``[B, F]`` son las features del backbone (ya agrupadas), ``x``
        ``[B, H, D]`` el estado del ODE y ``t`` ``[B]`` el tiempo por muestra.
        """
        batch = cond.shape[0]
        temb = self.time_mlp(t.reshape(batch, 1))
        entrada = torch.cat([cond, x.reshape(batch, -1), temb], dim=1)
        return self.net(entrada).reshape(batch, self._chunk_size, self._action_dim)

    def sample(self, features: Tensor, deterministic: bool) -> tuple[Tensor, Tensor | None, dict]:
        cond = features.mean(dim=1)  # [B, F]
        batch = cond.shape[0]
        forma = (batch, self._chunk_size, self._action_dim)
        if deterministic:
            x = torch.zeros(forma, device=cond.device, dtype=cond.dtype)
        else:
            x = torch.randn(forma, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / self._num_sampling_steps
        for paso in range(self._num_sampling_steps):
            t = torch.full((batch,), paso * dt, device=cond.device, dtype=cond.dtype)
            x = x + dt * self._velocity(cond, x, t)
        # Sin log-prob (flow matching); extras vacío.
        return x, None, {}

    def log_prob(self, features: Tensor, actions: Tensor, extras: dict | None) -> Tensor:
        raise NotImplementedError(
            "FlowMatchingActionHead no expone log-prob (flow matching): no declara "
            "LOG_PROB. Entrená con Behavior Cloning; PPO/GRPO no son compatibles."
        )

    def entropy(self, features: Tensor) -> Tensor:
        raise NotImplementedError(
            "FlowMatchingActionHead no expone entropía (flow matching): no declara ENTROPY."
        )

    def bc_loss(self, features: Tensor, actions: Tensor) -> LossOutput:
        cond = features.mean(dim=1)
        a1 = actions  # [B, H, D] normalizadas
        eps = torch.randn_like(a1)
        t = torch.rand(a1.shape[0], device=a1.device, dtype=a1.dtype)
        t_exp = t.reshape(-1, 1, 1)
        x_t = t_exp * a1 + (1.0 - t_exp) * eps
        objetivo = a1 - eps  # velocidad constante del camino recto
        velocidad = self._velocity(cond, x_t, t)
        loss = F.mse_loss(velocidad, objetivo)
        return LossOutput(loss=loss, metrics={"fm_mse": float(loss.detach())})
