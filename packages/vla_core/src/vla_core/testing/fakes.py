"""Fakes en memoria de los componentes que Core consume, y sus servidores ZeroMQ.

Definidos en ``docs/core/Test Core.md``. Son livianos, deterministas y en CPU:
implementan las mismas ABCs que un componente real, para construir y probar Core
sin simuladores, world models ni reward models de verdad. Cada fake trae además
un servidor ``RouterServer`` para probar su cliente ``DEALER`` por la red.

El ambiente es un "reach" en 2D (como ``ToyReachEnv`` del Learner, pero propio de
``vla_core`` para no depender del Learner): un punto debe llegar a una meta.
"""

from __future__ import annotations

import threading

import numpy as np
import zmq

from vla_common.environment import (
    Environment,
    EnvSpec,
    check_action_chunk,
    handle_environment,
    resolve_task,
)
from vla_common.types import Action, ActionChunk, Observation, StepResult
from vla_core.core import Core
from vla_core.dataset import AppendableDataset, Dataset
from vla_core.environments import EnvironmentRegistry
from vla_core.episodes import EpisodeManager, InMemoryEpisodeStore
from vla_core.errors import DownstreamTimeout
from vla_core.reward import RewardModel, RewardScorer, ScoreItem, handle_reward_model
from vla_core.rpc import RouterServer, make_handler
from vla_core.world_model import LatentState, WorldModel, handle_world_model

BOUND = 1.0
THRESHOLD = 0.1
MAX_STEPS = 20
TASK = "reach the goal"
FAKE_SPEC = EnvSpec(action_dim=2, cameras=["front"], tasks=[TASK])


def _observation(pos: np.ndarray, goal: np.ndarray, instruction: str = TASK) -> Observation:
    return Observation(
        images={"front": np.zeros((3, 8, 8), dtype=np.float32)},
        proprio=np.concatenate([pos, goal]).astype(np.float32),
        instruction=instruction,
    )


class FakeEnvironment(Environment):
    """Ambiente reach 2D, determinista por semilla. ``step`` no calcula recompensa.

    Cumple las convenciones del Environment (``docs/environment``): corta el chunk
    en la acción que alcanza la meta, marca ``info["success"]`` y rechaza un
    ``step`` después de un final. El límite de pasos cuenta **chunks** (no
    acciones) para seguir idéntico al ``ToyReachEnv`` del Learner.
    """

    def __init__(self) -> None:
        self._pos = np.zeros(2, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._steps = 0
        self._done = False

    def reset(self, task: str | None, seed: int | None) -> Observation:
        resolve_task(task, FAKE_SPEC)
        rng = np.random.default_rng(seed)
        self._pos = rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
        self._goal = rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
        self._steps = 0
        self._done = False
        return _observation(self._pos, self._goal)

    def step(self, actions: ActionChunk) -> StepResult:
        if self._done:
            raise RuntimeError("el episodio ya terminó; hace falta reset()")
        check_action_chunk(actions, FAKE_SPEC.action_dim)
        reached = False
        for action in actions:
            self._pos = np.clip(
                self._pos + np.asarray(action.values, dtype=np.float32), -BOUND, BOUND
            ).astype(np.float32)
            if float(np.linalg.norm(self._pos - self._goal)) < THRESHOLD:
                reached = True
                break  # el episodio termina a mitad del chunk: se corta acá
        self._steps += 1
        distancia = float(np.linalg.norm(self._pos - self._goal))
        truncated = self._steps >= MAX_STEPS and not reached
        self._done = reached or truncated
        return StepResult(
            observation=_observation(self._pos, self._goal),
            reward=None,  # la agrega Core
            terminated=reached,
            truncated=truncated,
            info={"success": reached, "distance": distancia},
        )

    def close(self) -> None:
        pass

    def emergency_stop(self) -> None:
        pass

    @property
    def spec(self) -> EnvSpec:
        return FAKE_SPEC


class FakeWorldModel(WorldModel):
    """Simula la misma dinámica reach. El latente guarda ``[pos, goal]``.

    ``rollout`` devuelve ``StepResult`` con ``reward=None`` (la recompensa la pone
    Core), ``terminated`` al llegar a la meta y ``truncated`` al límite de pasos,
    cortando la secuencia si el episodio termina. Misma semántica que el
    ``ToyWorldModel`` del Learner.
    """

    def encode(self, observation: Observation) -> LatentState:
        return LatentState(values=np.asarray(observation.proprio, dtype=np.float32).copy())

    def rollout(
        self, latent: LatentState, action_sequences: list[list[ActionChunk]]
    ) -> list[list[StepResult]]:
        salida: list[list[StepResult]] = []
        for secuencia in action_sequences:
            pos = latent.values[:2].astype(np.float32).copy()
            goal = latent.values[2:4].astype(np.float32).copy()
            pasos: list[StepResult] = []
            steps = 0
            for chunk in secuencia:
                reached = False
                for action in chunk:
                    pos = np.clip(
                        pos + np.asarray(action.values, dtype=np.float32), -BOUND, BOUND
                    ).astype(np.float32)
                    if float(np.linalg.norm(pos - goal)) <= THRESHOLD:
                        reached = True
                        break
                steps += 1
                truncated = (not reached) and steps >= MAX_STEPS
                pasos.append(
                    StepResult(
                        observation=_observation(pos, goal),
                        reward=None,  # la agrega Core
                        terminated=reached,
                        truncated=truncated,
                        info={"distance": float(np.linalg.norm(pos - goal))},
                    )
                )
                if reached or truncated:
                    break
            salida.append(pasos)
        return salida


class FlakyEnvironment(Environment):
    """Ambiente caído en memoria: simula un proceso que no responde.

    ``reset``/``step`` levantan ``DownstreamTimeout`` (como lo haría el
    ``EnvironmentClient`` tras agotar timeout y reintentos), para ejercitar el
    manejo de fallas de Core sin red. Registra cuántas veces se invocó la parada
    de seguridad, para verificar la política del robot.
    """

    def __init__(self, *, fail_reset: bool = False, fail_step: bool = True) -> None:
        self._fail_reset = fail_reset
        self._fail_step = fail_step
        self.emergency_stops = 0
        self.closed = False

    def reset(self, task: str | None, seed: int | None) -> Observation:
        if self._fail_reset:
            raise DownstreamTimeout("ambiente caído en reset")
        return _observation(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32))

    def step(self, actions: ActionChunk) -> StepResult:
        if self._fail_step:
            raise DownstreamTimeout("ambiente caído en step")
        return StepResult(
            observation=_observation(np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)),
            reward=None,
            terminated=False,
            truncated=False,
            info={"success": False},
        )

    def close(self) -> None:
        self.closed = True

    def emergency_stop(self) -> None:
        self.emergency_stops += 1

    @property
    def spec(self) -> EnvSpec:
        return FAKE_SPEC


class FakeRewardModel(RewardModel):
    """Recompensa = distancia negativa del punto a la meta en la observación siguiente."""

    def score(self, items: list[ScoreItem]) -> list[float]:
        recompensas: list[float] = []
        for item in items:
            proprio = np.asarray(item.next_observation.proprio, dtype=np.float32)
            distancia = float(np.linalg.norm(proprio[:2] - proprio[2:4]))
            recompensas.append(-distancia)
        return recompensas


class FakeDataset(AppendableDataset):
    """Pares supervisados deterministas con un controlador perfecto.

    Respeta el filtro ``task`` (la instrucción de los pares generados) e ignora
    los criterios que no modela (p. ej. ``embodiment``, que la observación reach
    no tiene). Admite reinyección (``add``): los pares agregados por el gancho de
    episodios exitosos se sirven antes que los generados al vuelo.
    """

    def __init__(self, seed: int = 0, demo_chunk_size: int = 1) -> None:
        self._rng = np.random.default_rng(seed)
        self._demo_chunk_size = demo_chunk_size
        self._ingested: list[tuple[Observation, ActionChunk]] = []

    def sample(self, n: int, filters: dict | None = None) -> list[tuple[Observation, ActionChunk]]:
        task = (filters or {}).get("task")
        # Primero los pares reinyectados que cumplen el filtro; luego se rellena
        # generando al vuelo hasta llegar a n.
        pares: list[tuple[Observation, ActionChunk]] = [
            par for par in self._ingested if task is None or par[0].instruction == task
        ][:n]
        instruction = task if task is not None else "reach the goal"
        while len(pares) < n:
            pos = self._rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
            goal = self._rng.uniform(-BOUND, BOUND, size=2).astype(np.float32)
            delta = ((goal - pos) / self._demo_chunk_size).astype(np.float32)
            chunk: ActionChunk = [Action(values=delta.copy()) for _ in range(self._demo_chunk_size)]
            pares.append((_observation(pos, goal, instruction), chunk))
        return pares

    def add(self, pairs: list[tuple[Observation, ActionChunk]]) -> int:
        self._ingested.extend(pairs)
        return len(pairs)


# -- Servidores ZeroMQ de prueba (para ejercitar los clientes DEALER) -------


def serve_environment(env: Environment, address: str = "tcp://127.0.0.1:0") -> RouterServer:
    return RouterServer(make_handler(env, handle_environment), address)


def serve_world_model(model: WorldModel, address: str = "tcp://127.0.0.1:0") -> RouterServer:
    return RouterServer(make_handler(model, handle_world_model), address)


def serve_reward_model(model: RewardModel, address: str = "tcp://127.0.0.1:0") -> RouterServer:
    return RouterServer(make_handler(model, handle_reward_model), address)


class SilentRouter:
    """Socket ``ROUTER`` que recibe los pedidos y **nunca** responde.

    Simula un proceso colgado (vivo, pero sin atender): el ``RpcChannel`` del
    cliente agota su timeout y reintentos y levanta ``DownstreamTimeout``. Sirve
    para probar el timeout y la reconexión sin depender de tiempos de red.
    """

    def __init__(self, address: str = "tcp://127.0.0.1:0") -> None:
        self._requested_address = address
        self.address = address
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> str:
        socket = self._context.socket(zmq.ROUTER)
        socket.bind(self._requested_address)
        self.address = socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._socket = socket
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self.address

    def _serve(self) -> None:
        assert self._socket is not None
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            # Vacía la cola de entrada pero jamás contesta.
            if dict(poller.poll(timeout=100)).get(self._socket) == zmq.POLLIN:
                self._socket.recv_multipart()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None

    def __enter__(self) -> SilentRouter:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


# -- Ensamblaje de un Core de prueba ----------------------------------------


class FakeEnvironmentRegistry(EnvironmentRegistry):
    """Registro que crea un ``FakeEnvironment`` al vuelo para cualquier ``env_id``.

    Imita el auto-registro del ``FakeCoreClient`` del Learner, para poder correr
    la misma suite CORE (que usa ids arbitrarios como ``"e0"`` o ``"a"``) contra
    el Core real sin configurar los ambientes de antemano.
    """

    def get(self, env_id: str) -> Environment:
        if env_id not in self._envs:
            self._envs[env_id] = FakeEnvironment()
        return self._envs[env_id]


def make_fake_core(
    demo_chunk_size: int = 1,
    dataset: Dataset | None = None,
    environments: EnvironmentRegistry | None = None,
) -> Core:
    """Arma un ``Core`` con todos los componentes fake en memoria.

    ``dataset`` permite inyectar otro ``Dataset`` (p. ej. un ``LocalDataset``
    construido desde config); si es ``None`` se usa ``FakeDataset``.
    ``environments`` permite inyectar un registro con ambientes reales (p. ej.
    clientes a procesos ``vla-env``); si es ``None``, cualquier ``env_id``
    obtiene un ``FakeEnvironment`` al vuelo.
    """
    return Core(
        environments=environments if environments is not None else FakeEnvironmentRegistry(),
        episodes=EpisodeManager(),
        store=InMemoryEpisodeStore(),
        scorer=RewardScorer(FakeRewardModel()),
        world_model=FakeWorldModel(),
        dataset=dataset if dataset is not None else FakeDataset(demo_chunk_size=demo_chunk_size),
    )
