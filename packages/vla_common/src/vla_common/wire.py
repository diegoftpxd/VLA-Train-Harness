"""Conversión de los tipos del protocolo a/desde estructuras serializables.

Los procesos hablan con diccionarios y listas de tipos simples y arrays de numpy
(lo que ``serialization.encode`` sabe mandar como frames). Aquí están las
conversiones entre esos diccionarios y los dataclasses de ``types`` (``Observation``,
``Action``/``ActionChunk``, ``StepResult``, ``Transition``). Core y el Learner
usan las mismas, por eso viven en ``vla_common``.
"""

from __future__ import annotations

from typing import Any

from vla_common.types import Action, ActionChunk, Observation, StepResult, Transition


def observation_to_wire(observation: Observation) -> dict[str, Any]:
    return {
        "images": dict(observation.images),
        "proprio": observation.proprio,
        "instruction": observation.instruction,
    }


def observation_from_wire(data: dict[str, Any]) -> Observation:
    return Observation(
        images=dict(data["images"]),
        proprio=data["proprio"],
        instruction=data["instruction"],
    )


def action_chunk_to_wire(chunk: ActionChunk) -> list[Any]:
    return [action.values for action in chunk]


def action_chunk_from_wire(data: list[Any]) -> ActionChunk:
    return [Action(values=values) for values in data]


def step_result_to_wire(result: StepResult) -> dict[str, Any]:
    return {
        "observation": observation_to_wire(result.observation),
        "reward": result.reward,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "info": result.info,
    }


def step_result_from_wire(data: dict[str, Any]) -> StepResult:
    return StepResult(
        observation=observation_from_wire(data["observation"]),
        reward=data["reward"],
        terminated=data["terminated"],
        truncated=data["truncated"],
        info=data["info"],
    )


def transition_to_wire(transition: Transition) -> dict[str, Any]:
    return {
        "episode_id": transition.episode_id,
        "step": transition.step,
        "source": transition.source,
        "observation": observation_to_wire(transition.observation),
        "action": action_chunk_to_wire(transition.action),
        "result": step_result_to_wire(transition.result),
    }


def transition_from_wire(data: dict[str, Any]) -> Transition:
    return Transition(
        episode_id=data["episode_id"],
        step=data["step"],
        source=data["source"],
        observation=observation_from_wire(data["observation"]),
        action=action_chunk_from_wire(data["action"]),
        result=step_result_from_wire(data["result"]),
    )
