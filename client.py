"""NOC Agent Environment — OpenEnv WebSocket client."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ActionType, IncidentType, NOCAction, NOCObservation, SystemMetrics


class NocAgentEnvClient(EnvClient[NOCAction, NOCObservation, State]):
    """
    WebSocket client for the NOC Agent Environment server.

    Example::

        with NocAgentEnvClient(base_url="http://localhost:8000") as client:
            result = client.reset()
            print(result.observation.incident_type)

            result = client.step(NOCAction(action_type=ActionType.THROTTLE_CPU))
            print(result.observation.explanation)
    """

    def _step_payload(self, action: NOCAction) -> dict[str, Any]:
        return {"action_type": action.action_type.value}

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[NOCObservation]:
        obs_data = payload.get("observation", {})
        metrics_data = obs_data.get("metrics", {})

        metrics = SystemMetrics(
            cpu_usage=metrics_data.get("cpu_usage", 0.5),
            memory_usage=metrics_data.get("memory_usage", 0.5),
            latency=metrics_data.get("latency", 0.1),
            packet_loss=metrics_data.get("packet_loss", 0.0),
            service_healthy=metrics_data.get("service_healthy", 1.0),
            error_rate=metrics_data.get("error_rate", 0.0),
        )
        observation = NOCObservation(
            metrics=metrics,
            incident_type=IncidentType(obs_data.get("incident_type", IncidentType.CPU_OVERLOAD)),
            step=obs_data.get("step", 0),
            explanation=obs_data.get("explanation", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
