"""
NocAgentEnvironment — OpenEnv server-side environment.

Wraps SystemSimulator behind the openenv.core Environment interface so
the trained policy can be exercised over HTTP / WebSocket.

The RL logic lives entirely in simulator.py / gym_env.py; this file only
handles protocol translation (OpenEnv Action → ActionType → OpenEnv Observation).
"""

from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..explainability import ActionExplainer
    from ..models import ACTION_INDEX, IncidentType, NOCAction, NOCObservation, SystemMetrics
    from ..simulator import SystemSimulator
except ImportError:
    import sys
    from pathlib import Path
    # Docker / standalone context: add the parent of noc_agent/ to sys.path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from noc_agent.explainability import ActionExplainer  # type: ignore[no-redef]
    from noc_agent.models import ACTION_INDEX, IncidentType, NOCAction, NOCObservation, SystemMetrics  # type: ignore[no-redef]
    from noc_agent.simulator import SystemSimulator  # type: ignore[no-redef]

import random


class NocAgentEnvironment(Environment):
    """
    OpenEnv server environment for the NOC Incident Triage agent.

    Each WebSocket session gets its own instance (SUPPORTS_CONCURRENT_SESSIONS=True).
    The environment accepts NOCAction (with an action_type enum value) and
    returns NOCObservation (with full system metrics + explanation).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._simulator = SystemSimulator()
        self._explainer = ActionExplainer()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_incident: IncidentType = IncidentType.CPU_OVERLOAD
        self._last_metrics: SystemMetrics | None = None

    def reset(self, incident_type: str | None = None) -> NOCObservation:  # type: ignore[override]
        """Start a new episode.  Pass incident_type to pin the scenario."""
        if incident_type is not None:
            self._current_incident = IncidentType(incident_type)
        else:
            self._current_incident = random.choice(list(IncidentType))
        self._state = State(episode_id=str(uuid4()), step_count=0)

        metrics = self._simulator.reset(self._current_incident)
        self._last_metrics = metrics

        return NOCObservation(
            metrics=metrics,
            incident_type=self._current_incident,
            step=0,
            explanation="New incident detected. Agent ready.",
            done=False,
            reward=0.0,
        )

    def step(self, action: NOCAction) -> NOCObservation:  # type: ignore[override]
        """Execute one action and return the updated observation."""
        self._state.step_count += 1

        action_type = action.action_type
        metrics, step_info = self._simulator.step(action_type)
        self._last_metrics = metrics

        explanation = self._explainer.explain(action_type, metrics, self._current_incident)

        return NOCObservation(
            metrics=metrics,
            incident_type=self._current_incident,
            step=self._state.step_count,
            explanation=explanation,
            done=step_info.done,
            reward=step_info.reward,
            metadata={
                "resolved": step_info.resolved,
                "crashed": step_info.crashed,
                "health_score": metrics.health_score,
                "action_was_effective": step_info.action_was_effective,
            },
        )

    @property
    def state(self) -> State:
        return self._state
