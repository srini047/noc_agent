"""
NOCSystemEnv — Gymnasium-compatible environment for RL training.

This wraps SystemSimulator behind the standard gymnasium.Env interface so
that stable-baselines3 (and any other Gymnasium-compatible RL library) can
train against it without needing the OpenEnv HTTP server.

Observation space : Box(6,) — normalised system metrics in [0, 1]
Action space      : Discrete(6) — indexed by ACTION_INDEX order
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from .explainability import ActionExplainer
from .models import ACTION_INDEX, ActionType, IncidentType, SystemMetrics
from .simulator import SystemSimulator, StepInfo


class NOCSystemEnv(gymnasium.Env):
    """
    Single-incident NOC triage environment for Gymnasium / SB3.

    Parameters
    ----------
    incident_type:
        Fix to a specific incident (useful for curriculum learning or evaluation).
        If None, a random incident is chosen on each reset.
    seed:
        RNG seed forwarded to the simulator for reproducibility.
    max_steps:
        Override the profile's default max_steps (None = use profile default).
    """

    metadata = {"render_modes": ["human", "ansi"]}

    NUM_METRICS = 6
    NUM_ACTIONS = len(ACTION_INDEX)

    observation_space = spaces.Box(
        low=np.zeros(NUM_METRICS, dtype=np.float32),
        high=np.ones(NUM_METRICS, dtype=np.float32),
        dtype=np.float32,
    )
    action_space = spaces.Discrete(NUM_ACTIONS)

    def __init__(
        self,
        incident_type: IncidentType | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._fixed_incident = incident_type
        self._max_steps_override = max_steps
        self._render_mode = render_mode

        self._simulator = SystemSimulator(seed=seed)
        self._explainer = ActionExplainer()
        self._rng = random.Random(seed)

        self._current_incident: IncidentType | None = None
        self._last_metrics: SystemMetrics | None = None
        self._last_action: ActionType | None = None
        self._last_explanation: str = ""

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002 — required by Gymnasium API
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        self._current_incident = (
            self._fixed_incident
            if self._fixed_incident is not None
            else self._rng.choice(list(IncidentType))
        )

        metrics = self._simulator.reset(self._current_incident)
        self._last_metrics = metrics
        self._last_action = None
        self._last_explanation = ""

        info = self._build_info(metrics, step=0, resolved=False, crashed=False)
        return metrics.to_array(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_type = ACTION_INDEX[action]
        metrics, step_info = self._simulator.step(action_type)

        self._last_metrics = metrics
        self._last_action = action_type
        self._last_explanation = self._explainer.explain(
            action_type, metrics, self._current_incident  # type: ignore[arg-type]
        )

        info = self._build_info(
            metrics,
            step=step_info.step,
            resolved=step_info.resolved,
            crashed=step_info.crashed,
            action_type=action_type,
            explanation=self._last_explanation,
            action_was_effective=step_info.action_was_effective,
        )
        return (
            metrics.to_array(),
            step_info.reward,
            step_info.done,
            step_info.truncated,
            info,
        )

    def render(self) -> str | None:
        if self._render_mode not in ("human", "ansi"):
            return None
        if self._last_metrics is None:
            return "Environment not started. Call reset() first."

        m = self._last_metrics
        action_str = self._last_action.value if self._last_action else "—"
        lines = [
            f"Incident : {self._current_incident.value if self._current_incident else '—'}",
            f"Step     : {self._simulator.current_step}",
            f"Action   : {action_str}",
            f"CPU      : {m.cpu_usage:.1%}",
            f"Memory   : {m.memory_usage:.1%}",
            f"Latency  : {m.latency * 500:.0f} ms",
            f"PktLoss  : {m.packet_loss:.1%}",
            f"Service  : {'OK' if m.service_healthy >= 1.0 else 'DEGRADED'}",
            f"ErrorRate: {m.error_rate:.1%}",
            f"Health   : {m.health_score:.1%}",
        ]
        output = "\n".join(lines)
        if self._render_mode == "human":
            print(output)
            return None
        return output

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Properties for external use (e.g. Gradio demo)
    # ------------------------------------------------------------------

    @property
    def current_incident(self) -> IncidentType | None:
        return self._current_incident

    @property
    def last_explanation(self) -> str:
        return self._last_explanation

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_info(
        metrics: SystemMetrics,
        step: int,
        resolved: bool,
        crashed: bool,
        action_type: ActionType | None = None,
        explanation: str = "",
        action_was_effective: bool = False,
    ) -> dict[str, Any]:
        return {
            "step": step,
            "resolved": resolved,
            "crashed": crashed,
            "health_score": metrics.health_score,
            "action": action_type.value if action_type else None,
            "explanation": explanation,
            "action_was_effective": action_was_effective,
            "metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "latency_ms": metrics.latency * 500,
                "packet_loss": metrics.packet_loss,
                "service_healthy": metrics.service_healthy,
                "error_rate": metrics.error_rate,
            },
        }
