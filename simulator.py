"""
SystemSimulator — the core state machine for NOC incident simulation.

Responsibilities (Single Responsibility Principle):
  - Maintain and evolve SystemMetrics across episode steps
  - Apply incident drift and action effects
  - Track episode termination conditions

The simulator is framework-agnostic: it is shared by both the Gymnasium
training env and the OpenEnv HTTP server.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .incidents import IncidentProfile, get_profile
from .models import ActionType, IncidentType, SystemMetrics

# Gaussian noise std applied to every metric delta (adds realism)
_NOISE_STD: float = 0.008

# Consecutive steps all metrics must be within healthy range to count as resolved
_RESOLUTION_STREAK_REQUIRED: int = 3


@dataclass
class StepInfo:
    """Carries extra context returned alongside metrics after each step."""

    reward: float
    done: bool
    truncated: bool
    crashed: bool
    resolved: bool
    action_was_effective: bool
    step: int


class SystemSimulator:
    """
    Simulates a Linux system node affected by a single incident.

    Usage::

        sim = SystemSimulator(seed=42)
        metrics = sim.reset(IncidentType.CPU_OVERLOAD)
        metrics, info = sim.step(ActionType.THROTTLE_CPU)
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._random = random.Random(seed)

        self._profile: IncidentProfile | None = None
        self._metrics: SystemMetrics | None = None
        self._step: int = 0
        self._resolution_streak: int = 0
        self._prev_health: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, incident_type: IncidentType) -> SystemMetrics:
        """
        Start a new episode with the given incident type.

        Returns the initial SystemMetrics observation.
        """
        self._profile = get_profile(incident_type)
        self._step = 0
        self._resolution_streak = 0

        self._metrics = SystemMetrics(
            cpu_usage=self._jitter(self._profile.initial_cpu),
            memory_usage=self._jitter(self._profile.initial_memory),
            latency=self._jitter(self._profile.initial_latency),
            packet_loss=self._jitter(self._profile.initial_packet_loss),
            service_healthy=self._profile.initial_service_healthy,
            error_rate=self._jitter(self._profile.initial_error_rate),
        )
        self._prev_health = self._metrics.health_score
        return self._metrics

    def step(self, action: ActionType) -> tuple[SystemMetrics, StepInfo]:
        """
        Advance the simulation by one step.

        - First applies incident drift (worsening)
        - Then applies the chosen action effect
        - Adds small Gaussian noise for realism
        - Checks resolution and crash conditions

        Returns the updated metrics and a StepInfo with reward and flags.
        """
        if self._profile is None or self._metrics is None:
            raise RuntimeError("Call reset() before step()")

        effect = self._profile.action_effects[action]
        drift = self._profile.drift

        # Combine drift + action effect into a single delta per metric
        cpu = self._metrics.cpu_usage + drift.cpu_usage + effect.cpu_usage
        memory = self._metrics.memory_usage + drift.memory_usage + effect.memory_usage
        latency = self._metrics.latency + drift.latency + effect.latency
        packet_loss = self._metrics.packet_loss + drift.packet_loss + effect.packet_loss
        error_rate = self._metrics.error_rate + drift.error_rate + effect.error_rate

        # Service healthy: restart brings it back to 1.0 after one step down
        if action == ActionType.RESTART_SERVICE:
            service_healthy = 0.0  # momentarily down during restart
        else:
            # Gradually recover if error_rate is falling
            service_healthy = 1.0 if error_rate < 0.50 else max(0.0, self._metrics.service_healthy - 0.1)

        # Add Gaussian noise to all continuous metrics
        noise = self._rng.normal(0.0, _NOISE_STD, 5)
        cpu += noise[0]
        memory += noise[1]
        latency += noise[2]
        packet_loss += noise[3]
        error_rate += noise[4]

        # Clamp all values to [0, 1]
        self._metrics = SystemMetrics(
            cpu_usage=float(np.clip(cpu, 0.0, 1.0)),
            memory_usage=float(np.clip(memory, 0.0, 1.0)),
            latency=float(np.clip(latency, 0.0, 1.0)),
            packet_loss=float(np.clip(packet_loss, 0.0, 1.0)),
            service_healthy=float(np.clip(service_healthy, 0.0, 1.0)),
            error_rate=float(np.clip(error_rate, 0.0, 1.0)),
        )

        self._step += 1
        current_health = self._metrics.health_score

        # Resolution streak tracking
        if self._metrics.is_resolved:
            self._resolution_streak += 1
        else:
            self._resolution_streak = 0

        resolved = self._resolution_streak >= _RESOLUTION_STREAK_REQUIRED
        crashed = self._metrics.is_critical
        truncated = self._step >= self._profile.max_steps
        done = resolved or crashed or truncated

        # Detect whether the action had any meaningful positive effect
        action_was_effective = current_health > self._prev_health + 0.005
        self._prev_health = current_health

        reward = self._calculate_reward(
            metrics=self._metrics,
            prev_health=self._prev_health,
            current_health=current_health,
            resolved=resolved,
            crashed=crashed,
            action=action,
            action_was_effective=action_was_effective,
        )

        return self._metrics, StepInfo(
            reward=reward,
            done=done,
            truncated=truncated,
            crashed=crashed,
            resolved=resolved,
            action_was_effective=action_was_effective,
            step=self._step,
        )

    @property
    def current_metrics(self) -> SystemMetrics | None:
        return self._metrics

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def profile(self) -> IncidentProfile | None:
        return self._profile

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _jitter(self, value: float, std: float = 0.02) -> float:
        """Add small noise to initial values so each episode differs slightly."""
        noisy = value + float(self._rng.normal(0.0, std))
        return float(np.clip(noisy, 0.0, 1.0))

    def _calculate_reward(
        self,
        metrics: SystemMetrics,  # noqa: ARG002 — reserved for future metric-specific shaping
        prev_health: float,
        current_health: float,
        resolved: bool,
        crashed: bool,
        action: ActionType,
        action_was_effective: bool,
    ) -> float:
        # Continuous health signal: reward improvement, penalise worsening
        health_delta = (current_health - prev_health) * 5.0

        # Absolute health bonus: always positive nudge toward staying healthy
        health_bonus = current_health * 0.5

        # Per-step cost: encourages speed
        step_penalty = -0.10

        # Terminal bonuses/penalties
        resolution_bonus = 15.0 if resolved else 0.0
        crash_penalty = -10.0 if crashed else 0.0

        # Penalise completely ineffective actions (not DO_NOTHING which is a valid choice)
        ineffective_penalty = (
            -0.30
            if (not action_was_effective and action != ActionType.DO_NOTHING and not resolved)
            else 0.0
        )

        return health_delta + health_bonus + step_penalty + resolution_bonus + crash_penalty + ineffective_penalty
