"""
Data models for the NOC Agent environment.

Defines all Pydantic models for actions, observations, and system metrics
used by both the Gymnasium training environment and the OpenEnv server.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Discrete actions available to the NOC agent."""

    DO_NOTHING = "do_nothing"
    RESTART_SERVICE = "restart_service"
    THROTTLE_CPU = "throttle_cpu"
    CLEAR_CACHE = "clear_cache"
    REROUTE_TRAFFIC = "reroute_traffic"
    SCALE_UP = "scale_up"


class IncidentType(str, Enum):
    """Supported incident types the simulator can inject."""

    CPU_OVERLOAD = "cpu_overload"
    MEMORY_LEAK = "memory_leak"
    NETWORK_CONGESTION = "network_congestion"


# Ordered list used to map integer indices to ActionType (for Gymnasium Discrete space)
ACTION_INDEX: list[ActionType] = list(ActionType)


class SystemMetrics(BaseModel):
    """
    Normalised system health metrics.

    All values are in [0.0, 1.0] unless noted.
    Higher values indicate more stress (worse health) except service_healthy.
    """

    cpu_usage: float = Field(..., ge=0.0, le=1.0, description="CPU utilisation (0=idle, 1=fully saturated)")
    memory_usage: float = Field(..., ge=0.0, le=1.0, description="RAM utilisation")
    latency: float = Field(..., ge=0.0, le=1.0, description="Network latency normalised over 500 ms")
    packet_loss: float = Field(..., ge=0.0, le=1.0, description="Fraction of packets dropped")
    service_healthy: float = Field(..., ge=0.0, le=1.0, description="1.0 = healthy, 0.0 = down")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Fraction of requests returning errors")

    def to_array(self) -> np.ndarray:
        """Return metrics as a flat float32 numpy array for the Gymnasium observation space."""
        return np.array(
            [
                self.cpu_usage,
                self.memory_usage,
                self.latency,
                self.packet_loss,
                self.service_healthy,
                self.error_rate,
            ],
            dtype=np.float32,
        )

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "SystemMetrics":
        """Reconstruct from a flat numpy array (must have 6 elements)."""
        return cls(
            cpu_usage=float(arr[0]),
            memory_usage=float(arr[1]),
            latency=float(arr[2]),
            packet_loss=float(arr[3]),
            service_healthy=float(arr[4]),
            error_rate=float(arr[5]),
        )

    @property
    def health_score(self) -> float:
        """Aggregate health score in [0, 1].  1.0 = fully healthy."""
        stress = (
            self.cpu_usage * 0.25
            + self.memory_usage * 0.25
            + self.latency * 0.20
            + self.packet_loss * 0.15
            + (1.0 - self.service_healthy) * 0.10
            + self.error_rate * 0.05
        )
        return max(0.0, 1.0 - stress)

    @property
    def is_critical(self) -> bool:
        """True if any metric has exceeded crash thresholds."""
        return (
            self.cpu_usage >= 0.98
            or self.memory_usage >= 0.98
            or self.error_rate >= 0.90
        )

    @property
    def is_resolved(self) -> bool:
        """True when all metrics are comfortably below healthy thresholds."""
        return (
            self.cpu_usage < 0.65
            and self.memory_usage < 0.65
            and self.latency < 0.20
            and self.packet_loss < 0.05
            and self.service_healthy >= 1.0
            and self.error_rate < 0.10
        )


# ---------------------------------------------------------------------------
# OpenEnv action / observation (used by the server and HTTP client)
# ---------------------------------------------------------------------------


class NOCAction(Action):
    """Action sent by a client to the NOC environment server."""

    action_type: ActionType = Field(..., description="Discrete action to apply")


class NOCObservation(Observation):
    """Observation returned by the NOC environment server after each step."""

    metrics: SystemMetrics = Field(..., description="Current normalised system metrics")
    incident_type: IncidentType = Field(..., description="Active incident in this episode")
    step: int = Field(default=0, description="Current step within the episode")
    explanation: str = Field(default="", description="Post-hoc explanation for last agent action")
