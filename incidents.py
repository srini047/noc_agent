"""
Incident definitions and state-transition profiles.

Each IncidentProfile describes:
  - The initial metric values when the incident starts
  - How metrics drift (worsen) per step if untreated
  - Which actions reduce which metrics and by how much

Open/Closed principle: add new incidents by creating a new IncidentProfile
and registering it in INCIDENT_REGISTRY — no existing logic needs to change.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .models import ActionType, IncidentType


class MetricDelta(BaseModel):
    """Per-step change applied to a single metric (positive = increase = worse)."""

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    latency: float = 0.0
    packet_loss: float = 0.0
    service_healthy: float = 0.0
    error_rate: float = 0.0


class ActionEffect(BaseModel):
    """
    Net metric delta when a specific action is applied to a specific incident.

    Negative values mean the action *reduces* (improves) that metric.
    Zero means the action has no effect on that metric.
    Positive values mean the action makes that metric worse.
    """

    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    latency: float = 0.0
    packet_loss: float = 0.0
    service_healthy: float = 0.0
    error_rate: float = 0.0


class IncidentProfile(BaseModel):
    """
    Complete specification for one incident type.

    - initial_metrics: starting values at reset
    - drift_per_step: how each metric worsens each step when nothing helpful is done
    - action_effects: per-action metric deltas (applied *instead of* raw drift for that step)
    - max_steps: episode truncation if unresolved
    """

    incident_type: IncidentType
    description: str

    # Initial state
    initial_cpu: float = Field(..., ge=0.0, le=1.0)
    initial_memory: float = Field(..., ge=0.0, le=1.0)
    initial_latency: float = Field(..., ge=0.0, le=1.0)
    initial_packet_loss: float = Field(..., ge=0.0, le=1.0)
    initial_service_healthy: float = Field(default=1.0, ge=0.0, le=1.0)
    initial_error_rate: float = Field(..., ge=0.0, le=1.0)

    # Worsening drift applied every step regardless of action
    drift: MetricDelta

    # Action → metric change mapping
    action_effects: dict[ActionType, ActionEffect]

    max_steps: int = 60


def _cpu_overload_profile() -> IncidentProfile:
    return IncidentProfile(
        incident_type=IncidentType.CPU_OVERLOAD,
        description="A runaway process is consuming CPU. Services are degrading under load.",
        initial_cpu=0.85,
        initial_memory=0.52,
        initial_latency=0.28,
        initial_packet_loss=0.04,
        initial_service_healthy=1.0,
        initial_error_rate=0.12,
        drift=MetricDelta(cpu_usage=0.018, memory_usage=0.004, error_rate=0.008, latency=0.008),
        action_effects={
            ActionType.DO_NOTHING: ActionEffect(),
            # Throttling directly cuts CPU and helps latency
            ActionType.THROTTLE_CPU: ActionEffect(cpu_usage=-0.18, latency=-0.06, error_rate=-0.04),
            # Scaling up adds capacity — slower recovery but sustainable
            ActionType.SCALE_UP: ActionEffect(cpu_usage=-0.12, latency=-0.04, error_rate=-0.03),
            # Cache clear has a mild side effect on memory but nothing meaningful for CPU
            ActionType.CLEAR_CACHE: ActionEffect(memory_usage=-0.04),
            # Restarting service briefly worsens CPU (restart overhead) then helps error rate
            ActionType.RESTART_SERVICE: ActionEffect(cpu_usage=0.05, error_rate=-0.06, service_healthy=0.0),
            # Rerouting traffic does not help a CPU-bound issue
            ActionType.REROUTE_TRAFFIC: ActionEffect(),
        },
    )


def _memory_leak_profile() -> IncidentProfile:
    return IncidentProfile(
        incident_type=IncidentType.MEMORY_LEAK,
        description="A service has a memory leak. RAM is steadily climbing toward OOM.",
        initial_cpu=0.55,
        initial_memory=0.82,
        initial_latency=0.16,
        initial_packet_loss=0.02,
        initial_service_healthy=1.0,
        initial_error_rate=0.10,
        drift=MetricDelta(memory_usage=0.025, cpu_usage=0.003, error_rate=0.007),
        action_effects={
            ActionType.DO_NOTHING: ActionEffect(),
            # Clearing cache is the primary fix — drops memory significantly
            ActionType.CLEAR_CACHE: ActionEffect(memory_usage=-0.22, error_rate=-0.05),
            # Restarting the leaking service reclaims all memory (larger drop, slower)
            ActionType.RESTART_SERVICE: ActionEffect(memory_usage=-0.30, error_rate=-0.08, service_healthy=0.0),
            # Throttling CPU has minor indirect benefit
            ActionType.THROTTLE_CPU: ActionEffect(cpu_usage=-0.05),
            # Scale-up temporarily reduces memory pressure per instance
            ActionType.SCALE_UP: ActionEffect(memory_usage=-0.06, cpu_usage=-0.04),
            # Rerouting traffic is irrelevant to a memory problem
            ActionType.REROUTE_TRAFFIC: ActionEffect(),
        },
    )


def _network_congestion_profile() -> IncidentProfile:
    return IncidentProfile(
        incident_type=IncidentType.NETWORK_CONGESTION,
        description="Network links are saturated. Latency is high and packets are being dropped.",
        initial_cpu=0.48,
        initial_memory=0.44,
        initial_latency=0.60,
        initial_packet_loss=0.25,
        initial_service_healthy=1.0,
        initial_error_rate=0.20,
        drift=MetricDelta(latency=0.035, packet_loss=0.012, error_rate=0.009),
        action_effects={
            ActionType.DO_NOTHING: ActionEffect(),
            # Rerouting is the primary fix
            ActionType.REROUTE_TRAFFIC: ActionEffect(latency=-0.12, packet_loss=-0.06, error_rate=-0.05),
            # Scaling up adds bandwidth capacity
            ActionType.SCALE_UP: ActionEffect(latency=-0.06, packet_loss=-0.03, error_rate=-0.03),
            # Throttling CPU is irrelevant for a network problem
            ActionType.THROTTLE_CPU: ActionEffect(),
            # Clear cache is irrelevant
            ActionType.CLEAR_CACHE: ActionEffect(),
            # Restarting service temporarily drops connections (worsens packet loss briefly)
            ActionType.RESTART_SERVICE: ActionEffect(packet_loss=0.04, error_rate=0.05, service_healthy=0.0),
        },
    )


# ---------------------------------------------------------------------------
# Registry — the single source of truth for all incident profiles
# ---------------------------------------------------------------------------

INCIDENT_REGISTRY: dict[IncidentType, IncidentProfile] = {
    IncidentType.CPU_OVERLOAD: _cpu_overload_profile(),
    IncidentType.MEMORY_LEAK: _memory_leak_profile(),
    IncidentType.NETWORK_CONGESTION: _network_congestion_profile(),
}


def get_profile(incident_type: IncidentType) -> IncidentProfile:
    """Return the profile for the given incident type."""
    return INCIDENT_REGISTRY[incident_type]
