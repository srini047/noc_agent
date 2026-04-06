"""
ActionExplainer — post-hoc natural language explanations for agent decisions.

Explanations are derived from the *current observation* at decision time,
not from the model internals.  This is honest and transparent: we describe
what the system state shows, which is exactly what a human NOC engineer
would say when justifying the same action.
"""

from __future__ import annotations

from .models import ActionType, IncidentType, SystemMetrics


class ActionExplainer:
    """
    Maps (action, metrics, incident_type) → a human-readable explanation string.

    Each explanation surfaces the dominant signal that makes the action
    reasonable.  If the action is wrong/irrelevant, that is also noted —
    useful during the demo to show the agent has learned to avoid bad moves.
    """

    def explain(
        self,
        action: ActionType,
        metrics: SystemMetrics,
        incident_type: IncidentType,  # reserved for future incident-aware explanations
    ) -> str:
        if action == ActionType.DO_NOTHING:
            return self._do_nothing(metrics)
        if action == ActionType.RESTART_SERVICE:
            return self._restart_service(metrics)
        if action == ActionType.THROTTLE_CPU:
            return self._throttle_cpu(metrics)
        if action == ActionType.CLEAR_CACHE:
            return self._clear_cache(metrics)
        if action == ActionType.REROUTE_TRAFFIC:
            return self._reroute_traffic(metrics)
        if action == ActionType.SCALE_UP:
            return self._scale_up(metrics)
        return f"Agent applied action. System health: {metrics.health_score:.0%}."

    # ------------------------------------------------------------------
    # Per-action explanation builders
    # ------------------------------------------------------------------

    @staticmethod
    def _do_nothing(metrics: SystemMetrics) -> str:
        if metrics.health_score > 0.75:
            return "System metrics are within acceptable range. Monitoring — no intervention needed."
        return (
            f"Agent chose to wait. Health score is {metrics.health_score:.0%}. "
            "Observing for one step before acting."
        )

    @staticmethod
    def _restart_service(metrics: SystemMetrics) -> str:
        reasons: list[str] = []
        if metrics.service_healthy < 1.0:
            reasons.append(f"service is partially down ({metrics.service_healthy:.0%} health)")
        if metrics.error_rate > 0.35:
            reasons.append(f"error rate is critically high ({metrics.error_rate:.0%})")
        if not reasons:
            reasons.append("attempting to clear transient faults via process restart")
        return "Restarting service — " + "; ".join(reasons) + "."

    @staticmethod
    def _throttle_cpu(metrics: SystemMetrics) -> str:
        if metrics.cpu_usage > 0.80:
            return (
                f"CPU is saturated at {metrics.cpu_usage:.0%}. "
                "Throttling runaway process to free cycles and reduce latency."
            )
        return (
            f"CPU usage is {metrics.cpu_usage:.0%}. "
            "Throttling as a precaution — marginal effect expected."
        )

    @staticmethod
    def _clear_cache(metrics: SystemMetrics) -> str:
        if metrics.memory_usage > 0.75:
            return (
                f"Memory usage is {metrics.memory_usage:.0%}. "
                "Clearing application cache to reclaim heap space and prevent OOM."
            )
        return (
            f"Memory usage is {metrics.memory_usage:.0%}. "
            "Cache flush has limited benefit at this memory level."
        )

    @staticmethod
    def _reroute_traffic(metrics: SystemMetrics) -> str:
        reasons: list[str] = []
        if metrics.latency > 0.40:
            reasons.append(f"latency is {metrics.latency * 500:.0f} ms")
        if metrics.packet_loss > 0.10:
            reasons.append(f"packet loss is {metrics.packet_loss:.0%}")
        if not reasons:
            reasons.append("network metrics are within normal range — limited benefit expected")
        return "Rerouting traffic — " + "; ".join(reasons) + "."

    @staticmethod
    def _scale_up(metrics: SystemMetrics) -> str:
        if metrics.cpu_usage > 0.70 or metrics.memory_usage > 0.70:
            return (
                f"Resource pressure detected (CPU {metrics.cpu_usage:.0%}, "
                f"Mem {metrics.memory_usage:.0%}). Scaling up to distribute load."
            )
        return (
            "Scaling up to add capacity headroom. "
            "Effective but slow — better for sustained overload than spikes."
        )
