"""
EpisodeAnalyser — post-episode incident analysis using Cohere.

Produces a NOC-style incident report grounded in quantitative episode data.
The LLM's job here is synthesis and narrative, not reasoning about model internals.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cohere
from dotenv import load_dotenv

from .models import IncidentType

load_dotenv()

_MODEL = "command-a-03-2025"

_SYSTEM_PROMPT = """\
You are a Senior Site Reliability Engineer writing a post-incident report.
You will receive structured telemetry from an automated remediation run.
Write a concise, stat-heavy incident report in the following format — do not deviate from the structure.
Use plain text only (no markdown, no bullet symbols, no headers with #).
Be precise. Use exact numbers from the data provided. Do not infer or speculate.\
"""

_USER_TEMPLATE = """\
INCIDENT REPORT DATA
====================
Incident Type    : {incident_type}
Outcome          : {outcome}
Total Steps      : {total_steps}
Steps to Resolve : {steps_to_resolve}
Total Reward     : {total_reward:.2f}

METRIC TRAJECTORY
-----------------
               Start    Peak(worst)   End
CPU Usage   :  {cpu_start:.1%}     {cpu_peak:.1%}       {cpu_end:.1%}
Memory      :  {mem_start:.1%}     {mem_peak:.1%}       {mem_end:.1%}
Latency     :  {lat_start:.0f}ms   {lat_peak:.0f}ms     {lat_end:.0f}ms
Packet Loss :  {pkt_start:.1%}     {pkt_peak:.1%}       {pkt_end:.1%}
Error Rate  :  {err_start:.1%}     {err_peak:.1%}       {err_end:.1%}
Health Score:  {health_start:.1%}  {health_min:.1%}     {health_end:.1%}

ACTION BREAKDOWN
----------------
{action_table}

KEY INFLECTION POINTS
---------------------
{inflection_points}

Write the incident report now. Use this structure exactly:
1. INCIDENT SUMMARY (2 sentences max: what happened, outcome)
2. IMPACT ANALYSIS (quantify peak degradation and duration)
3. REMEDIATION TIMELINE (what the agent did, in order, with step numbers)
4. EFFECTIVENESS ASSESSMENT (which actions worked, which were wasted)
5. MTTR (mean time to resolve in steps; interpret what this means operationally)
"""


@dataclass
class EpisodeSummary:
    """Structured episode data passed to the analyser."""

    incident_type: IncidentType
    resolved: bool
    crashed: bool
    total_steps: int
    total_reward: float
    action_log: list[dict]      # list of {step, action, health, reward, resolved, crashed}
    metric_history: list[dict]  # list of {cpu_usage, memory_usage, latency_ms, packet_loss, service_healthy, error_rate}


class EpisodeAnalyser:
    """
    Calls Cohere to generate a stats-grounded post-incident report.

    Usage::

        summary = EpisodeSummary(...)
        analyser = EpisodeAnalyser()
        report = analyser.analyse(summary)
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY not set — add it to .env or pass api_key=")
        self._client = cohere.ClientV2(api_key=key)

    def analyse(self, summary: EpisodeSummary) -> str:
        """Generate the incident report. Returns plain text."""
        prompt = self._build_prompt(summary)
        response = self._client.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,   # low temp: stat-heavy, not creative
            max_tokens=600,
        )
        return response.message.content[0].text.strip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(s: EpisodeSummary) -> str:
        h = s.metric_history
        if not h:
            return "No episode data available."

        # Metric trajectory
        first, last = h[0], h[-1]

        cpu_vals   = [m["cpu_usage"]    for m in h]
        mem_vals   = [m["memory_usage"] for m in h]
        lat_vals   = [m["latency_ms"]   for m in h]
        pkt_vals   = [m["packet_loss"]  for m in h]
        err_vals   = [m["error_rate"]   for m in h]
        health_vals = [
            1.0 - (
                m["cpu_usage"] * 0.25
                + m["memory_usage"] * 0.25
                + (m["latency_ms"] / 500) * 0.20
                + m["packet_loss"] * 0.15
                + m["error_rate"] * 0.05
            )
            for m in h
        ]

        # Action breakdown: count + avg reward per action
        action_stats: dict[str, list[float]] = {}
        steps_to_resolve = s.total_steps  # default = never resolved
        for entry in s.action_log:
            name = entry["action"] or "do_nothing"
            action_stats.setdefault(name, []).append(entry["reward"])
            if entry.get("resolved") and steps_to_resolve == s.total_steps:
                steps_to_resolve = entry["step"]

        action_lines = []
        for action, rewards in sorted(action_stats.items(), key=lambda x: -len(x[1])):
            action_lines.append(
                f"  {action:<20} count={len(rewards):>3}  avg_reward={sum(rewards)/len(rewards):+.3f}"
            )
        action_table = "\n".join(action_lines) if action_lines else "  (no actions recorded)"

        # Key inflection points: steps where health changed direction significantly
        inflections: list[str] = []
        for i in range(1, min(len(s.action_log), len(health_vals))):
            delta = health_vals[i] - health_vals[i - 1]
            if abs(delta) >= 0.06:
                direction = "improved" if delta > 0 else "degraded"
                entry = s.action_log[i - 1]
                inflections.append(
                    f"  Step {entry['step']:>3}: health {direction} by {abs(delta):.1%} after {entry['action']}"
                )
        if not inflections:
            inflections = ["  No significant inflection points detected"]

        outcome = (
            "RESOLVED" if s.resolved
            else ("CRASHED (SLA breach)" if s.crashed else "TRUNCATED (max steps reached)")
        )

        return _USER_TEMPLATE.format(
            incident_type=s.incident_type.value.replace("_", " ").upper(),
            outcome=outcome,
            total_steps=s.total_steps,
            steps_to_resolve=steps_to_resolve if s.resolved else "N/A",
            total_reward=s.total_reward,
            cpu_start=first["cpu_usage"],   cpu_peak=max(cpu_vals),   cpu_end=last["cpu_usage"],
            mem_start=first["memory_usage"],mem_peak=max(mem_vals),   mem_end=last["memory_usage"],
            lat_start=first["latency_ms"],  lat_peak=max(lat_vals),   lat_end=last["latency_ms"],
            pkt_start=first["packet_loss"], pkt_peak=max(pkt_vals),   pkt_end=last["packet_loss"],
            err_start=first["error_rate"],  err_peak=max(err_vals),   err_end=last["error_rate"],
            health_start=health_vals[0],    health_min=min(health_vals), health_end=health_vals[-1],
            action_table=action_table,
            inflection_points="\n".join(inflections),
        )
