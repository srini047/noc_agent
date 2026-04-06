"""
Gradio demo for the NOC Incident Triage Agent.

Layout:
  Left  — controls + health bar + status
  Right — live metric charts + action log + post-episode analysis

Run locally:
    uv run demo
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import gradio as gr
import logfire
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO

try:
    from ..analysis import EpisodeAnalyser, EpisodeSummary
    from ..gym_env import NOCSystemEnv
    from ..models import IncidentType
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from noc_agent.analysis import EpisodeAnalyser, EpisodeSummary  # type: ignore[no-redef]
    from noc_agent.gym_env import NOCSystemEnv  # type: ignore[no-redef]
    from noc_agent.models import IncidentType  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.zip"
FINAL_MODEL_PATH = MODELS_DIR / "noc_ppo_final.zip"

INCIDENT_DISPLAY = {
    IncidentType.CPU_OVERLOAD: "CPU Overload",
    IncidentType.MEMORY_LEAK: "Memory Leak",
    IncidentType.NETWORK_CONGESTION: "Network Congestion",
}

_WARN_HEALTH = 0.55
_CRIT_HEALTH = 0.35

# ---------------------------------------------------------------------------
# Logfire
# ---------------------------------------------------------------------------

logfire.configure(service_name="noc-agent-demo")

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_model() -> PPO | None:
    for path in (BEST_MODEL_PATH, FINAL_MODEL_PATH):
        if path.exists():
            logfire.info("Loading model from {path}", path=str(path))
            return PPO.load(str(path))
    logfire.warn("No trained model found. Run `uv run train` first.")
    return None


def load_analyser() -> EpisodeAnalyser | None:
    try:
        return EpisodeAnalyser()
    except ValueError:
        logfire.warn("COHERE_API_KEY not set — post-episode analysis disabled.")
        return None


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

class DemoSession:
    """Holds all mutable state for a single Gradio session."""

    def __init__(self) -> None:
        self.env = NOCSystemEnv()
        self.model = load_model()
        self.analyser = load_analyser()
        self.obs: np.ndarray | None = None
        self.episode_reward: float = 0.0
        self.action_log: list[dict[str, Any]] = []
        self.metric_history: list[dict[str, float]] = []
        self.done: bool = True
        self._current_incident: IncidentType = IncidentType.CPU_OVERLOAD

    def reset(self, incident: str) -> None:
        self._current_incident = IncidentType(incident)
        self.env = NOCSystemEnv(incident_type=self._current_incident)
        obs, info = self.env.reset()
        self.obs = obs
        self.episode_reward = 0.0
        self.action_log = []
        self.metric_history = [info["metrics"]]
        self.done = False
        logfire.info("Episode started {incident}", incident=self._current_incident.value)

    def agent_step(self) -> None:
        if self.obs is None or self.done:
            return
        if self.model is None:
            action = self.env.action_space.sample()
        else:
            action, _ = self.model.predict(self.obs, deterministic=True)
            action = int(action)

        obs, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs
        self.episode_reward += reward
        self.done = terminated or truncated

        self.action_log.append({
            "step": info["step"],
            "action": info["action"],
            "explanation": info["explanation"],
            "health": info["health_score"],
            "reward": reward,
            "resolved": info["resolved"],
            "crashed": info["crashed"],
        })
        self.metric_history.append(info["metrics"])

        logfire.info(
            "Agent step {step} — {action}",
            step=info["step"],
            action=info["action"],
            health_score=round(info["health_score"], 3),
            reward=round(reward, 3),
            resolved=info["resolved"],
        )

    def generate_analysis(self) -> str:
        if self.analyser is None:
            return "COHERE_API_KEY not configured — analysis unavailable."
        if not self.action_log:
            return "No episode data to analyse."

        last = self.action_log[-1]
        summary = EpisodeSummary(
            incident_type=self._current_incident,
            resolved=last.get("resolved", False),
            crashed=last.get("crashed", False),
            total_steps=last["step"],
            total_reward=self.episode_reward,
            action_log=self.action_log,
            metric_history=self.metric_history,
        )
        with logfire.span("Cohere post-episode analysis {incident}", incident=self._current_incident.value):
            return self.analyser.analyse(summary)


# ---------------------------------------------------------------------------
# HTML / chart renderers
# ---------------------------------------------------------------------------

def _build_metrics_chart(history: list[dict[str, float]]) -> go.Figure:
    steps = list(range(len(history)))
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=["CPU Usage %", "Memory Usage %", "Latency ms",
                        "Packet Loss %", "Service Health %", "Error Rate %"],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )
    series = [
        ("cpu_usage",       1, 1, "#ef4444"),
        ("memory_usage",    1, 2, "#f97316"),
        ("latency_ms",      1, 3, "#eab308"),
        ("packet_loss",     2, 1, "#8b5cf6"),
        ("service_healthy", 2, 2, "#22c55e"),
        ("error_rate",      2, 3, "#ec4899"),
    ]
    for key, row, col, colour in series:
        values = [h[key] for h in history]
        if key != "latency_ms":
            values = [v * 100 for v in values]
        fig.add_trace(
            go.Scatter(x=steps, y=values, mode="lines+markers",
                       marker=dict(size=4), line=dict(color=colour, width=2),
                       showlegend=False),
            row=row, col=col,
        )
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    return fig


def _action_log_html(log: list[dict[str, Any]]) -> str:
    if not log:
        return "<p style='color:#888;padding:12px'>No actions yet.</p>"
    rows = []
    for entry in reversed(log[-10:]):
        health = entry["health"]
        colour = "#22c55e" if health > _WARN_HEALTH else ("#f97316" if health > _CRIT_HEALTH else "#ef4444")
        badge = f"<span style='background:{colour};color:#fff;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600'>{health:.0%}</span>"
        status = ""
        if entry["resolved"]:
            status = "<span style='color:#22c55e;font-weight:700;margin-left:6px'>✓ RESOLVED</span>"
        elif entry["crashed"]:
            status = "<span style='color:#ef4444;font-weight:700;margin-left:6px'>✗ CRASHED</span>"
        action_name = entry["action"].replace("_", " ").upper() if entry["action"] else "—"
        rows.append(f"""
        <div style='border-left:3px solid {colour};padding:8px 12px;margin-bottom:6px;
                    background:rgba(0,0,0,0.02);border-radius:0 6px 6px 0'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:3px'>
                <strong style='font-size:0.84rem'>Step {entry['step']} — {action_name}</strong>
                <div>{badge}{status}</div>
            </div>
            <div style='color:#555;font-size:0.81rem'>{entry['explanation']}</div>
        </div>""")
    return "".join(rows)


def _health_bar_html(health: float) -> str:
    colour = "#22c55e" if health > _WARN_HEALTH else ("#f97316" if health > _CRIT_HEALTH else "#ef4444")
    pct = health * 100
    return f"""
    <div style='padding:6px 0'>
        <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
            <span style='font-weight:600;font-size:0.9rem'>System Health</span>
            <span style='font-weight:700;color:{colour};font-size:1.1rem'>{pct:.0f}%</span>
        </div>
        <div style='background:#e5e7eb;border-radius:999px;height:10px'>
            <div style='background:{colour};width:{pct}%;height:10px;border-radius:999px;transition:width 0.3s'></div>
        </div>
    </div>"""


def _analysis_html(text: str) -> str:
    """Render the plain-text Cohere report in a styled card."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = escaped.strip().split("\n")
    formatted = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            formatted.append("<br>")
        # Section headers (ALL CAPS or ending with colon)
        elif stripped.isupper() or (stripped.endswith(":") and len(stripped) < 60):
            formatted.append(f"<div style='font-weight:700;color:#1e40af;margin-top:10px;font-size:0.88rem;letter-spacing:0.05em'>{stripped}</div>")
        else:
            formatted.append(f"<div style='margin:2px 0;font-size:0.85rem;line-height:1.55;color:#1f2937'>{line}</div>")
    body = "\n".join(formatted)
    return f"""
    <div style='background:#f0f7ff;border:1px solid #bfdbfe;border-radius:10px;
                padding:16px 20px;margin-top:8px;font-family:monospace'>
        <div style='font-weight:700;color:#1e3a8a;margin-bottom:10px;font-size:0.9rem'>
            POST-INCIDENT REPORT &nbsp;·&nbsp; Cohere {_MODEL_TAG}
        </div>
        {body}
    </div>"""

_MODEL_TAG = "command-a-03-2025"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    session = DemoSession()

    def _refresh_outputs():
        if not session.metric_history:
            return (
                go.Figure(),
                "<p style='color:#888;padding:12px'>Start an episode to begin.</p>",
                _health_bar_html(1.0),
                "Ready", "0.00",
            )
        last = session.action_log[-1] if session.action_log else None
        health_val = last["health"] if last else 1.0
        status = "Done ✓" if session.done else f"Step {len(session.action_log)}"
        return (
            _build_metrics_chart(session.metric_history),
            _action_log_html(session.action_log),
            _health_bar_html(health_val),
            status,
            f"{session.episode_reward:.2f}",
        )

    def on_start(incident: str):
        session.reset(incident)
        chart, log_html, health_html, status, reward = _refresh_outputs()
        return (chart, log_html, health_html, status, reward,
                gr.update(interactive=True), gr.update(interactive=True),
                "", gr.update(visible=False))

    def on_step():
        if not session.done:
            session.agent_step()
        chart, log_html, health_html, status, reward = _refresh_outputs()
        active = not session.done
        analyse_visible = session.done and bool(session.action_log)
        return (chart, log_html, health_html, status, reward,
                gr.update(interactive=active), gr.update(interactive=active),
                "", gr.update(visible=analyse_visible))

    def on_auto_run(incident: str):
        session.reset(incident)
        while not session.done:
            session.agent_step()
            time.sleep(0.03)
        chart, log_html, health_html, status, reward = _refresh_outputs()
        return (chart, log_html, health_html, status, reward,
                gr.update(interactive=False), gr.update(interactive=False),
                "", gr.update(visible=True))

    def on_analyse():
        report_text = session.generate_analysis()
        return _analysis_html(report_text)

    # ---- Layout ----
    with gr.Blocks(
        title="NOC Agent — Autonomous Incident Triage",
        theme=gr.themes.Soft(primary_hue="blue"),
        css=".gradio-container { max-width: 1200px !important }",
    ) as demo:
        gr.Markdown(
            "# NOC Agent — Autonomous Incident Triage\n"
            "RL agent (PPO) that triages simulated Linux system incidents. "
            "Select an incident, run the agent, then generate a Cohere-powered incident report."
        )

        with gr.Row():
            # ---- Left: controls ----
            with gr.Column(scale=1, min_width=260):
                incident_radio = gr.Radio(
                    choices=[(v, k.value) for k, v in INCIDENT_DISPLAY.items()],
                    value=IncidentType.CPU_OVERLOAD.value,
                    label="Incident Type",
                )
                with gr.Row():
                    btn_start = gr.Button("▶ Start", variant="primary")
                    btn_step  = gr.Button("Step →", interactive=False)
                btn_auto = gr.Button("⚡ Auto Run", variant="secondary")

                health_display = gr.HTML(_health_bar_html(1.0))

                with gr.Row():
                    status_box = gr.Textbox(label="Status",            value="Ready", interactive=False, scale=2)
                    reward_box = gr.Textbox(label="Cumul. Reward",     value="0.00",  interactive=False, scale=1)

                gr.Markdown(
                    "✅ **Model loaded**" if session.model else
                    "⚠️ **No model** — random policy. Run `uv run train` first."
                )
                gr.Markdown(
                    "✅ **Cohere ready**" if session.analyser else
                    "⚠️ **No COHERE_API_KEY** — analysis disabled."
                )

            # ---- Right: charts + log + analysis ----
            with gr.Column(scale=2):
                chart      = gr.Plot(label="System Metrics Over Time")
                action_log = gr.HTML()
                btn_analyse = gr.Button("📋 Generate Incident Report", variant="primary", visible=False)
                analysis   = gr.HTML()

        # ---- event wiring ----
        step_outputs = [chart, action_log, health_display, status_box, reward_box,
                        btn_step, btn_auto, analysis, btn_analyse]

        btn_start.click(fn=on_start,    inputs=[incident_radio], outputs=step_outputs)
        btn_step.click( fn=on_step,     inputs=[],               outputs=step_outputs)
        btn_auto.click( fn=on_auto_run, inputs=[incident_radio], outputs=step_outputs)
        btn_analyse.click(fn=on_analyse, inputs=[], outputs=[analysis])

    return demo


def main() -> None:
    logfire.info("Starting NOC Agent demo")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
