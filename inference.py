"""
NOC Agent — Inference Script
=============================
Runs all 3 incident tasks sequentially, one Docker container per task.
Prints a final summary table after all tasks complete.

Required environment variables:
    API_BASE_URL   LLM API endpoint  (e.g. https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier   (e.g. Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key for the above endpoint
    IMAGE_NAME     Docker image name  (e.g. noc_agent)

Optional:
    TASK_NAME      Run a single task instead of all 3
                   cpu_overload | memory_leak | network_congestion

Stdout format (mandatory per task):
    [START] task=<task_name> env=noc_agent model=<model_name>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY      = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
IMAGE_NAME   = os.getenv("IMAGE_NAME", "openenv-noc_agent")
BENCHMARK    = os.getenv("NOC_AGENT_BENCHMARK", "noc_agent")

# Set ENV_BASE_URL to override (e.g. for a remote server).
# Set USE_DOCKER=true to force Docker-based startup for local testing.
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://srini047-noc-agent.hf.space")
USE_DOCKER   = os.getenv("USE_DOCKER", "false").lower() in ("1", "true", "yes")

# If set, run only that task; otherwise defaults to network_congestion
_SINGLE_TASK: Optional[str] = os.getenv("NOC_AGENT_TASK", "network_congestion")

ALL_TASKS: list[str] = [
    "network_congestion",   # easy
    "memory_leak",          # medium
    "cpu_overload",         # hard
]

MAX_STEPS:               int   = 30
TEMPERATURE:             float = 0.2
MAX_TOKENS:              int   = 20
SUCCESS_SCORE_THRESHOLD: float = 0.5
MAX_TOTAL_REWARD:        float = 25.0

VALID_ACTIONS: list[str] = [
    "do_nothing",
    "restart_service",
    "throttle_cpu",
    "clear_cache",
    "reroute_traffic",
    "scale_up",
]

# ---------------------------------------------------------------------------
# Mandatory stdout logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an autonomous NOC (Network Operations Centre) engineer.
    You observe live system telemetry from a Linux server under an active incident.
    Your job: choose the single best remediation action to resolve the incident as quickly as possible.

    Available actions (output EXACTLY one, lowercase, underscore-separated):
      do_nothing        — wait and observe
      restart_service   — restart the degraded service process
      throttle_cpu      — reduce CPU allocation of runaway processes
      clear_cache       — flush application/OS cache to reclaim memory
      reroute_traffic   — redirect traffic to alternate network paths
      scale_up          — add compute/bandwidth capacity

    Rules:
      - Output ONLY the action name. No explanation, no punctuation, no other text.
      - Match actions to the dominant symptom:
          high CPU            → throttle_cpu, scale_up
          high memory / OOM   → clear_cache, restart_service
          high latency / loss → reroute_traffic, scale_up
          service down        → restart_service
      - Avoid do_nothing unless all metrics are near-healthy.
""").strip()


def _build_user_prompt(
    step: int,
    incident: str,
    metrics: dict,
    last_reward: float,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-5:]) if history else "  (none)"
    return textwrap.dedent(f"""\
        Step: {step}
        Incident type: {incident.replace('_', ' ').upper()}

        Current system metrics:
          CPU usage     : {metrics['cpu_usage']:.1%}
          Memory usage  : {metrics['memory_usage']:.1%}
          Latency       : {metrics.get('latency_ms', 0):.0f} ms
          Packet loss   : {metrics['packet_loss']:.1%}
          Service health: {metrics['service_healthy']:.1%}
          Error rate    : {metrics['error_rate']:.1%}

        Last reward: {last_reward:+.3f}
        Recent actions:
        {history_block}

        Choose your action:
    """).strip()


def _parse_action(raw: str) -> str:
    cleaned = raw.strip().lower().replace("-", "_").split()[0] if raw.strip() else ""
    if cleaned in VALID_ACTIONS:
        return cleaned
    for action in VALID_ACTIONS:
        if action.startswith(cleaned) or cleaned in action:
            return action
    print(f"[DEBUG] Unrecognised action {raw!r} — defaulting to do_nothing", flush=True)
    return "do_nothing"


def get_llm_action(
    client: OpenAI,
    step: int,
    incident: str,
    metrics: dict,
    last_reward: float,
    history: List[str],
) -> str:
    prompt = _build_user_prompt(step, incident, metrics, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return _parse_action(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return "do_nothing"

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(task: str, client: OpenAI) -> dict:
    """
    Run one full episode for the given task.
    Spins up a fresh Docker container, runs the agent, tears it down.
    Returns a result dict for the summary table.
    """
    from noc_agent.client import NocAgentEnvClient
    from noc_agent.models import ActionType, NOCAction

    rewards:     List[float]    = []
    history:     List[str]      = []
    steps_taken: int            = 0
    score:       float          = 0.0
    success:     bool           = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    if USE_DOCKER:
        env = await NocAgentEnvClient.from_docker_image(IMAGE_NAME)
    else:
        env = NocAgentEnvClient(base_url=ENV_BASE_URL)
        await env.connect()

    try:
        # Pin exact incident type via reset kwargs (server forwards to reset(incident_type=...))
        result = await env.reset(incident_type=task)
        incident_type: str = result.observation.incident_type

        obs = result.observation
        metrics = obs.metrics.model_dump()
        metrics["latency_ms"] = metrics.pop("latency", 0) * 500
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_str  = get_llm_action(client, step, incident_type, metrics, last_reward, history)
            action_type = ActionType(action_str)

            result      = await env.step(NOCAction(action_type=action_type))
            obs         = result.observation
            reward      = result.reward or 0.0
            done        = result.done

            metrics = obs.metrics.model_dump()
            metrics["latency_ms"] = metrics.pop("latency", 0) * 500
            last_reward = reward

            rewards.append(reward)
            steps_taken = step
            history.append(f"Step {step}: {action_str} → reward {reward:+.3f}")

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        raw_score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0
        score     = min(max(raw_score, 0.0), 1.0)
        success   = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error ({task}): {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task, "success": success, "steps": steps_taken, "score": score}

# ---------------------------------------------------------------------------
# Main — run all tasks sequentially, print summary
# ---------------------------------------------------------------------------

async def main() -> None:
    # Ensure noc_agent package is importable
    sys.path.insert(0, str(Path(__file__).parent))

    tasks = [_SINGLE_TASK] if _SINGLE_TASK else ALL_TASKS
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results: list[dict] = []
    for task in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"[INFO] Starting task: {task}", flush=True)
        print(f"{'='*60}", flush=True)
        result = await run_episode(task=task, client=client)
        results.append(result)

    # Summary table
    print(f"\n{'='*60}", flush=True)
    print("[SUMMARY]", flush=True)
    print(f"{'Task':<25} {'Success':<10} {'Steps':<8} {'Score'}", flush=True)
    print(f"{'-'*25} {'-'*10} {'-'*8} {'-'*6}", flush=True)
    for r in results:
        print(
            f"{r['task']:<25} {str(r['success']).lower():<10} {r['steps']:<8} {r['score']:.3f}",
            flush=True,
        )
    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n[SUMMARY] overall_score={overall:.3f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
