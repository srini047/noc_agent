---
title: Noc Agent Environment Server
emoji: 🏉
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# NOC Agent Environment

An OpenEnv environment that simulates a Linux server under an active incident. An LLM-based agent acts as an autonomous NOC (Network Operations Centre) engineer, observing live system telemetry and choosing remediation actions to resolve the incident as quickly as possible.

## What It Does

The environment simulates three incident types, each with its own drift rate and action-response characteristics:

| Incident | Difficulty | Primary symptom |
|---|---|---|
| `network_congestion` | Easy | High latency and packet loss |
| `memory_leak` | Medium | RAM climbing toward OOM |
| `cpu_overload` | Hard | Runaway process saturating CPU |

At each step the agent receives a `NOCObservation` with 6 normalised system metrics and must choose one of 6 discrete actions:

| Action | Best used for |
|---|---|
| `do_nothing` | When metrics are near-healthy |
| `throttle_cpu` | High CPU usage |
| `scale_up` | Capacity pressure (CPU or network) |
| `clear_cache` | High memory / cache pressure |
| `restart_service` | Service down or memory leak |
| `reroute_traffic` | High latency / packet loss |

An episode ends when the incident is **resolved** (all metrics below healthy thresholds for 3 consecutive steps), the system **crashes** (any metric exceeds the critical threshold), or the episode is **truncated** at `max_steps`.

## Setup

```bash
uv sync
```

This installs all dependencies from the lockfile into an isolated virtual environment. No manual `venv` or `pip install` needed.

## Quick Start

### Running the LLM Agent

The simplest way to evaluate an LLM against all three tasks:

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_KEY="<your-api-key>"
export IMAGE_NAME="openenv-noc_agent"

# Build the Docker image first (see below)
docker build -t openenv-noc_agent -f server/Dockerfile .

# Run all three tasks sequentially
uv run python -m noc_agent.inference
```

To run a single task:
```bash
NOC_AGENT_TASK=cpu_overload uv run python -m noc_agent.inference
```

The script prints structured logs per step and a final summary table:
```
[START] task=cpu_overload env=noc_agent model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=throttle_cpu reward=1.23 done=false error=null
...
[END]   success=true steps=12 score=0.742 rewards=1.23,0.95,...

[SUMMARY]
Task                      Success    Steps    Score
------------------------- ---------- -------- ------
network_congestion        true       8        0.810
memory_leak               true       14       0.623
cpu_overload              false      30       0.312

[SUMMARY] overall_score=0.582
```

### Using the Python Client Directly

```python
import asyncio
from noc_agent.client import NocAgentEnvClient
from noc_agent.models import ActionType, NOCAction

async def main():
    async with await NocAgentEnvClient.from_docker_image("openenv-noc_agent") as env:
        result = await env.reset(incident_type="cpu_overload")
        print(f"Incident: {result.observation.incident_type}")
        print(f"CPU: {result.observation.metrics.cpu_usage:.1%}")

        result = await env.step(NOCAction(action_type=ActionType.THROTTLE_CPU))
        print(f"Reward: {result.reward:.3f}")
        print(f"Explanation: {result.observation.explanation}")
        print(f"Done: {result.done}")

asyncio.run(main())
```

### RL Training with Gymnasium / Stable-Baselines3

The environment exposes a standard `gymnasium.Env` interface for training RL policies without the HTTP server:

```python
from noc_agent.gym_env import NOCSystemEnv
from stable_baselines3 import PPO

env = NOCSystemEnv()  # random incident each episode
# env = NOCSystemEnv(incident_type=IncidentType.CPU_OVERLOAD)  # fixed incident

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

Observation space: `Box(6,)` — normalised metrics `[cpu, memory, latency, packet_loss, service_healthy, error_rate]`  
Action space: `Discrete(6)` — indexed by `ACTION_INDEX` order

## Building the Docker Image

```bash
docker build -t openenv-noc_agent -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or with options
openenv push --namespace my-org --private
```

The `openenv push` command validates the environment, prepares a Hugging Face Docker space build, and uploads it. After deployment, your space exposes:

- **Web Interface** at `/web` — interactive UI for exploring the environment
- **API Documentation** at `/docs` — full OpenAPI/Swagger interface
- **Health Check** at `/health` — container health monitoring
- **WebSocket** at `/ws` — persistent session endpoint

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format `username/repo-name`
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

## Environment Details

### Observation (`NOCObservation`)

| Field | Type | Description |
|---|---|---|
| `metrics` | `SystemMetrics` | Current normalised system metrics |
| `incident_type` | `IncidentType` | Active incident in this episode |
| `step` | `int` | Current step within the episode |
| `explanation` | `str` | Post-hoc explanation of the last action's effect |
| `done` | `bool` | Whether the episode has ended |
| `reward` | `float` | Reward received for the last action |

### SystemMetrics

All values are normalised to `[0.0, 1.0]`. Higher values indicate more stress (except `service_healthy`).

| Field | Description | Healthy threshold |
|---|---|---|
| `cpu_usage` | CPU utilisation | < 0.65 |
| `memory_usage` | RAM utilisation | < 0.65 |
| `latency` | Network latency (normalised over 500 ms) | < 0.20 |
| `packet_loss` | Fraction of packets dropped | < 0.05 |
| `service_healthy` | 1.0 = healthy, 0.0 = down | ≥ 1.0 |
| `error_rate` | Fraction of requests returning errors | < 0.10 |

The aggregate `health_score` is a weighted sum:  
`1.0 − (cpu×0.25 + memory×0.25 + latency×0.20 + packet_loss×0.15 + (1−service_healthy)×0.10 + error_rate×0.05)`

### Reward

The reward at each step combines:
- **Health delta** — reward for improving overall health (`Δhealth × 5.0`)
- **Health bonus** — absolute nudge for staying healthy (`health × 0.5`)
- **Step penalty** — encourages faster resolution (`−0.10` per step)
- **Resolution bonus** — `+15.0` on successful resolution
- **Crash penalty** — `−10.0` if metrics exceed critical thresholds
- **Ineffective action penalty** — `−0.30` for actions with no measurable positive effect

## Project Structure

```
noc_agent/
├── __init__.py                   # Module exports
├── README.md                     # This file
├── openenv.yaml                  # OpenEnv manifest
├── pyproject.toml                # Project metadata and dependencies
├── models.py                     # Pydantic models: NOCAction, NOCObservation, SystemMetrics
├── simulator.py                  # Core state machine: incident drift + action effects
├── incidents.py                  # Incident profiles (initial state, drift, action mappings)
├── gym_env.py                    # Gymnasium-compatible env for RL training
├── explainability.py             # Natural-language explanations for agent actions
├── inference.py                  # LLM agent runner (OpenAI-compatible API)
├── client.py                     # OpenEnv WebSocket client
└── server/
    ├── noc_agent_environment.py  # OpenEnv server environment (wraps SystemSimulator)
    ├── app.py                    # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile                # Container image definition
```

## Development

### Test the environment logic directly

```bash
uv run python -m noc_agent.simulator
```

### Run the server locally

```bash
uv run server
```

Or with auto-reload during development:

```bash
uv run uvicorn server.app:app --reload
```

### Other entry points

```bash
uv run train   # Run RL training
uv run demo    # Launch the Gradio demo UI
```

### Add / remove dependencies

```bash
uv add <package>     # Add a runtime dependency
uv add --dev <pkg>   # Add a dev-only dependency
uv remove <package>  # Remove a dependency
```
