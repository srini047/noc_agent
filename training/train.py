"""
PPO training script for the NOC Incident Triage agent.

Observability:
  - logfire : structured traces for each training run (configure LOGFIRE_TOKEN)
  - wandb   : reward curves, episode lengths, loss metrics (configure WANDB_API_KEY)

Usage:
    uv run train
    uv run train --timesteps 200000 --incident cpu_overload
    uv run train --no-wandb  (skip wandb logging)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import logfire
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

try:
    from ..gym_env import NOCSystemEnv
    from ..models import IncidentType
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from noc_agent.gym_env import NOCSystemEnv  # type: ignore[no-redef]
    from noc_agent.models import IncidentType  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).parent.parent / "models"
TOTAL_TIMESTEPS = 200_000
N_ENVS = 4  # parallel rollout workers
EVAL_FREQ = 10_000
EVAL_EPISODES = 20


# ---------------------------------------------------------------------------
# Wandb callback
# ---------------------------------------------------------------------------

class WandbLoggingCallback(BaseCallback):
    """Logs SB3 training metrics to wandb at each rollout end."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep = info["episode"]
                self._episode_rewards.append(ep["r"])
                self._episode_lengths.append(ep["l"])

        return True

    def _on_rollout_end(self) -> None:
        if not self._episode_rewards:
            return
        wandb.log(
            {
                "rollout/ep_rew_mean": np.mean(self._episode_rewards),
                "rollout/ep_rew_min": np.min(self._episode_rewards),
                "rollout/ep_rew_max": np.max(self._episode_rewards),
                "rollout/ep_len_mean": np.mean(self._episode_lengths),
                "train/timesteps": self.num_timesteps,
            }
        )
        self._episode_rewards.clear()
        self._episode_lengths.clear()


# ---------------------------------------------------------------------------
# Logfire callback
# ---------------------------------------------------------------------------

class LogfireCallback(BaseCallback):
    """Emits a logfire span at the end of each rollout for structured tracing."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._rollout_start: float = time.time()

    def _on_rollout_start(self) -> None:
        self._rollout_start = time.time()

    def _on_rollout_end(self) -> None:
        duration_ms = (time.time() - self._rollout_start) * 1000
        logfire.info(
            "PPO rollout complete {timesteps} steps",
            timesteps=self.num_timesteps,
            rollout_duration_ms=round(duration_ms, 1),
        )

    def _on_step(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    total_timesteps: int = TOTAL_TIMESTEPS,
    incident_type: IncidentType | None = None,
    use_wandb: bool = True,
    seed: int = 42,
) -> Path:
    """
    Train a PPO agent on the NOC environment.

    Returns the path to the saved model.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Logfire setup ----
    logfire.configure(service_name="noc-agent-training")
    logfire.info(
        "Training started",
        total_timesteps=total_timesteps,
        incident_type=incident_type.value if incident_type else "random",
        seed=seed,
    )

    # ---- Wandb setup ----
    run = None
    if use_wandb:
        run = wandb.init(
            project="noc-agent",
            config={
                "total_timesteps": total_timesteps,
                "incident_type": incident_type.value if incident_type else "random",
                "n_envs": N_ENVS,
                "algorithm": "PPO",
                "seed": seed,
            },
            sync_tensorboard=False,
        )

    # ---- Environment ----
    def make_env() -> Monitor:
        env = NOCSystemEnv(incident_type=incident_type, seed=seed)
        return Monitor(env)

    vec_env = make_vec_env(make_env, n_envs=N_ENVS)
    eval_env = Monitor(NOCSystemEnv(incident_type=incident_type, seed=seed + 1))

    # ---- Model ----
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    # ---- Callbacks ----
    callbacks: list[BaseCallback] = [LogfireCallback()]
    if use_wandb and run is not None:
        callbacks.append(WandbLoggingCallback())

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR),
        log_path=str(MODELS_DIR / "logs"),
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    # ---- Train ----
    with logfire.span("PPO training {total_timesteps} timesteps", total_timesteps=total_timesteps):
        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)

    # ---- Save ----
    model_path = MODELS_DIR / "noc_ppo_final"
    model.save(str(model_path))
    logfire.info("Model saved {path}", path=str(model_path))

    if run is not None:
        wandb.save(str(model_path) + ".zip")
        run.finish()

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NOC PPO agent")
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS})",
    )
    parser.add_argument(
        "--incident",
        choices=[i.value for i in IncidentType],
        default=None,
        help="Fix to a specific incident type (default: random per episode)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    incident = IncidentType(args.incident) if args.incident else None
    model_path = train(
        total_timesteps=args.timesteps,
        incident_type=incident,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )
    print(f"\nTraining complete. Model saved to: {model_path}.zip")


if __name__ == "__main__":
    main()
