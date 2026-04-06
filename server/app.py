"""
FastAPI application for the NOC Agent Environment.

Exposes the NocAgentEnvironment over HTTP and WebSocket endpoints
using the openenv.core HTTP server factory.

Endpoints:
    POST /reset   — start a new episode
    POST /step    — send an action, receive observation
    GET  /state   — current episode state
    GET  /schema  — action / observation JSON schemas
    WS   /ws      — persistent WebSocket session

Usage (development):
    uv run --project . server
    # or:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install with:\n    uv sync"
    ) from e

try:
    from ..models import NOCAction, NOCObservation
    from .noc_agent_environment import NocAgentEnvironment
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from noc_agent.models import NOCAction, NOCObservation  # type: ignore[no-redef]
    from noc_agent.server.noc_agent_environment import NocAgentEnvironment  # type: ignore[no-redef]


app = create_app(
    NocAgentEnvironment,
    NOCAction,
    NOCObservation,
    env_name="noc_agent",
    max_concurrent_envs=4,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for ``uv run --project . server``."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
