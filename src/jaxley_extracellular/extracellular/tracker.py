"""Experiment tracking protocol and implementations.

``NullTracker`` is always available (zero dependencies).
``MLflowTracker`` wraps mlflow as a pure HTTP client pointing at a
running MLflow tracking server.  Start one with::

    mlflow server --backend-store-uri sqlite:///results/tracking.db \\
                  --default-artifact-root ./results/mlartifacts \\
                  --host 127.0.0.1 --port 5000

Or via the taskfile: ``task tracking:server``.

For GCS-backed artifact storage, configure ``--default-artifact-root
gs://bucket/artifacts`` on the server and ensure ``google-cloud-storage``
is installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from jaxley_extracellular.extracellular.system_monitor import (
    Platform,
    SystemMonitor,
    create_monitor,
    detect_platform,
)


@runtime_checkable
class TrackerProtocol(Protocol):
    """Minimal surface shared by MLflow / wandb / Aim."""

    def __enter__(self) -> TrackerProtocol: ...
    def __exit__(self, *args: object) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...
    def set_status(self, status: str) -> None: ...
    def log_artifact(self, local_path: Path) -> None: ...

    @property
    def run_id(self) -> str: ...


# ------------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------------


def _get_git_hash() -> str:
    """Return the current git HEAD hash, or ``'unknown'``."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_environment_params() -> dict[str, str]:
    """Collect runtime environment metadata for experiment lineage tracking.

    Returns a flat ``dict[str, str]`` with ``env.`` prefixed keys suitable
    for passing directly to ``tracker.log_params()``.
    """
    import jax

    return {
        "env.git_hash": _get_git_hash(),
        "env.jax_version": jax.__version__,
        "env.platform": detect_platform().name,
        "env.device_count": str(jax.device_count()),
        "env.python_version": sys.version.split()[0],
    }


# ------------------------------------------------------------------
# NullTracker -- no-op, zero dependencies
# ------------------------------------------------------------------


class NullTracker:
    """No-op tracker usable as a drop-in for any ``TrackerProtocol``."""

    @property
    def run_id(self) -> str:
        return "null"

    def __enter__(self) -> NullTracker:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        pass

    def set_status(self, status: str) -> None:
        pass

    def log_artifact(self, local_path: Path) -> None:
        pass


# ------------------------------------------------------------------
# MLflowTracker -- HTTP client to a running tracking server
# ------------------------------------------------------------------

MLFLOW_DEFAULT_URI = "http://127.0.0.1:5000"


class MLflowTracker:
    """HTTP client to a running MLflow tracking server.

    The server owns the backend store (SQLite, Postgres, etc.).
    This client only speaks HTTP -- no local database concerns.

    Parameters
    ----------
    platform
        Explicit platform override. ``None`` (default) auto-detects via
        ``detect_platform()`` at ``__enter__`` time.
    """

    def __init__(
        self,
        experiment_name: str = "ecs_sweeps",
        tracking_uri: str = MLFLOW_DEFAULT_URI,
        run_name: str | None = None,
        platform: Platform | None = None,
    ) -> None:
        import mlflow

        self._mlflow: Any = mlflow
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._run_name = run_name
        self._platform_override = platform
        self._run: Any = None
        self._monitor: SystemMonitor | None = None

    @property
    def run_id(self) -> str:
        if self._run is None:
            return ""
        return str(self._run.info.run_id)

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> MLflowTracker:
        platform = self._platform_override or detect_platform()
        self._monitor = create_monitor(platform, tracker=self)

        self._mlflow.set_tracking_uri(self._tracking_uri)
        self._mlflow.set_experiment(self._experiment_name)
        self._run = self._mlflow.start_run(
            run_name=self._run_name,
            log_system_metrics=(platform == Platform.GPU),
        )

        self._monitor.start()
        return self

    def __exit__(self, *args: object) -> None:
        # Stop monitor BEFORE end_run so the drain thread can flush final metrics.
        if self._monitor is not None:
            self._monitor.stop()

        exc_type = args[0] if args else None
        status = "FAILED" if exc_type is not None else "FINISHED"
        self._mlflow.end_run(status=status)
        self._run = None

    # -- logging ------------------------------------------------------------

    @staticmethod
    def _flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, str]:
        flat: dict[str, str] = {}
        for k, v in params.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(MLflowTracker._flatten_params(v, key))
            else:
                flat[key] = str(v)
        return flat

    def log_params(self, params: dict[str, Any]) -> None:
        flat = self._flatten_params(params)
        self._mlflow.log_params(flat)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def set_status(self, status: str) -> None:
        self._mlflow.set_tag("status", status)

    def log_artifact(self, local_path: Path) -> None:
        """Upload a file or directory to the MLflow artifact store."""
        if local_path.is_dir():
            self._mlflow.log_artifacts(str(local_path), artifact_path=local_path.name)
        else:
            self._mlflow.log_artifact(str(local_path))
