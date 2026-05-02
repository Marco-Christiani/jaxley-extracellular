"""Experiment tracking protocol and implementations.

``NullTracker`` is always available (zero dependencies).
``MLflowTracker`` wraps mlflow as a pure HTTP client pointing at a
running MLflow tracking server.  Start one with::

    mlflow server --backend-store-uri sqlite:///results/tracking.db \\
                  --default-artifact-root ./results/mlartifacts \\
                  --host 127.0.0.1 --port 5000

For GCS-backed artifact storage, configure ``--default-artifact-root
gs://bucket/artifacts`` on the server and ensure ``google-cloud-storage``
is installed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
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


# Environment helpers
DEL = object()


@contextmanager
def env(**env_vars: str | object) -> Generator[None, None, None]:
    """Temporarily set environment variables, restore on exit."""
    original: dict[str, str | None] = {}

    try:
        for key, value in env_vars.items():
            original[key] = os.environ.get(key)

            if value is DEL:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)

        yield
    finally:
        for key, orig_value in original.items():
            if orig_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig_value


def _get_git_hash() -> str:
    """Return the current git HEAD hash, or ``'unknown'``.

    Strips ``LD_PRELOAD`` for the subprocess: on TPU VMs the worker python
    runs with nix-built tcmalloc preloaded, and inheriting that into the
    system ``git`` binary (linked against an older glibc) prints a wall of
    "GLIBC_2.36 not found" warnings before failing.
    """
    env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env=env,
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


# NullTracker (no-op, zero dependencies)


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


# MLflowTracker (HTTP client to a running tracking server)

MLFLOW_DEFAULT_URI = "http://127.0.0.1:5000"


class MLflowTracker:
    """HTTP client to a running MLflow tracking server.

    The server owns the backend store (SQLite, Postgres, etc.).
    This client only speaks HTTP, no local database concerns.

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

        self._mlflow = mlflow
        self._experiment_name = experiment_name
        self._tracking_uri = tracking_uri
        self._run_name = run_name
        self._platform_override = platform
        self._run: mlflow.ActiveRun | None = None
        self._monitor: SystemMonitor | None = None

    @property
    def run_id(self) -> str:
        if self._run is None:
            return ""
        return str(self._run.info.run_id)

    # Context manager

    def __enter__(self) -> MLflowTracker:
        platform = self._platform_override or detect_platform()
        self._monitor = create_monitor(platform, tracker=self)
        with env(LD_PRELOAD=DEL):
            self._mlflow.set_tracking_uri(self._tracking_uri)
            self._mlflow.set_experiment(self._experiment_name)
            self._run = self._mlflow.start_run(
                run_name=self._run_name,
                log_system_metrics=(platform == Platform.GPU),
            )

        self._monitor.start()
        return self

    def __exit__(self, *args: object) -> None:
        # Stop monitor before end_run so the drain thread flushes final metrics.
        if self._monitor is not None:
            self._monitor.stop()

        # Attach run-wrapper log files (set by infra/scripts/tpu-run.sh) as
        # MLflow artifacts under run_logs/ for lineage on uncaught exceptions.
        # SIGKILL still bypasses __exit__.
        log_dir_env = os.environ.get("TPU_RUN_LOG_DIR")
        log_tag = os.environ.get("TPU_RUN_TAG")
        if log_dir_env and log_tag:
            log_dir = Path(log_dir_env)
            for suffix in ("log", "exit", "time", "dmesg", "journal", "trace"):
                p = log_dir / f"{log_tag}.{suffix}"
                if p.exists():
                    try:
                        self._mlflow.log_artifact(str(p), artifact_path="run_logs")
                    except Exception as e:
                        print(f"Failed to save artifact {p}", e)

        # Flush any async-logged metrics before closing the run.
        try:
            self._mlflow.flush_async_logging()
        except Exception:
            pass

        exc_type = args[0] if args else None
        status = "FAILED" if exc_type is not None else "FINISHED"
        self._mlflow.end_run(status=status)
        self._run = None

    # Logging

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
        # synchronous=True so each metric is flushed before return; otherwise
        # an interrupted teardown can drop the async queue and lose them.
        self._mlflow.log_metrics(metrics, step=step, synchronous=True)

    def set_status(self, status: str) -> None:
        self._mlflow.set_tag("status", status)

    def log_artifact(self, local_path: Path) -> None:
        """Upload a file or directory to the MLflow artifact store."""
        if local_path.is_dir():
            self._mlflow.log_artifacts(str(local_path), artifact_path=local_path.name)
        else:
            self._mlflow.log_artifact(str(local_path))
