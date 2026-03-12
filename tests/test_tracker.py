"""Tests for experiment tracking protocol and implementations."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from jaxley_extracellular.extracellular.tracker import (
    MLflowTracker,
    NullTracker,
    TrackerProtocol,
)

# ------------------------------------------------------------------
# NullTracker
# ------------------------------------------------------------------


class TestNullTracker:
    def test_satisfies_protocol(self) -> None:
        t = NullTracker()
        assert isinstance(t, TrackerProtocol)

    def test_run_id(self) -> None:
        assert NullTracker().run_id == "null"

    def test_context_manager_noop(self) -> None:
        t = NullTracker()
        with t as tracker:
            assert tracker is t
            tracker.log_params({"a": 1})
            tracker.log_metrics({"loss": 0.5}, step=0)
            tracker.set_status("running")
            tracker.log_artifact_path(Path("/tmp/test"))
            tracker.log_artifact(Path("/tmp/test"))
        # No exception -> success


# ------------------------------------------------------------------
# MLflowTracker -- requires a tracking server
# ------------------------------------------------------------------


@pytest.fixture(scope="module")
def mlflow_server(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Start a temporary MLflow tracking server for the test module."""
    mlflow = pytest.importorskip("mlflow")
    del mlflow  # only needed for the skip check

    db_dir = tmp_path_factory.mktemp("mlflow")
    db_uri = f"sqlite:///{db_dir / 'tracking.db'}"
    port = 5123  # avoid colliding with a real server on 5000

    proc = subprocess.Popen(
        [
            "mlflow",
            "server",
            "--backend-store-uri",
            db_uri,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--workers",
            "1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    server_uri = f"http://127.0.0.1:{port}"

    # Wait for server to be ready
    import urllib.request

    for _ in range(40):
        try:
            urllib.request.urlopen(f"{server_uri}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.25)
    else:
        proc.kill()
        raise RuntimeError("MLflow server did not start")

    yield server_uri

    proc.terminate()
    proc.wait(timeout=5)


@pytest.mark.slow
class TestMLflowTracker:
    def test_satisfies_protocol(self) -> None:
        pytest.importorskip("mlflow")
        t = MLflowTracker()
        assert isinstance(t, TrackerProtocol)

    def test_context_manager_starts_and_ends_run(self, mlflow_server: str) -> None:
        import mlflow

        t = MLflowTracker(
            tracking_uri=mlflow_server, experiment_name="test_exp", run_name="test_run"
        )
        with t:
            assert t.run_id != ""
            assert mlflow.active_run() is not None
        # After exit, no active run
        assert mlflow.active_run() is None

    def test_log_params_retrievable(self, mlflow_server: str) -> None:
        import mlflow

        t = MLflowTracker(tracking_uri=mlflow_server, experiment_name="test_params")
        with t:
            t.log_params({"lr": 0.01, "nested": {"depth": 3}})
            run_id = t.run_id

        client = mlflow.MlflowClient(tracking_uri=mlflow_server)
        run: Any = client.get_run(run_id)
        assert run.data.params["lr"] == "0.01"
        assert run.data.params["nested.depth"] == "3"

    def test_log_metrics(self, mlflow_server: str) -> None:
        import mlflow

        t = MLflowTracker(tracking_uri=mlflow_server, experiment_name="test_metrics")
        with t:
            t.log_metrics({"loss": 0.5, "acc": 0.9}, step=1)
            run_id = t.run_id

        client = mlflow.MlflowClient(tracking_uri=mlflow_server)
        run: Any = client.get_run(run_id)
        assert float(run.data.metrics["loss"]) == pytest.approx(0.5)

    def test_set_status_and_artifact_path(self, mlflow_server: str) -> None:
        import mlflow

        t = MLflowTracker(tracking_uri=mlflow_server, experiment_name="test_tags")
        with t:
            t.set_status("running")
            t.log_artifact_path(Path("/tmp/sweep.zarr"))
            run_id = t.run_id

        client = mlflow.MlflowClient(tracking_uri=mlflow_server)
        run: Any = client.get_run(run_id)
        assert run.data.tags["status"] == "running"
        assert run.data.tags["artifact_path"] == "/tmp/sweep.zarr"

    def test_log_artifact_file(self, mlflow_server: str, tmp_path: Path) -> None:
        import mlflow

        test_file = tmp_path / "metrics.txt"
        test_file.write_text("threshold=42.0\n")

        t = MLflowTracker(tracking_uri=mlflow_server, experiment_name="test_artifact_file")
        with t:
            t.log_artifact(test_file)
            run_id = t.run_id

        client = mlflow.MlflowClient(tracking_uri=mlflow_server)
        artifacts: Any = client.list_artifacts(run_id)
        names = [a.path for a in artifacts]
        assert "metrics.txt" in names

    def test_log_artifact_directory(self, mlflow_server: str, tmp_path: Path) -> None:
        import mlflow

        # Create a minimal directory structure (mimics a Zarr store)
        zarr_dir = tmp_path / "test.zarr"
        zarr_dir.mkdir()
        (zarr_dir / ".zmetadata").write_text("{}")
        (zarr_dir / "data.bin").write_bytes(b"\x00" * 16)

        t = MLflowTracker(tracking_uri=mlflow_server, experiment_name="test_artifact_dir")
        with t:
            t.log_artifact(zarr_dir)
            run_id = t.run_id

        client = mlflow.MlflowClient(tracking_uri=mlflow_server)
        artifacts: Any = client.list_artifacts(run_id)
        names = [a.path for a in artifacts]
        assert "test.zarr" in names
