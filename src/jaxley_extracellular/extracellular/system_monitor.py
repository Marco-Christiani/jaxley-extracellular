"""System metrics monitoring for TPU and CPU platforms.

Provides a ``SystemMonitor`` ABC with concrete implementations:

- ``NullMonitor`` - CPU and GPU (no custom collection needed; trackers handle GPU natively)
- ``TpuMonitor`` - TPU, daemon subprocess polling ``libtpu.sdk.tpumonitoring`` at 1 Hz

``TpuMonitor`` accepts any ``MetricsLogger`` (the minimal protocol) so it is not
coupled to a specific tracker backend.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Sentinel value to stop the drain thread.
_STOP: None = None

_MetricsItem = tuple[dict[str, float], int] | None


class MetricsLogger(Protocol):
    """Minimal protocol required by ``TpuMonitor`` to log collected metrics.

    Any ``TrackerProtocol`` implementation satisfies this.
    """

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None: ...


class Platform(Enum):
    GPU = auto()
    TPU = auto()
    CPU = auto()


def detect_platform() -> Platform:
    """Detect the current JAX platform via ``jax.devices()[0].platform``."""
    import jax

    platform_str: str = str(jax.devices()[0].platform)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType,reportAttributeAccessIssue]
    match platform_str:
        case "gpu" | "cuda":
            return Platform.GPU
        case "tpu":
            return Platform.TPU
        case "cpu":
            return Platform.CPU
        case _:
            logger.warning("Unknown JAX platform %r, falling back to CPU", platform_str)
            return Platform.CPU


class SystemMonitor(ABC):
    """Context-managed lifecycle for system metrics collection."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    def __enter__(self) -> SystemMonitor:
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


class NullMonitor(SystemMonitor):
    """No-op monitor for CPU."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _tpu_polling_loop(
    metrics_queue: multiprocessing.Queue[_MetricsItem],
    poll_interval: float,
) -> None:
    """Target for the TPU monitoring subprocess.

    Runs in a separate process (not thread) for jaxlib compatibility.
    Polls ``libtpu.sdk.tpumonitoring`` at ``poll_interval`` Hz and puts
    ``(metrics, step)`` tuples into ``metrics_queue``.  The main process
    drains the queue and logs via the tracker.
    """
    try:
        from libtpu.sdk import tpumonitoring  # type: ignore[import-not-found]
    except ImportError:
        return

    step = 0
    while True:
        try:
            metrics_raw: Any = tpumonitoring.get_metrics()  # pyright: ignore[reportUnknownVariableType]
            metrics: dict[str, float] = {}
            if hasattr(metrics_raw, "tensorcore_util"):
                metrics["system/tpu_tensorcore_util"] = float(metrics_raw.tensorcore_util)
            if hasattr(metrics_raw, "duty_cycle_pct"):
                metrics["system/tpu_duty_cycle_pct"] = float(metrics_raw.duty_cycle_pct)
            if hasattr(metrics_raw, "hbm_capacity_usage"):
                metrics["system/tpu_hbm_usage_bytes"] = float(metrics_raw.hbm_capacity_usage)
            if hasattr(metrics_raw, "hbm_capacity_total"):
                metrics["system/tpu_hbm_total_bytes"] = float(metrics_raw.hbm_capacity_total)

            if metrics:
                metrics_queue.put((metrics, step))
                step += 1
        except Exception:
            logger.debug("TPU metrics poll failed", exc_info=True)

        time.sleep(poll_interval)


class TpuMonitor(SystemMonitor):
    """TPU monitor using a daemon subprocess polling libtpu at 1 Hz.

    Metrics are relayed from the subprocess to the main process via a
    ``multiprocessing.Queue`` and logged through ``tracker.log_metrics``.
    This keeps ``TpuMonitor`` decoupled from any specific tracker backend.
    """

    def __init__(self, tracker: MetricsLogger, poll_interval: float = 1.0) -> None:
        self._tracker = tracker
        self._poll_interval = poll_interval
        self._queue: multiprocessing.Queue[_MetricsItem] = multiprocessing.Queue()
        self._process: multiprocessing.Process | None = None
        self._drain_thread: threading.Thread | None = None

    def _drain_loop(self) -> None:
        """Drain the metrics queue in the main process and forward to tracker."""
        while True:
            try:
                item = self._queue.get(timeout=2.0)
                if item is None:
                    break
                metrics, step = item
                self._tracker.log_metrics(metrics, step=step)
            except queue.Empty:
                continue
            except Exception:
                logger.debug("TpuMonitor drain error", exc_info=True)

    def start(self) -> None:
        self._process = multiprocessing.Process(
            target=_tpu_polling_loop,
            args=(self._queue, self._poll_interval),
            daemon=True,
        )
        self._process.start()
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()
        logger.info("TpuMonitor: daemon process started (pid=%s)", self._process.pid)

    def stop(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
            logger.info("TpuMonitor: daemon process stopped")
        self._process = None

        # Send sentinel so drain thread flushes remaining items and exits.
        self._queue.put(_STOP)
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=5)
        self._drain_thread = None


def create_monitor(
    platform: Platform,
    tracker: MetricsLogger | None = None,
) -> SystemMonitor:
    """Factory: create the appropriate monitor for *platform*.

    Raises ``ValueError`` if TPU is requested without a *tracker*.
    """
    match platform:
        case Platform.CPU | Platform.GPU:
            return NullMonitor()
        case Platform.TPU:
            if tracker is None:
                raise ValueError("TpuMonitor requires a tracker for metric logging")
            return TpuMonitor(tracker=tracker)
