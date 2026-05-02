"""System metrics monitoring for TPU and CPU platforms.

Provides a ``SystemMonitor`` ABC with concrete implementations:

- ``NullMonitor`` - CPU and GPU (no custom collection needed; trackers handle GPU natively)
- ``TpuMonitor`` - TPU, daemon subprocess polling ``libtpu`` (or ``tpu-info`` as
  a CLI fallback) at 1 Hz

``TpuMonitor`` accepts any ``MetricsLogger`` (the minimal protocol) so it is not
coupled to a specific tracker backend.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Sentinel value to stop the drain thread.
_STOP: None = None

_MetricsItem = tuple[dict[str, float], int] | None

# Need to use forkserver to avoid os.fork() after JAX has started its thread pool.
# fork() in a multithreaded process risks deadlock. forkserver forks from a
#  clean single-threaded helper instead.
_mp = multiprocessing.get_context("forkserver")


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


def _resolve_tpumonitoring() -> Any | None:
    """Try several known import paths for the libtpu monitoring API.

    The module has moved between ``libtpu.sdk.tpumonitoring`` and
    ``libtpu.tpumonitoring`` across libtpu versions. Returns the first one
    that imports, or ``None`` if none are available.
    """
    for path in ("libtpu.sdk.tpumonitoring", "libtpu.tpumonitoring"):
        try:
            mod_root, attr = path.rsplit(".", 1)
            mod = __import__(mod_root, fromlist=[attr])
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            continue
    return None


def _libtpu_metrics(tpumonitoring: Any) -> dict[str, float]:
    """One libtpu polling sample, normalised to MLflow-friendly metric names."""
    metrics_raw: Any = tpumonitoring.get_metrics()
    out: dict[str, float] = {}
    if hasattr(metrics_raw, "tensorcore_util"):
        out["system/tpu_tensorcore_util"] = float(metrics_raw.tensorcore_util)
    if hasattr(metrics_raw, "duty_cycle_pct"):
        out["system/tpu_duty_cycle_pct"] = float(metrics_raw.duty_cycle_pct)
    if hasattr(metrics_raw, "hbm_capacity_usage"):
        out["system/tpu_hbm_usage_bytes"] = float(metrics_raw.hbm_capacity_usage)
    if hasattr(metrics_raw, "hbm_capacity_total"):
        out["system/tpu_hbm_total_bytes"] = float(metrics_raw.hbm_capacity_total)
    return out


_TPU_INFO_NUM_RE = re.compile(r"([\d.]+)")


def _tpu_info_metrics(tpu_info_bin: str) -> dict[str, float]:
    """One `tpu-info` CLI sample, parsed to MLflow-friendly metric names.

    Falls back to a coarse parse: extract the first numeric value per row of
    the table that mentions a known column. Robust to small format changes
    in the upstream `tpu-info` output but loses sub-fields.
    """
    try:
        out = subprocess.run(
            [tpu_info_bin, "--metric"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    metrics: dict[str, float] = {}
    chip_idx = -1
    for line in out.stdout.splitlines():
        low = line.lower()
        if "chip" in low and "device" in low:
            chip_idx += 1
        nums = _TPU_INFO_NUM_RE.findall(line)
        if not nums:
            continue
        suffix = f"_chip{chip_idx}" if chip_idx >= 0 else ""
        if "tensorcore" in low and "util" in low:
            metrics[f"system/tpu_tensorcore_util{suffix}"] = float(nums[0])
        elif "duty" in low and "cycle" in low:
            metrics[f"system/tpu_duty_cycle_pct{suffix}"] = float(nums[0])
        elif "hbm" in low and ("usage" in low or "used" in low) and len(nums) >= 1:
            metrics[f"system/tpu_hbm_usage_bytes{suffix}"] = float(nums[0])
    return metrics


def _tpu_polling_loop(
    metrics_queue: multiprocessing.Queue[_MetricsItem],
    poll_interval: float,
) -> None:
    """Target for the TPU monitoring subprocess.

    Runs in a separate process (not thread) for jaxlib compatibility.
    Polls libtpu (preferred) or the ``tpu-info`` CLI (fallback) at
    ``poll_interval`` Hz and puts ``(metrics, step)`` tuples into
    ``metrics_queue``.  The main process drains the queue and logs via
    the tracker.

    Failure modes are surfaced to stderr (captured by ``infra/scripts/tpu-run.sh``
    into the run's ``.log`` file) instead of swallowed silently, so the
    "no TPU metrics in MLflow" symptom is debuggable from the run log alone.
    """
    tpumonitoring = _resolve_tpumonitoring()
    tpu_info_bin = shutil.which("tpu-info") or os.path.expanduser("~/jx-tpu-env/bin/tpu-info")
    has_tpu_info = os.path.exists(tpu_info_bin) and os.access(tpu_info_bin, os.X_OK)

    if tpumonitoring is not None:
        source = "libtpu"
    elif has_tpu_info:
        source = "tpu-info"
    else:
        print(
            "TpuMonitor: no metrics source available "
            "(libtpu.sdk.tpumonitoring import failed and `tpu-info` not on PATH); "
            "TPU system metrics will not be logged.",
            file=sys.stderr, flush=True,
        )
        # Heartbeat: prove the monitor process started, so its absence in
        # MLflow is unambiguously a wiring issue rather than a silent miss.
        metrics_queue.put(({"system/tpu_monitor_alive": 0.0}, 0))
        return

    print(f"TpuMonitor: polling source={source} interval={poll_interval}s",
          file=sys.stderr, flush=True)
    metrics_queue.put(({"system/tpu_monitor_alive": 1.0}, 0))

    step = 1
    consecutive_failures = 0
    while True:
        try:
            if tpumonitoring is not None:
                metrics = _libtpu_metrics(tpumonitoring)
            else:
                metrics = _tpu_info_metrics(tpu_info_bin)

            if metrics:
                metrics_queue.put((metrics, step))
                step += 1
                consecutive_failures = 0
            else:
                consecutive_failures += 1
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures in (1, 10, 100):
                print(f"TpuMonitor: poll failed ({source}, n={consecutive_failures}): {e!r}",
                      file=sys.stderr, flush=True)

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
        self._queue: multiprocessing.Queue[_MetricsItem] = _mp.Queue()
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
        self._process = _mp.Process(
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
