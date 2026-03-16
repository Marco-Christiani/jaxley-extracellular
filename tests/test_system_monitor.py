"""Tests for system metrics monitoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jaxley_extracellular.extracellular.system_monitor import (
    NullMonitor,
    Platform,
    TpuMonitor,
    create_monitor,
    detect_platform,
)


class TestPlatformEnum:
    def test_members(self) -> None:
        assert Platform.GPU.name == "GPU"
        assert Platform.TPU.name == "TPU"
        assert Platform.CPU.name == "CPU"

    def test_from_name(self) -> None:
        assert Platform["GPU"] is Platform.GPU
        assert Platform["TPU"] is Platform.TPU
        assert Platform["CPU"] is Platform.CPU


class TestDetectPlatform:
    def test_detects_cpu(self) -> None:
        mock_device = MagicMock()
        mock_device.platform = "cpu"
        with patch("jax.devices", return_value=[mock_device]):
            assert detect_platform() == Platform.CPU

    def test_detects_gpu(self) -> None:
        mock_device = MagicMock()
        mock_device.platform = "gpu"
        with patch("jax.devices", return_value=[mock_device]):
            assert detect_platform() == Platform.GPU

    def test_detects_cuda_as_gpu(self) -> None:
        mock_device = MagicMock()
        mock_device.platform = "cuda"
        with patch("jax.devices", return_value=[mock_device]):
            assert detect_platform() == Platform.GPU

    def test_detects_tpu(self) -> None:
        mock_device = MagicMock()
        mock_device.platform = "tpu"
        with patch("jax.devices", return_value=[mock_device]):
            assert detect_platform() == Platform.TPU

    def test_unknown_falls_back_to_cpu(self) -> None:
        mock_device = MagicMock()
        mock_device.platform = "xla_futuristic"
        with patch("jax.devices", return_value=[mock_device]):
            assert detect_platform() == Platform.CPU


class TestNullMonitor:
    def test_context_manager(self) -> None:
        with NullMonitor() as m:
            assert isinstance(m, NullMonitor)

    def test_start_stop_noop(self) -> None:
        m = NullMonitor()
        m.start()
        m.stop()


class TestTpuMonitor:
    def test_init_accepts_metrics_logger(self) -> None:
        tracker = MagicMock()
        m = TpuMonitor(tracker=tracker)
        assert m._tracker is tracker  # pyright: ignore[reportPrivateUsage]

    def test_stop_without_start_is_safe(self) -> None:
        tracker = MagicMock()
        m = TpuMonitor(tracker=tracker)
        m.stop()  # should not raise

    def test_queue_relay_forwards_metrics_to_tracker(self) -> None:
        """Metrics put into the queue by the subprocess are forwarded to the tracker."""
        tracker = MagicMock()
        m = TpuMonitor(tracker=tracker, poll_interval=1.0)

        # Start the drain thread without the subprocess.
        import threading

        m._drain_thread = threading.Thread(  # pyright: ignore[reportPrivateUsage]
            target=m._drain_loop,  # pyright: ignore[reportPrivateUsage]
            daemon=True,
        )
        m._drain_thread.start()  # pyright: ignore[reportPrivateUsage]

        m._queue.put(({"system/tpu_tensorcore_util": 0.75}, 0))  # pyright: ignore[reportPrivateUsage]
        m._queue.put(None)  # pyright: ignore[reportPrivateUsage]
        m._drain_thread.join(timeout=2.0)  # pyright: ignore[reportPrivateUsage]

        tracker.log_metrics.assert_called_once_with({"system/tpu_tensorcore_util": 0.75}, step=0)


class TestCreateMonitor:
    def test_cpu_returns_null_monitor(self) -> None:
        assert isinstance(create_monitor(Platform.CPU), NullMonitor)

    def test_gpu_returns_null_monitor(self) -> None:
        assert isinstance(create_monitor(Platform.GPU), NullMonitor)

    def test_tpu_returns_tpu_monitor(self) -> None:
        tracker = MagicMock()
        m = create_monitor(Platform.TPU, tracker=tracker)
        assert isinstance(m, TpuMonitor)

    def test_tpu_raises_without_tracker(self) -> None:
        with pytest.raises(ValueError, match="tracker"):
            create_monitor(Platform.TPU)
