import sys

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch

from jaxley_extracellular.smoke import smoke_devices, smoke_integrate


def test_smoke_devices_runs(capsys: CaptureFixture[str]) -> None:
    smoke_devices()
    out = capsys.readouterr().out
    assert "jax" in out
    assert "devices" in out


def test_smoke_integrate_runs(capsys: CaptureFixture[str]) -> None:
    smoke_integrate()
    out = capsys.readouterr().out
    assert "integrate ok" in out


def test_cli_smoke_devices(monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
    from jaxley_extracellular.cli import main

    monkeypatch.setattr(sys, "argv", ["jaxley-extracellular", "smoke-devices"])
    main()
    out = capsys.readouterr().out
    assert "matmul ok" in out
