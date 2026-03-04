import sys

from jaxley_extracellular.smoke import smoke_devices, smoke_integrate


def test_smoke_devices_runs(capsys):
    smoke_devices()
    out = capsys.readouterr().out
    assert "jax" in out
    assert "devices" in out


def test_smoke_integrate_runs(capsys):
    smoke_integrate()
    out = capsys.readouterr().out
    assert "integrate ok" in out


def test_cli_smoke_devices(monkeypatch, capsys):
    from jaxley_extracellular.cli import main

    monkeypatch.setattr(sys, "argv", ["jaxley-extracellular", "smoke-devices"])
    main()
    out = capsys.readouterr().out
    assert "matmul ok" in out
