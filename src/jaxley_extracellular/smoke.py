from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jaxley as jx


def smoke_devices() -> None:
    print("jax", jax.__version__)
    print("default_backend", jax.default_backend())
    print("devices", jax.devices())

    x = jnp.ones((1024, 1024), dtype=jnp.float32)
    y = (x @ x).block_until_ready()
    print("matmul ok", float(y[0, 0]))


def smoke_integrate() -> None:
    print("jax", jax.__version__)
    print("backend", jax.default_backend())
    print("devices", jax.devices())

    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    cell = jx.Cell([branch], parents=[-1])
    net = jx.Network([cell])
    net.set("v", -65.0)
    net.init_states()

    # record from the single compartment so integrate() returns traces.
    net.cell(0).branch(0).comp(0).record(verbose=False)

    dt = 0.025
    t_max = 5.0
    steps = int(t_max // dt + 1)
    stim = jnp.zeros((steps,), dtype=jnp.float32).at[10:50].set(0.1)
    data_stimuli = net.cell(0).branch(0).comp(0).data_stimulate(stim, data_stimuli=None)

    v = jx.integrate(net, delta_t=dt, t_max=t_max, data_stimuli=data_stimuli, solver="bwd_euler")
    v = jax.device_get(v)
    print("integrate ok", v.shape, "v_min", float(v.min()), "v_max", float(v.max()))


def smoke_tpu() -> None:
    devices: list[Any] = jax.devices()  # pyright: ignore[reportUnknownVariableType]
    print("jax", jax.__version__)
    print("default_backend", jax.default_backend())
    print("devices", devices)

    if not any(d.platform == "tpu" for d in devices):
        raise SystemExit("No TPU device found")

    x = jnp.ones((512, 512), dtype=jnp.float32)
    y = (x @ x).block_until_ready()
    print("tpu matmul ok", float(y[0, 0]))
