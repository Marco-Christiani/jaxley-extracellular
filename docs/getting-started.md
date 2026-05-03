# Getting Started

## Installation

```bash
pip install jaxley-extracellular
```

Or with Nix:

```bash
nix develop        # GPU shell
nix develop .#tpu  # TPU shell
```

## Smoke test

```bash
jaxley-extracellular smoke-devices
jaxley-extracellular smoke-integrate
```

## Your first simulation

The low-level workflow is explicit: build a Jaxley module, compute the
extracellular potential at compartment centres, convert that potential into the
equivalent `data_stimuli` current, then call `jx.integrate`.

### 1. Build a cable neuron

```python
import jaxley as jx
import jaxley.channels as ch

branch = jx.Branch(jx.Compartment(), ncomp=50)
branch.set("length", 25.0)  # per-compartment length in um
branch.set("radius", 10.0)
branch.set("axial_resistivity", 100.0)
branch.set("capacitance", 1.0)
branch.set("v", -65.0)
branch.insert(ch.HH())
branch.init_states()

for i in range(50):
    branch.comp(i).record(verbose=False)
```

### 2. Choose an electrode waveform

This example uses
{func}`~jaxley_extracellular.extracellular.waveforms.make_monophasic_pulse`.

```python
from jaxley_extracellular.extracellular.waveforms import make_monophasic_pulse

waveform_uA = make_monophasic_pulse(
    amplitude_uA=200.0,
    pulse_width_ms=1.0,
    dt_ms=0.025,
    T_ms=10.0,
    cathodic=True,
)
```

### 3. Compute phi_e and integrate

```python
import jax.numpy as jnp
import jaxley as jx

from jaxley_extracellular.extracellular.field import point_source_potential
from jaxley_extracellular.extracellular.jaxley_adapter import (
    ensure_compartment_centers,
    get_compartment_xyz,
    build_ecs_stimuli_nA,
    package_data_stimuli,
)

ensure_compartment_centers(branch)
phi_e = point_source_potential(
    comp_xyz=get_compartment_xyz(branch),
    electrode_positions=jnp.array([[0.0, 50.0, 0.0]]),
    electrode_currents=waveform_uA[jnp.newaxis, :],
    sigma=0.3,
)
data_stimuli = package_data_stimuli(branch, build_ecs_stimuli_nA(branch, phi_e))
v = jx.integrate(
    branch,
    delta_t=0.025,
    t_max=10.0,
    data_stimuli=data_stimuli,
    solver="bwd_euler",
)
print(v.shape)  # (50, T+1)
```

## Higher-level experiment wrapper

For sweeps, prefer
{class}`~jaxley_extracellular.extracellular.experiment.ECSExperiment` or a
constructor helper such as
{func}`~jaxley_extracellular.extracellular.experiment.make_hh_cable_experiment`.
These cache the compartment coordinates, sparse voltage operator, capacitance,
and area vectors once, then reuse them across waveform batches.

```python
import jax.numpy as jnp

from jaxley_extracellular.extracellular.experiment import make_hh_cable_experiment
from jaxley_extracellular.extracellular.waveforms import make_biphasic_pulse

exp = make_hh_cable_experiment(T_ms=5.0)
waveform = make_biphasic_pulse(
    amplitude_uA=100.0,
    pulse_width_ms=0.5,
    dt_ms=exp.dt_ms,
    T_ms=exp.T_ms,
)

features = exp.simulate_and_extract(waveform[jnp.newaxis, :])
print(features["spiked"], features["latency_ms"])
```
