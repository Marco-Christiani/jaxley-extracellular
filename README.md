# jaxley-extracellular

Building a differentiable extracellular stimulation pipeline, from electrode waveform to membrane response.

## Mathematical Model

Given compartment centers and an electrode current waveform, the pipeline is:

1. Compute extracellular potential at each compartment:

   `phi_e [mV] = I [uA] * 1e3 / (4 * pi * sigma [S/m] * r [um])`

   Implemented in
   `src/jaxley_extracellular/extracellular/field.py::point_source_potential`.

2. Build the voltage diffusion operator `G` (units `1/ms`) consistent with Jaxley:

   `dv/dt [mV/ms] = G @ v [mV] + membrane_terms`

   Implemented in
   `src/jaxley_extracellular/extracellular/discretization.py::build_voltage_operator_G`.

3. Convert extracellular forcing into equivalent injected current:

   `f_ecs [mV/ms] = G @ phi_e`

   `i_ecs [nA] = cm * f_ecs * area / 1e5`

   Implemented in
   `src/jaxley_extracellular/extracellular/equivalent_current.py::phi_e_to_ecs_nA`.

4. Package into Jaxley stimulation inputs and integrate over time.

## Important Functions

- `field.py::point_source_potential`
  - Point-source electrode model (JAX-traceable).
- `discretization.py::build_voltage_operator_G`
  - Dense compartment operator `G`, including branchpoint elimination.
- `equivalent_current.py::phi_e_to_ecs_nA`
  - Unit-consistent conversion from `phi_e` to `data_stimulate` current.
- `jaxley_adapter.py::build_ecs_stimuli_nA`
  - End-to-end adapter: `phi_e -> i_ecs` for a Jaxley module.
- `jaxley_adapter.py::ensure_compartment_centers`, `get_compartment_xyz`
  - Coordinate preparation/extraction for compartment geometry.
- `experiment.py::ECSExperiment`
  - High-level experiment wrapper with simulation, feature extraction, and threshold search.
- `waveforms.py::*`
  - Clinical pulse generators (`monophasic`, `biphasic`, grid sweeps).
- `response.py::*`
  - Spike detection and latency/feature extraction helpers.

## Worked Example

Minimal end-to-end example for one waveform:

```python
import jaxley as jx
import jaxley.channels as ch
import jax.numpy as jnp

from jaxley_extracellular.extracellular.field import point_source_potential
from jaxley_extracellular.extracellular.jaxley_adapter import (
    build_ecs_stimuli_nA,
    ensure_compartment_centers,
    get_compartment_xyz,
    package_data_stimuli,
)
from jaxley_extracellular.extracellular.waveforms import make_biphasic_pulse

# 1) Build a simple HH cable and record all compartments.
comp = jx.Compartment()
branch = jx.Branch(comp, ncomp=8)
branch.set("length", 800.0)
branch.set("radius", 1.0)
branch.set("axial_resistivity", 100.0)
branch.set("capacitance", 1.0)
branch.set("v", -65.0)
branch.insert(ch.HH())
branch.init_states()
for i in range(8):
    branch.comp(i).record(verbose=False)

# 2) Build an electrode waveform in uA.
dt_ms = 0.025
T_ms = 5.0
waveform_uA = make_biphasic_pulse(
    amplitude_uA=100.0,
    pulse_width_ms=0.5,
    dt_ms=dt_ms,
    T_ms=T_ms,
    cathodic_first=True,
)

# 3) Compute phi_e [mV] at compartment centers.
ensure_compartment_centers(branch)
comp_xyz = get_compartment_xyz(branch)
electrode_pos = jnp.array([50.0, 50.0, 0.0])  # um
phi_e_mV = point_source_potential(
    comp_xyz=comp_xyz,
    electrode_pos=electrode_pos,
    electrode_current=waveform_uA,
    sigma=0.3,
)

# 4) Convert phi_e -> i_ecs [nA] and package for integrate.
i_ecs_nA = build_ecs_stimuli_nA(branch, phi_e_mV)
data_stimuli = package_data_stimuli(branch, i_ecs_nA)

# 5) Integrate.
v = jx.integrate(
    branch,
    delta_t=dt_ms,
    t_max=T_ms,
    data_stimuli=data_stimuli,
    solver="bwd_euler",
)
print(v.shape)  # (Ncomp, T+1)
```

## Quick Start

```bash
# GPU shell (default)
nix develop

# TPU shell
nix develop .#tpu

# Tests & Checks
nix flake check
```

Smoke commands:

```bash
jaxley-extracellular smoke-devices
jaxley-extracellular smoke-integrate
jaxley-extracellular smoke-tpu  # TPU shell only
```

## TPU infra

Infrastructure code for TPU VM lifecycle lives in `infra/tofu/` see [./infra/tofu/README.md](./infra/tofu/README.md) for instructions on provisioning and managing TPU instances.
