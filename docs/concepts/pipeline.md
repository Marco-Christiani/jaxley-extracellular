# The ECS Pipeline

This page walks through the complete extracellular stimulation path: from an
electrode waveform in microamps to a Jaxley voltage trace. The implementation is
split into small functions so each physical conversion is inspectable and still
compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

The public path has four stages:

1. compute extracellular potential `phi_e` at every compartment centre,
2. build the Jaxley-consistent voltage diffusion operator `G`,
3. encode `G @ phi_e` as an equivalent injected current in nA, and
4. pass that current through Jaxley's standard `data_stimuli` API.

## Build a small cable

Start with an ordinary Jaxley module. This example uses a 50-compartment
Hodgkin-Huxley cable and records every compartment.

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

The extracellular helpers need compartment-centre coordinates. For simple
synthetic cables,
{func}`~jaxley_extracellular.extracellular.jaxley_adapter.ensure_compartment_centers`
asks Jaxley to populate those coordinates if they are missing.

```python
from jaxley_extracellular.extracellular.jaxley_adapter import (
    ensure_compartment_centers,
    get_compartment_xyz,
)

ensure_compartment_centers(branch)
comp_xyz = get_compartment_xyz(branch)
```

## Step 1: Extracellular potential

For a point-source electrode at position $\mathbf{x}_e$ injecting current
$I(t)$ into a homogeneous medium with conductivity $\sigma$, the distance from
compartment $j$ is

$$
d_{je} = \|\mathbf{x}_j - \mathbf{x}_e\|_2 .
$$

With $I(t)$ in microamps, $\sigma$ in S/m, and $d_{je}$ in micrometres, the
working formula for extracellular potential in millivolts is

$$
\phi_j(t) =
\frac{I(t) \cdot 10^3}{4\pi\sigma d_{je}}.
$$

For multiple electrodes,
{func}`~jaxley_extracellular.extracellular.field.point_source_potential`
applies linear superposition:

$$
\phi_j(t) = \sum_e \frac{I_e(t) \cdot 10^3}{4\pi\sigma d_{je}}.
$$

The returned array is $\boldsymbol{\Phi}$ with shape `(Ncomp, T)`, where
$\boldsymbol{\Phi}_{jt}=\phi_j(t)$.

```python
import jax.numpy as jnp

from jaxley_extracellular.extracellular.field import point_source_potential
from jaxley_extracellular.extracellular.waveforms import make_biphasic_pulse

dt_ms = 0.025
T_ms = 10.0

waveform_uA = make_biphasic_pulse(
    amplitude_uA=200.0,
    pulse_width_ms=0.5,
    dt_ms=dt_ms,
    T_ms=T_ms,
    cathodic_first=True,
)

electrode_positions = jnp.array([[0.0, 50.0, 0.0]])  # (N_elec, 3), um
electrode_currents = waveform_uA[jnp.newaxis, :]      # (N_elec, T), uA

phi_e = point_source_potential(
    comp_xyz=comp_xyz,
    electrode_positions=electrode_positions,
    electrode_currents=electrode_currents,
    sigma=0.3,
)
```

The sign convention is that cathodic current is negative. The field model is
linear in current, so stacking multiple electrodes is just superposition over
the leading electrode axis.

## Step 2: Voltage diffusion operator

The axial coupling term in Jaxley's cable equation is linear in voltage:

$$
\frac{d\mathbf{v}}{dt} = G\mathbf{v} + \text{membrane terms}.
$$

{func}`~jaxley_extracellular.extracellular.discretization.build_voltage_operator_G`
delegates to Jaxley's internal transition-matrix construction, then removes
branchpoint pseudo-nodes so `G` is defined only over real compartments. The
resulting operator has shape `(Ncomp, Ncomp)` and units `1/ms`.

Because the extracellular potential appears as an additive voltage offset, the
linear operator gives the extracellular forcing term:

$$
G(\mathbf{v} + \boldsymbol{\phi})
= G\mathbf{v} + G\boldsymbol{\phi}.
$$

For a straight uniform cable, this is the discrete analogue of the activating
function: the second spatial derivative of extracellular potential along the
cable.

```python
from jaxley_extracellular.extracellular.discretization import (
    build_voltage_operator_G,
)

branch.to_jax()
params = branch.get_all_parameters(pstate=[])
G = build_voltage_operator_G(branch, params)
```

Most users do not need to call this directly;
{func}`~jaxley_extracellular.extracellular.jaxley_adapter.build_ecs_stimuli_nA`
and {class}`~jaxley_extracellular.extracellular.experiment.ECSExperiment` do it
for you. It is useful to know that `G` is defined over real compartments only,
after Jaxley's branchpoint pseudo-nodes are removed.

## Step 3: Equivalent injected current

The term `G @ phi_e` is the activating function $\mathbf{f}$ and has units of
`mV/ms`, matching the right-hand side of the cable equation. Jaxley's public
stimulation API accepts current in nA, so the pipeline inverts Jaxley's
current-density conversion:

$$
\mathbf{f} = G\boldsymbol{\Phi}
$$

$$
\mathbf{I}_\mathrm{ecs} =
\left(\mathbf{c} \odot \frac{\mathbf{A}}{10^5}\right)
\odot (G\boldsymbol{\Phi}).
$$

Here $\mathbf{c}$ is specific membrane capacitance in `uF/cm^2`, $\mathbf{A}$
is compartment surface area in `um^2`, and $\mathbf{I}_\mathrm{ecs}$ is the
equivalent injected current in nA.
The capacitance and area factors are an encoding detail: they make Jaxley's
public `data_stimuli` path produce the intended voltage derivative.

The adapter path bundles this conversion:

```python
from jaxley_extracellular.extracellular.jaxley_adapter import (
    build_ecs_stimuli_nA,
)

i_ecs_nA = build_ecs_stimuli_nA(branch, phi_e)
```

## Step 4: Integration

{func}`~jaxley_extracellular.extracellular.jaxley_adapter.package_data_stimuli`
wraps `I_ecs` into the heterogeneous tuple consumed by `jx.integrate`. No
solver internals need to be patched.

```python
from jaxley_extracellular.extracellular.jaxley_adapter import package_data_stimuli

data_stimuli = package_data_stimuli(branch, i_ecs_nA)

v = jx.integrate(
    branch,
    delta_t=dt_ms,
    t_max=T_ms,
    data_stimuli=data_stimuli,
    solver="bwd_euler",
)

print(v.shape)  # (50, T+1)
```

At this point `v[i, :]` is the transmembrane voltage trace for recorded
compartment `i`.

## Reuse the static pieces

For repeated simulations, build an
{class}`~jaxley_extracellular.extracellular.experiment.ECSExperiment`. It
caches `comp_xyz`, the sparse voltage operator, capacitance, and surface area
so waveform sweeps do not recompute geometry and discretization.

```python
import jax.numpy as jnp

from jaxley_extracellular.extracellular.experiment import make_hh_cable_experiment
from jaxley_extracellular.extracellular.waveforms import make_monophasic_pulse

exp = make_hh_cable_experiment(T_ms=5.0)

waveform = make_monophasic_pulse(
    amplitude_uA=100.0,
    pulse_width_ms=0.5,
    dt_ms=exp.dt_ms,
    T_ms=exp.T_ms,
)

features = exp.simulate_and_extract(waveform[jnp.newaxis, :])
print(features["spiked"], features["latency_ms"])
```

For a batch of waveforms, pass an array with shape `(B, N_elec, T)` to
{meth}`~jaxley_extracellular.extracellular.experiment.ECSExperiment.run_sweep`.

## Differentiability

Every operation in the stimulation path is JAX-native. Gradients can flow
through electrode positions, waveform samples, conductivity, and the membrane
response, subject to the differentiability of the chosen Jaxley solver and
model components.

For example, the potential model is differentiable with respect to electrode
position:

$$
\nabla_{\mathbf{x}_e}
\sum_{j,t} \phi_j(t)
$$

is a JAX gradient through
{func}`~jaxley_extracellular.extracellular.field.point_source_potential`. The
full simulation path can also be differentiated when the chosen model and
solver path are differentiable.
