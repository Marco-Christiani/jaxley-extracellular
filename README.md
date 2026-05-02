# jaxley-extracellular

Building a differentiable extracellular stimulation pipeline, from electrode waveform to membrane response.

## Mathematical Model

Given compartment centers and an electrode current waveform, the pipeline is:

**1 - Compute extracellular potential at each compartment**
<!-- **1 - Point-source extracellular potential** -->

Let $\mathbf{x}_e \in \mathbb{R}^3$ denote the electrode position and $\mathbf{x}_j \in \mathbb{R}^3$ the centre of compartment $j$. Define the electrode-to-compartment distance:

$$r_j = \|\mathbf{x}_j - \mathbf{x}_e\|_2$$

For a point-source electrode delivering current $I(t)$ into an infinite, homogeneous, isotropic medium of conductivity $\sigma$, the extracellular potential at compartment $j$ is:

$$\phi_j(t) = \frac{I(t)}{4\pi\sigma r_j}$$

**Units.** With $I(t)$ in uA, $\sigma$ in S/m, and $r_j$ in um, the $10^{-6}$ factors in numerator and denominator cancel exactly, and multiplying by $10^3$ to convert V to mV gives the working formula:

$$\phi_j(t) \ [\text{mV}] = \frac{I(t) \ [\mu\text{A}] \cdot 10^3}{4\pi\,\sigma \ [\text{S/m}] \cdot r_j \ [\mu\text{m}]}$$

The spatial dependence and time dependence are fully separable. Defining the transfer factor:

$$h_j = \frac{10^3}{4\pi\sigma r_j} \quad [\text{mV}/\mu\text{A}]$$

the potential is $\phi_j(t) = h_j \cdot I(t)$, which is the factorization implemented as `prefactor` in `point_source_potential`. For a waveform $I(t)$ sampled at $T$ timesteps, $\Phi$ is the outer product $\mathbf{h} \otimes \mathbf{I} \in \mathbb{R}^{N_\text{comp} \times T}$. Notice that $\Phi_{jt} = \phi_j(t)$ and it is more natural to index into $\Phi$ in code (it is a matrix) and prefer $\phi_j(t)$ in formulations (since we are always considering functions of time). Therefore, the matrix constructed in `point_source_potential` is exactly $\Phi$.

<!-- `phi_e [mV] = I [uA] * 1e3 / (4 * pi * sigma [S/m] * r [um])` -->
Implemented in

`src/jaxley_extracellular/extracellular/field.py::point_source_potential`.

**2 - Build the voltage diffusion operator `G` (units `1/ms`) consistent with Jaxley**

  <!-- `dv/dt [mV/ms] = G [1/ms] @ v [mV] + membrane_terms` -->
**The voltage diffusion operator**

The axial coupling term in the cable equation is governed by a linear operator $G \in \mathbb{R}^{N_\text{comp} \times N_\text{comp}}$ assembled from the axial conductances of the cable. Its entries are:

$$G_{ij} = \begin{cases} +g_{ij} & j \neq i,\ j \text{ adjacent to } i \\ -\displaystyle\sum_{k \neq i} g_{ik} & j = i \\ 0 & \text{otherwise} \end{cases}$$

where $g_{ij}$ is the axial conductance between compartments $i$ and $j$. This is the graph Laplacian of the compartment connectivity graph, weighted by axial conductances. Jaxley pre-normalizes $G$ by specific membrane capacitance $c_m$, compartment area, and unit conversion factors so that it has units of $1/\text{ms}$ and the cable equation it solves internally takes the form:

$$\frac{dv_i}{dt} = [Gv]_i - \frac{I_\text{ion}(v_i, \mathbf{s}_i)}{c_m} + \frac{i_\text{ext}}{c_m}$$

In matrix form over all compartments simultaneously:

$$\frac{d\mathbf{v}}{dt} = G\mathbf{v} + \text{membrane terms}$$

where $\mathbf{v} \in \mathbb{R}^{N_\text{comp}}$ is the vector of transmembrane voltages. $G$ is a sparse, symmetric, negative-semidefinite matrix with row sums being zero by construction, encoding current conservation.

**Branchpoint elimination.** Jaxley's internal wiring includes branchpoint pseudo-nodes at cable junctions. These nodes carry no membrane and appear only to enforce Kirchhoff's current law at branch points. `build_voltage_operator_G` eliminates them via Gaussian substitution (the same procedure Jaxley performs internally) producing a reduced operator defined only over real compartments $G \in \mathbb{R}^{N_\text{comp} \times N_\text{comp}}$.

**Extracellular coupling.** By linearity of $G$, the extracellular forcing enters through the same operator:

$$G(\mathbf{v} + \boldsymbol{\phi}_e) = G\mathbf{v} + G\boldsymbol{\phi}_e$$

The term $G\boldsymbol{\phi}_e \in \mathbb{R}^{N_\text{comp}}$ is the activating function which is the discrete Laplacian of the extracellular potential along the cable. For a uniform straight cable with compartment spacing $\Delta x$ it approximates $\frac{\partial^2 \phi_e}{\partial x^2}$ to $O(\Delta x^2)$.

Implemented in
`src/jaxley_extracellular/extracellular/discretization.py::build_voltage_operator_G`.

**3 - Convert extracellular forcing into equivalent injected current.**

The extracellular forcing term $G\boldsymbol{\phi}_e(t)$ has units mV/ms, a rate of voltage change matching the right-hand side of the cable equation. Jaxley's public stimulus API however accepts current in nA, which it internally converts via:

$$i_\text{ext}\ [\mu\text{A/cm}^2] = \frac{i_\text{nA}\ [\text{nA}]}{\text{area}\ [\mu\text{m}^2]} \cdot 10^5$$

and then adds $i_\text{ext}/c_m$ to $dv/dt$. To recover $[G\boldsymbol{\phi}_e]_i$ exactly on the other side of this round-trip, we encode the forcing as an equivalent injected current by inverting the conversion:

$$i_\text{ecs}\ [\text{nA}] = c_m \cdot [G\boldsymbol{\phi}_e]_i \cdot \frac{\text{area}_i}{10^5}$$

The $c_m$ and area factors cancel in the round-trip and are purely an encoding artifact of working through Jaxley's public API rather than modifying the solver. In matrix form over all compartments and timesteps:

$$I_\text{ecs} = \left(c_m \odot \frac{\text{area}}{10^5}\right) \odot (G\Phi) \in \mathbb{R}^{N_\text{comp} \times T}$$

where $\odot$ denotes elementwise multiplication broadcast over the time axis and $G\Phi$ applies $G$ to each column of $\Phi$.

Implemented as:

```python
# See src/jaxley_extracellular/extracellular/equivalent_current.py::phi_e_to_ecs_nA`

# f_ecs [mV/ms]: induced rate-of-change from extracellular gradient
f_ecs: Array = G @ phi_e_mV  # (Ncomp, T)

# i_density [uA/cm^2]: multiply by capacitance to match Jaxley's ODE units
i_density: Array = cm[:, jnp.newaxis] * f_ecs  # (Ncomp, T)

# i_nA [nA]: invert Jaxley's convert_point_process_to_distributed
# i_density [uA/cm^2] = i_nA [nA] / area [um^2] * 1e5
# => i_nA = i_density * area / 1e5
i_ecs_nA: Array = i_density * area_um2[:, jnp.newaxis] / 1e5  # (Ncomp, T)

```

**4 - Package into Jaxley stimulation inputs and integrate over time.**

---

**Relationship to the second derivative**

We compute
$$[G\phi_e]_i = g(\phi_{i-1} - 2\phi_i + \phi_{i+1})$$

We are fundamentally modeling a cylindrical cable compartment of radius $a$, axial resistivity $r_a$ and compartment length $\delta x$ so Ohm's law gives:

$$ g = \frac{\beta}{\delta x}, \quad \text{where} \beta = \frac{\pi a^2}{2 r_a} $$

Which is conductance as cross-sectional area divided by resistivity and length.

---

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
- `sharding.py::make_device_mesh`, `shard_batch`, `iter_batches`
  - JAX device mesh utilities for multi-device distribution (replaces deprecated `pmap`).
- `results_store.py::make_flat_dataset`, `save_zarr`, `load_zarr`, `append_zarr`
  - xarray Dataset construction + Zarr I/O with incremental writes.
- `tracker.py::TrackerProtocol`, `NullTracker`, `MLflowTracker`
  - Backend-agnostic experiment tracking via a `Protocol` + context manager pattern.

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
ncomp = 8
cable_length_um = 800.0
branch = jx.Branch(comp, ncomp=ncomp)
# NB: set("length", x) is per-compartment, not total cable length.
branch.set("length", cable_length_um / ncomp)
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

## Results & Benchmarks

The paper (`paper/paper.tex`) is the canonical write-up of the claims,
evidence, and limitations. Headline numbers: median per-site voltage
RMSE of $0.446\,\mathrm{mV}$ on the full-morphology BBP Pyr cell under
extracellular stimulation against NEURON, and a $\sim\!34\times$
wall-clock speedup over serial single-core NEURON at $B=1000$ on a
single TPU. The reproducibility appendix in the paper documents the
infrastructure stack and the artefact distribution.

## Quick Start

```bash
# GPU shell (default)
nix develop

# TPU shell
nix develop .#tpu

# Tests and checks
nix flake check
```

Smoke commands:

```bash
jaxley-extracellular smoke-devices
jaxley-extracellular smoke-integrate
jaxley-extracellular smoke-tpu  # TPU shell only
```

## Reproducing results

```bash
# 1. Enter the dev shell (pins Python, CUDA, JAX, Jaxley).
nix develop

# 2. Run the unit-test suite (operator round-trip, convergence, gradients).
nix flake check

# 3. Build the paper PDF.
nix run .#paper

# 4. Regenerate every figure from the curated artefact set.
#    Requires the artefact distribution unpacked under
#    `results/paper_package/`. See the paper's reproducibility appendix
#    for the artefact-set URL.
python -m paper.make_figures --which all
```

For TPU-scale runs (NEURON parity, throughput sweeps, gradient receipt
at full morphology), `infra/tofu/` provisions the cloud topology and
`taskfile.yml` carries the lifecycle commands. See
[`infra/tofu/README.md`](./infra/tofu/README.md) for details.

## Infrastructure

Infrastructure code lives in `infra/tofu/` (OpenTofu). See [./infra/tofu/README.md](./infra/tofu/README.md) for provisioning TPU VMs, the experiment tracking server, Cloud SQL, and GCS artifact storage. Lifecycle tasks are in `taskfile.yml`.
