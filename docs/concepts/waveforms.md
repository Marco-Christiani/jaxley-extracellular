# Waveforms

Waveform helpers return one-dimensional JAX arrays in microamps. The sign
convention is clinical: cathodic current is negative, and anodic current is
positive.
{func}`~jaxley_extracellular.extracellular.field.point_source_potential`
expects a two-dimensional current array with shape `(N_elec, T)`, so a single
waveform is usually passed as `waveform[jnp.newaxis, :]`.

## Monophasic

{func}`~jaxley_extracellular.extracellular.waveforms.make_monophasic_pulse`
creates one rectangular phase.

```python
from jaxley_extracellular.extracellular.waveforms import make_monophasic_pulse

waveform = make_monophasic_pulse(
    amplitude_uA=200.0,
    pulse_width_ms=1.0,
    dt_ms=0.025,
    T_ms=10.0,
    cathodic=True,
)
```

With `cathodic=True`, samples during the pulse are `-amplitude_uA`; otherwise
they are positive.

## Biphasic

{func}`~jaxley_extracellular.extracellular.waveforms.make_biphasic_pulse` adds
an equal-magnitude second phase with opposite sign. The default is cathodic
first, followed by anodic charge balancing.

```python
from jaxley_extracellular.extracellular.waveforms import make_biphasic_pulse

waveform = make_biphasic_pulse(
    amplitude_uA=200.0,
    pulse_width_ms=0.5,
    dt_ms=0.025,
    T_ms=10.0,
    cathodic_first=True,
)
```

`interphase_ms` can insert a zero-current gap between phases. `delay_ms` shifts
the pulse onset.

## Pulse trains

{func}`~jaxley_extracellular.extracellular.waveforms.make_pulse_train` is the
shared implementation. Set `frequency_hz > 0` to repeat pulses until the
waveform reaches `T_ms`.

```python
from jaxley_extracellular.extracellular.waveforms import make_pulse_train

train = make_pulse_train(
    amplitude_uA=80.0,
    pulse_width_ms=0.2,
    dt_ms=0.025,
    T_ms=20.0,
    frequency_hz=100.0,
    biphasic=True,
    interphase_ms=0.05,
)
```

## Grid sweeps

{func}`~jaxley_extracellular.extracellular.waveforms.make_biphasic_grid`
builds a batch of biphasic waveforms over amplitude and pulse-width grids.

```python
import jax.numpy as jnp

from jaxley_extracellular.extracellular.waveforms import make_biphasic_grid

waveforms, grid_amps, grid_pws = make_biphasic_grid(
    amplitudes_uA=jnp.array([50.0, 100.0, 200.0]),
    pulse_widths_ms=jnp.array([0.1, 0.25, 0.5, 1.0]),
    dt_ms=0.025,
    T_ms=10.0,
)
```

The returned `waveforms` array has shape `(N, T)`, where `N` is
`len(amplitudes_uA) * len(pulse_widths_ms)`. The companion arrays record the
amplitude and pulse width used for each row.
