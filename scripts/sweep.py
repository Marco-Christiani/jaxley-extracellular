#!/usr/bin/env python
"""ECS parameter sweep: strength-duration curves across geometries and frequencies.

Key moving parts:
  1. JAX sharding for multi-device distribution
  2. xarray + Zarr for labeled, appendable storage
  3. Tracker protocol for experiment observability


Notes:
When running the remote tracking infrastructure, make sure `--tracking-uri` points to the correct (internal) IP
 for the tracking server in the VPC (which you can get from `tofu -chdir=infra/tofu output`)

In the case of MLFlow we have `--serve-artifacts` on the server (rather than specifying a specific artifact uri for the client
 in addition to the tracking uri) so the flow is:
  -> MLflowTracker.log_artifact(zarr_path)
    -> HTTP multipart upload to tracking server :5000
      -> tracking server streams to gs://bucket/mlflow/<run-id>/artifacts/
The compute instance never touches GCS directly. From the sweep's perspective it's just an HTTP POST to the tracking server URL,
 same as logging metrics.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np

# Dump a Python traceback on segfault/abort/fpe/bus error so libtpu and
# XLA aborts leave a record instead of dying silently.
faulthandler.enable(file=sys.stderr, all_threads=True)
_RUN_LOG_DIR = os.environ.get("TPU_RUN_LOG_DIR")
_RUN_TAG = os.environ.get("TPU_RUN_TAG")
if _RUN_LOG_DIR and _RUN_TAG:
    _faultlog_path = Path(_RUN_LOG_DIR) / f"{_RUN_TAG}.faulthandler"
    _faultlog_fp = open(_faultlog_path, "w", buffering=1)
    faulthandler.register(signal.SIGUSR1, file=_faultlog_fp, all_threads=True)


def _signal_handler(signum: int, _frame: object) -> None:
    """Log the signal then re-raise via the default handler so the OS still
    exits the process. Used to record spot-TPU SIGTERM and similar events
    that would otherwise leave no breadcrumb.
    """
    name = signal.Signals(signum).name
    _LOG.warning("received %s (%d); flushing and re-raising", name, signum)
    sys.stderr.flush()
    sys.stdout.flush()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_sig, _signal_handler)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("sweep")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


_LOG = _configure_logging()

from jaxley_extracellular.extracellular.experiment import (
    make_bbp_pyr_experiment,
    make_hh_cable_experiment,
)
from jaxley_extracellular.extracellular.results_store import (
    append_zarr,
    load_zarr,
    make_flat_dataset,
    save_zarr,
    sweep_metadata,
)
from jaxley_extracellular.extracellular.sharding import (
    config_sharding,
    make_device_mesh,
    shard_batch,
)
from jaxley_extracellular.extracellular.system_monitor import Platform
from jaxley_extracellular.extracellular.tracker import (
    MLflowTracker,
    NullTracker,
    TrackerProtocol,
    collect_environment_params,
)

# Sweep parameters

DEFAULT_PULSE_WIDTHS_MS = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0])
DEFAULT_ELECTRODE_DISTANCES_UM = np.array([25.0, 50.0, 100.0, 200.0])
DEFAULT_FIBER_RADII_UM = np.array([5.0, 10.0, 20.0])
DEFAULT_WAVEFORM_TYPES = [
    "monophasic_cathodic",
    "monophasic_anodic",
    "biphasic_cathodic_first",
]
DEFAULT_FREQUENCIES_HZ = np.array([0.0, 100.0, 200.0, 500.0])

# Binary search
DEFAULT_AMP_LO = 0.0
DEFAULT_AMP_HI = 5000.0  # uA
DEFAULT_N_ITER = 14  # precision: 5000/2^14 ~= 0.3 uA

# Model
DEFAULT_NCOMP = 50
DEFAULT_CABLE_LENGTH_UM = 1250.0
DEFAULT_MAX_BRANCH_LEN_UM = 100.0
DEFAULT_DT_MS = 0.025
DEFAULT_T_MS_SINGLE = 5.0  # for single-pulse (freq=0)
DEFAULT_T_MS_TRAIN = 50.0  # for pulse trains (freq>0)
DEFAULT_SIGMA = 0.3
DEFAULT_RECORD_SITE = "soma"
HH_RECORD_COMP = 0

# Waveform factories (vmap-compatible, mask-based)


class ECSExperimentProto(Protocol):
    def simulate_and_extract(self, waveform: jax.Array, record_comp: int) -> dict[str, Any]: ...


WaveformFactory = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


def _make_mono_cathodic(
    amplitude: jax.Array, pw_steps: jax.Array, period_steps: jax.Array, t_idx: jax.Array
) -> jax.Array:
    phase = t_idx % period_steps
    # (1, T) - single electrode
    return jnp.where(phase < pw_steps, -amplitude, 0.0)[jnp.newaxis, :]


def _make_mono_anodic(
    amplitude: jax.Array, pw_steps: jax.Array, period_steps: jax.Array, t_idx: jax.Array
) -> jax.Array:
    phase = t_idx % period_steps
    return jnp.where(phase < pw_steps, amplitude, 0.0)[jnp.newaxis, :]


def _make_biphasic_cathodic_first(
    amplitude: jax.Array,
    pw_steps: jax.Array,
    period_steps: jax.Array,
    t_idx: jax.Array,
) -> jax.Array:
    phase = t_idx % period_steps
    cathodic = jnp.where(phase < pw_steps, -amplitude, 0.0)
    anodic = jnp.where((phase >= pw_steps) & (phase < 2 * pw_steps), amplitude, 0.0)
    return (cathodic + anodic)[jnp.newaxis, :]


WAVEFORM_FACTORIES: dict[str, WaveformFactory] = {
    "monophasic_cathodic": _make_mono_cathodic,
    "monophasic_anodic": _make_mono_anodic,
    "biphasic_cathodic_first": _make_biphasic_cathodic_first,
}


# Batched binary search (accepts pre-sharded lo/hi)


def _find_thresholds_batched(
    exp: ECSExperimentProto,
    factory: WaveformFactory,
    pw_steps_arr: jax.Array,
    period_steps_arr: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    T: int,
    n_iter: int,
    *,
    record_comp: int,
) -> jax.Array:
    """Binary search over amplitude, vmapped across pulse widths."""
    t_idx = jnp.arange(T)

    @jax.jit
    @jax.vmap
    def _test(amp: jax.Array, pw_steps: jax.Array, per_steps: jax.Array) -> jax.Array:
        w = factory(amp, pw_steps, per_steps, t_idx)
        feats = exp.simulate_and_extract(w, record_comp)
        return cast(jax.Array, feats["spiked"])

    for _ in range(n_iter):
        mid = (lo + hi) / 2.0
        spiked: jax.Array = cast(jax.Array, _test(mid, pw_steps_arr, period_steps_arr))
        lo = jnp.where(spiked, lo, mid)
        hi = jnp.where(spiked, mid, hi)

    return (lo + hi) / 2.0


def _extract_features_at_threshold(
    exp: ECSExperimentProto,
    factory: WaveformFactory,
    thresholds: jax.Array,
    pw_steps_arr: jax.Array,
    period_steps_arr: jax.Array,
    T: int,
    *,
    record_comp: int,
) -> dict[str, jax.Array]:
    """Re-simulate at found thresholds and return full feature dicts."""
    t_idx = jnp.arange(T)

    @jax.jit
    @jax.vmap
    def _run(amp: jax.Array, pw_steps: jax.Array, per_steps: jax.Array) -> dict[str, jax.Array]:
        w = factory(amp, pw_steps, per_steps, t_idx)
        return exp.simulate_and_extract(w, record_comp)

    return cast(dict[str, jax.Array], _run(thresholds, pw_steps_arr, period_steps_arr))


# Resume support


def _already_completed(zarr_path: Path) -> set[tuple[str, float, float, float]]:
    """Load existing Zarr coordinates and return completed (wtype, dist, radius, freq) keys."""
    if not zarr_path.exists():
        return set()
    ds = load_zarr(zarr_path)
    completed: set[tuple[str, float, float, float]] = set()
    wt = ds.coords["waveform_type"].values
    dist = ds.coords["electrode_distance_um"].values
    rad = ds.coords["fiber_radius_um"].values
    freq = ds.coords["frequency_hz"].values
    for i in range(ds.sizes["config"]):
        completed.add((str(wt[i]), float(dist[i]), float(rad[i]), float(freq[i])))
    return completed


def _write_batch(
    ds: Any,
    zarr_path: Path,
) -> None:
    """Save or append a batch to the Zarr store."""
    if zarr_path.exists():
        append_zarr(ds, zarr_path)
    else:
        zarr_path.parent.mkdir(parents=True, exist_ok=True)
        save_zarr(ds, zarr_path)


# Main sweep


def run_sweep(
    outdir: Path,
    tracker: TrackerProtocol,
    batch_size: int = 64,
    *,
    model: str = "hh-cable",
    pulse_widths_ms: np.ndarray | None = None,
    electrode_distances_um: np.ndarray | None = None,
    fiber_radii_um: np.ndarray | None = None,
    waveform_types: list[str] | None = None,
    frequencies_hz: np.ndarray | None = None,
    amp_lo: float = DEFAULT_AMP_LO,
    amp_hi: float = DEFAULT_AMP_HI,
    amp_hi_per_dist: list[float] | None = None,
    n_iter: int = DEFAULT_N_ITER,
    ncomp: int = DEFAULT_NCOMP,
    cable_length_um: float = DEFAULT_CABLE_LENGTH_UM,
    max_branch_len_um: float | None = DEFAULT_MAX_BRANCH_LEN_UM,
    dt_ms: float = DEFAULT_DT_MS,
    t_ms_single: float = DEFAULT_T_MS_SINGLE,
    t_ms_train: float = DEFAULT_T_MS_TRAIN,
    sigma: float = DEFAULT_SIGMA,
    record_site: str = DEFAULT_RECORD_SITE,
) -> Path:
    """Run the full sweep and return the Zarr output path."""
    zarr_path = outdir / "sweep.zarr"

    pulse_widths_ms = (
        np.asarray(pulse_widths_ms, dtype=float)
        if pulse_widths_ms is not None else DEFAULT_PULSE_WIDTHS_MS
    )
    electrode_distances_um = (
        np.asarray(electrode_distances_um, dtype=float)
        if electrode_distances_um is not None else DEFAULT_ELECTRODE_DISTANCES_UM
    )
    fiber_radii_um = (
        np.asarray(fiber_radii_um, dtype=float)
        if fiber_radii_um is not None else DEFAULT_FIBER_RADII_UM
    )
    waveform_types = list(waveform_types) if waveform_types is not None else list(DEFAULT_WAVEFORM_TYPES)
    frequencies_hz = (
        np.asarray(frequencies_hz, dtype=float)
        if frequencies_hz is not None else DEFAULT_FREQUENCIES_HZ
    )

    if model not in {"hh-cable", "bbp-pyr"}:
        raise ValueError(f"unsupported model {model!r}")

    # f32 + bwd_euler diverges when phi_e_max grows past a few hundred mV,
    # so the safe binary-search upper bound shrinks with electrode distance.
    # Pass amp_hi_per_dist (one value per distance) to override the scalar.
    if amp_hi_per_dist is not None:
        if len(amp_hi_per_dist) != len(electrode_distances_um):
            raise ValueError(
                f"amp_hi_per_dist length {len(amp_hi_per_dist)} "
                f"!= electrode_distances_um length {len(electrode_distances_um)}"
            )
        dist_to_amp_hi = dict(zip(electrode_distances_um.tolist(), amp_hi_per_dist, strict=True))
    else:
        dist_to_amp_hi = {float(d): amp_hi for d in electrode_distances_um}

    geometry_radii = np.array([0.0], dtype=float) if model == "bbp-pyr" else fiber_radii_um

    # Sweep config for metadata / logging
    sweep_config: dict[str, Any] = {
        "model": model,
        "pulse_widths_ms": pulse_widths_ms.tolist(),
        "electrode_distances_um": electrode_distances_um.tolist(),
        "fiber_radii_um": fiber_radii_um.tolist(),
        "waveform_types": waveform_types,
        "frequencies_hz": frequencies_hz.tolist(),
        "amp_lo": amp_lo,
        "amp_hi": amp_hi,
        "n_iter": n_iter,
        "ncomp": ncomp,
        "cable_length_um": cable_length_um,
        "max_branch_len_um": max_branch_len_um,
        "dt_ms": dt_ms,
        "t_ms_single": t_ms_single,
        "t_ms_train": t_ms_train,
        "sigma": sigma,
        "record_site": record_site,
        "batch_size": batch_size,
    }

    total_configs = (
        len(pulse_widths_ms)
        * len(waveform_types)
        * len(frequencies_hz)
        * len(electrode_distances_um)
        * len(geometry_radii)
    )
    print(f"Sweep: {total_configs} configs")
    print(f"  model: {model}")
    print(f"  pulse widths: {pulse_widths_ms} ms")
    print(f"  waveform types: {waveform_types}")
    print(f"  frequencies: {frequencies_hz} Hz")
    print(f"  electrode distances: {electrode_distances_um} um")
    if model == "hh-cable":
        print(f"  fiber radii: {fiber_radii_um} um")
    else:
        print(f"  record site: {record_site}, ncomp={ncomp}, max_branch_len={max_branch_len_um}")
    print(f"  binary search: {n_iter} iterations, bracket [{amp_lo}, {amp_hi}] uA")
    print()

    # Sharding setup
    mesh = make_device_mesh()
    sharding = config_sharding(mesh)
    print(f"Device mesh: {mesh.devices.shape} devices")

    # Resume
    completed = _already_completed(zarr_path)
    if completed:
        print(f"Resuming: {len(completed)} groups already done")

    pw_steps_arr = jnp.array([int(pw / dt_ms) for pw in pulse_widths_ms])
    pw_steps_sharded = shard_batch(pw_steps_arr, sharding)

    meta = sweep_metadata(sweep_config)
    configs_done = 0
    configs_skipped = 0
    batch_idx = 0
    wtype_max_len = max(len(w) for w in waveform_types)

    sweep_t0 = time.time()

    with tracker:
        tracker.log_params(sweep_config)
        tracker.log_params(collect_environment_params())
        tracker.set_status("running")

        for freq_hz in frequencies_hz:
            T_ms = t_ms_single if freq_hz == 0.0 else t_ms_train
            T = int(T_ms / dt_ms)

            period_steps_val = int(1000.0 / (freq_hz * dt_ms)) if freq_hz > 0.0 else T

            N = len(pulse_widths_ms)
            period_steps_arr = jnp.full(N, period_steps_val, dtype=jnp.int32)
            period_steps_sharded = shard_batch(period_steps_arr, sharding)

            for dist_um in electrode_distances_um:
                for radius_um in geometry_radii:
                    if model == "hh-cable":
                        exp = make_hh_cable_experiment(
                            ncomp=ncomp,
                            cable_length_um=cable_length_um,
                            radius_um=float(radius_um),
                            electrode_distance_um=float(dist_um),
                            sigma=sigma,
                            dt_ms=dt_ms,
                            T_ms=T_ms,
                        )
                        record_comp = HH_RECORD_COMP
                    else:
                        exp, record_comp = make_bbp_pyr_experiment(
                            ncomp=ncomp,
                            max_branch_len=max_branch_len_um,
                            electrode_distance_um=float(dist_um),
                            record_site=record_site,
                            sigma=sigma,
                            dt_ms=dt_ms,
                            T_ms=T_ms,
                        )

                    for wtype in waveform_types:
                        key = (wtype, float(dist_um), float(radius_um), float(freq_hz))
                        if key in completed:
                            configs_skipped += N
                            print(f"  SKIP {wtype} dist={dist_um} r={radius_um} f={freq_hz}")
                            continue

                        factory = WAVEFORM_FACTORIES[wtype]
                        amp_hi_for_dist = dist_to_amp_hi[float(dist_um)]
                        lo = shard_batch(jnp.full(N, amp_lo, dtype=jnp.float32), sharding)
                        hi = shard_batch(jnp.full(N, amp_hi_for_dist, dtype=jnp.float32), sharding)

                        t0 = time.time()
                        thresholds = _find_thresholds_batched(
                            exp,
                            factory,
                            pw_steps_sharded,
                            period_steps_sharded,
                            lo,
                            hi,
                            T,
                            n_iter,
                            record_comp=record_comp,
                        )
                        # Extract firing pattern features at threshold
                        feats = _extract_features_at_threshold(
                            exp,
                            factory,
                            thresholds,
                            pw_steps_sharded,
                            period_steps_sharded,
                            T,
                            record_comp=record_comp,
                        )
                        elapsed = time.time() - t0
                        thresholds_np = np.asarray(thresholds)[:N]

                        config_arrays = {
                            "pulse_width_ms": np.array(pulse_widths_ms),
                            "model": np.array([model] * N, dtype="<U16"),
                            "waveform_type": np.array([wtype] * N, dtype=f"<U{wtype_max_len}"),
                            "electrode_distance_um": np.full(N, float(dist_um)),
                            "fiber_radius_um": np.full(N, float(radius_um)),
                            "frequency_hz": np.full(N, float(freq_hz)),
                        }
                        metric_arrays = {
                            "threshold_uA": thresholds_np,
                            "charge_nC": thresholds_np * np.array(pulse_widths_ms),
                            "spike_count": np.asarray(feats["spike_count"])[:N],
                            "mean_isi_ms": np.asarray(feats["mean_isi_ms"])[:N],
                            "firing_rate_hz": np.asarray(feats["firing_rate_hz"])[:N],
                            "time_s": np.full(N, elapsed / N),
                        }
                        ds = make_flat_dataset(config_arrays, metric_arrays, attrs=meta)
                        _write_batch(ds, zarr_path)

                        configs_done += N
                        batch_idx += 1
                        tracker.log_metrics(
                            {"configs_done": float(configs_done), "batch_time_s": elapsed},
                            step=batch_idx,
                        )

                        # Per-batch threshold metrics with structured keys
                        for i, pw_ms in enumerate(pulse_widths_ms):
                            thr = float(thresholds_np[i])
                            metric_key = (
                                f"threshold/{model}/{wtype}/d{dist_um:.0f}/r{radius_um:.0f}/pw{pw_ms:.2f}"
                            )
                            tracker.log_metrics({metric_key: thr}, step=batch_idx)
                            print(
                                f"  {model:8s} {wtype:30s}  f={freq_hz:5.0f}Hz  "
                                f"dist={dist_um:5.0f}  r={radius_um:4.0f}  "
                                f"pw={pw_ms:.2f}ms  thr={thr:8.1f}uA"
                            )
                        print(f"  [{wtype}] batch: {elapsed:.1f}s for {N} configs\n")

        # Summary metrics
        total_time = time.time() - sweep_t0
        tracker.log_metrics(
            {
                "summary/total_time_s": total_time,
                "summary/configs_computed": float(configs_done),
                "summary/configs_skipped": float(configs_skipped),
                "summary/total_configs": float(total_configs),
            }
        )

        tracker.log_artifact(zarr_path)
        tracker.set_status("completed")

    print(f"\nSaved {configs_done} results to {zarr_path}")
    return zarr_path


# CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="ECS parameter sweep")
    parser.add_argument("--outdir", type=str, default="results/sweeps")
    parser.add_argument("--tracker", choices=["null", "mlflow"], default="null")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="Tracking server URI (default: http://127.0.0.1:5000)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", choices=["hh-cable", "bbp-pyr"], default="hh-cable")
    parser.add_argument("--pulse-widths-ms", type=float, nargs="+", default=DEFAULT_PULSE_WIDTHS_MS.tolist())
    parser.add_argument(
        "--electrode-distances-um", type=float, nargs="+",
        default=DEFAULT_ELECTRODE_DISTANCES_UM.tolist(),
    )
    parser.add_argument(
        "--fiber-radii-um", type=float, nargs="+",
        default=DEFAULT_FIBER_RADII_UM.tolist(),
        help="Only used for --model hh-cable.",
    )
    parser.add_argument(
        "--waveform-types", nargs="+", default=DEFAULT_WAVEFORM_TYPES,
        choices=sorted(WAVEFORM_FACTORIES.keys()),
    )
    parser.add_argument("--frequencies-hz", type=float, nargs="+", default=DEFAULT_FREQUENCIES_HZ.tolist())
    parser.add_argument("--amp-lo", type=float, default=DEFAULT_AMP_LO)
    parser.add_argument("--amp-hi", type=float, default=DEFAULT_AMP_HI)
    parser.add_argument(
        "--amp-hi-per-dist",
        type=float,
        nargs="+",
        default=None,
        help="Per-distance binary-search upper bound (one float per --electrode-distances-um). Overrides --amp-hi.",
    )
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER)
    parser.add_argument("--ncomp", type=int, default=DEFAULT_NCOMP)
    parser.add_argument("--cable-length-um", type=float, default=DEFAULT_CABLE_LENGTH_UM)
    parser.add_argument("--max-branch-len-um", type=float, default=DEFAULT_MAX_BRANCH_LEN_UM)
    parser.add_argument("--dt-ms", type=float, default=DEFAULT_DT_MS)
    parser.add_argument("--t-ms-single", type=float, default=DEFAULT_T_MS_SINGLE)
    parser.add_argument("--t-ms-train", type=float, default=DEFAULT_T_MS_TRAIN)
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--record-site", choices=["soma", "apical", "basal", "axon"], default=DEFAULT_RECORD_SITE)
    parser.add_argument(
        "--platform",
        choices=["auto", "gpu", "tpu", "cpu"],
        default="auto",
        help="Platform for system metrics (default: auto-detect)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    platform: Platform | None = None
    if args.platform != "auto":
        platform = Platform[args.platform.upper()]

    tracker: TrackerProtocol
    if args.tracker == "mlflow":
        tracker = MLflowTracker(tracking_uri=args.tracking_uri, platform=platform)
    else:
        tracker = NullTracker()

    run_sweep(
        outdir,
        tracker,
        batch_size=args.batch_size,
        model=args.model,
        pulse_widths_ms=np.asarray(args.pulse_widths_ms, dtype=float),
        electrode_distances_um=np.asarray(args.electrode_distances_um, dtype=float),
        fiber_radii_um=np.asarray(args.fiber_radii_um, dtype=float),
        waveform_types=list(args.waveform_types),
        frequencies_hz=np.asarray(args.frequencies_hz, dtype=float),
        amp_lo=args.amp_lo,
        amp_hi=args.amp_hi,
        amp_hi_per_dist=args.amp_hi_per_dist,
        n_iter=args.n_iter,
        ncomp=args.ncomp,
        cable_length_um=args.cable_length_um,
        max_branch_len_um=args.max_branch_len_um,
        dt_ms=args.dt_ms,
        t_ms_single=args.t_ms_single,
        t_ms_train=args.t_ms_train,
        sigma=args.sigma,
        record_site=args.record_site,
    )


if __name__ == "__main__":
    # Top-level guard so an uncaught exception always leaves a full
    # traceback in stderr before the process exits non-zero.
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        _LOG.error("uncaught exception in main(); traceback follows")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
