#!/usr/bin/env python
"""ECS grid sweep on a fixed cell at fixed electrode geometry.

No binary search: each config is one simulation, response features recorded.
Designed to produce mostly-valid sims with a low per-config wall-time so the
output is suitable for direct researcher consumption.

Axes (all may be lists, full Cartesian product):
  - tp_us  : positive-phase pulse width (microseconds), >= 10
  - tn_us  : negative-phase pulse width (microseconds), >= 10
  - ap_uA  : positive-phase amplitude (uA), suprathreshold-ish
  - freq_hz: pulse repetition rate (Hz), <= 10000

Charge balance:
  Anodic amplitude is ALWAYS Ap * Tp / Tn (built into the waveform); the
  user does not pass An. This enforces invasive-stim safety.

Validity filter (configs failing are dropped before simulation):
  1/freq > (tp_us + tn_us) * 1e-6     (pulses fit in one period)
  freq <= 10 kHz
  tp_us >= 10, tn_us >= 10

Locked params (CLI defaults match the validated f32 setup):
  - cell           : BBP Pyr
  - electrode_dist : 100 um (validated, no NaN at amp_hi=30)
  - sigma          : 0.3 S/m
  - dt             : 0.025 ms
  - T_ms           : enough to fit at least 3 periods at the lowest freq

Output:
  results/grid/<run-name>/grid.zarr  (xarray Dataset, one row per valid config)
  optional MLflow run logs the same as scripts/sweep.py.

Polarity convention:
  cathodic-first biphasic (first phase negative). Set --polarity-first=anodic
  to flip; the charge-balance algebra is unchanged.
"""

from __future__ import annotations

import argparse
import faulthandler
import itertools
import logging
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

# Same crash/signal handling as sweep.py: dump tracebacks on segfault and
# log the signal name on SIGTERM/SIGINT/SIGHUP before the default handler
# terminates the process.
faulthandler.enable(file=sys.stderr, all_threads=True)
_RUN_LOG_DIR = os.environ.get("TPU_RUN_LOG_DIR")
_RUN_TAG = os.environ.get("TPU_RUN_TAG")
if _RUN_LOG_DIR and _RUN_TAG:
    _faultlog_path = Path(_RUN_LOG_DIR) / f"{_RUN_TAG}.faulthandler"
    _faultlog_fp = open(_faultlog_path, "w", buffering=1)
    faulthandler.register(signal.SIGUSR1, file=_faultlog_fp, all_threads=True)


def _signal_handler(signum: int, _frame: object) -> None:
    name = signal.Signals(signum).name
    _LOG.warning("received %s (%d); flushing and re-raising", name, signum)
    sys.stderr.flush()
    sys.stdout.flush()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(_sig, _signal_handler)


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("grid_sweep")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


_LOG = _configure_logging()

from jaxley_extracellular.extracellular.experiment import make_bbp_pyr_experiment
from jaxley_extracellular.extracellular.results_store import (
    make_flat_dataset,
    save_zarr,
    sweep_metadata,
)
from jaxley_extracellular.extracellular.sharding import (
    config_sharding,
    make_device_mesh,
    pad_to_devices,
    shard_batch,
)
from jaxley_extracellular.extracellular.tracker import (
    MLflowTracker,
    NullTracker,
    TrackerProtocol,
    collect_environment_params,
)


def _make_asymmetric_biphasic(
    ap: jax.Array,
    tp_steps: jax.Array,
    tn_steps: jax.Array,
    period_steps: jax.Array,
    t_idx: jax.Array,
    *,
    cathodic_first: bool,
) -> jax.Array:
    """Charge-balanced biphasic waveform.

    The first phase has |amplitude|=ap and width tp_steps; the second phase
    has |amplitude|=ap*tp_steps/tn_steps and width tn_steps so that
    Ap*Tp == An*Tn (zero net charge per period).

    Returns shape (1, T) for a single electrode.
    """
    phase = t_idx % period_steps
    in_first = phase < tp_steps
    in_second = (phase >= tp_steps) & (phase < tp_steps + tn_steps)

    sign_first = -1.0 if cathodic_first else 1.0
    sign_second = -sign_first
    an = ap * tp_steps.astype(jnp.float32) / tn_steps.astype(jnp.float32)

    waveform = (
        jnp.where(in_first, sign_first * ap, 0.0)
        + jnp.where(in_second, sign_second * an, 0.0)
    )
    return waveform[jnp.newaxis, :]


def _config_is_valid(tp_us: float, tn_us: float, freq_hz: float) -> bool:
    """Apply the four physical/safety invariants from the design doc."""
    if tp_us < 10.0 or tn_us < 10.0:
        return False
    if freq_hz > 10_000.0 or freq_hz <= 0.0:
        return False
    period_s = 1.0 / freq_hz
    pulses_s = (tp_us + tn_us) * 1e-6
    return period_s > pulses_s


def main() -> None:
    parser = argparse.ArgumentParser(description="ECS grid sweep on fixed BBP cell")
    parser.add_argument("--outdir", type=str, default="results/grid")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--tracker", choices=["null", "mlflow"], default="null")
    parser.add_argument("--tracking-uri", type=str, default="http://127.0.0.1:5000")
    parser.add_argument("--electrode-distance-um", type=float, default=100.0)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--dt-ms", type=float, default=0.025)
    parser.add_argument("--max-branch-len-um", type=float, default=100.0)
    parser.add_argument(
        "--tp-us", type=float, nargs="+", required=True,
        help="Positive-phase pulse widths in microseconds.",
    )
    parser.add_argument(
        "--tn-us", type=float, nargs="+", required=True,
        help="Negative-phase pulse widths in microseconds.",
    )
    parser.add_argument(
        "--ap-uA", type=float, nargs="+", required=True,
        help="Positive-phase amplitudes in microamperes (suprathreshold-ish).",
    )
    parser.add_argument(
        "--freq-hz", type=float, nargs="+", required=True,
        help="Pulse repetition rates in Hz (<= 10 kHz).",
    )
    parser.add_argument(
        "--polarity-first", choices=["cathodic", "anodic"], default="cathodic",
        help="Sign of the first phase (cathodic = -Ap, anodic = +Ap).",
    )
    parser.add_argument(
        "--n-periods", type=int, default=3,
        help="Simulate at least this many full periods at the lowest freq.",
    )
    parser.add_argument(
        "--t-ms-floor", type=float, default=10.0,
        help="Minimum trace duration in ms (overrides n-periods if longer).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=0,
        help=(
            "Process configs in chunks of this many (after padding to n_devices)."
            " 0 (default) = single batch with all valid configs at once."
            " Use a positive value if HBM cannot hold all configs."
        ),
    )
    parser.add_argument(
        "--checkpoint-lengths", type=int, nargs="+", default=None,
        help=(
            "Hierarchical scan-checkpointing factors for jx.integrate."
            " Product must be >= T (number of timesteps). Reduces HBM"
            " (helps multi-chip vmap fit) at the cost of recompute."
            " None (default) = no checkpointing."
        ),
    )
    args = parser.parse_args()

    cathodic_first = args.polarity_first == "cathodic"

    # Build the (Cartesian-product) candidate list, then filter for validity.
    raw = list(itertools.product(args.tp_us, args.tn_us, args.ap_uA, args.freq_hz))
    valid = [
        (tp, tn, ap, fh) for (tp, tn, ap, fh) in raw if _config_is_valid(tp, tn, fh)
    ]
    n_valid = len(valid)
    n_invalid = len(raw) - n_valid
    print(f"Grid: {len(raw)} candidates, {n_valid} valid, {n_invalid} dropped (period <= Tp+Tn or Tp/Tn<10us or freq>10kHz).")
    if n_valid == 0:
        sys.exit("No valid configs; check axis values vs invariants.")

    # Pick T_ms so the lowest-freq config sees n_periods periods.
    lowest_freq = min(fh for _, _, _, fh in valid)
    period_ms = 1000.0 / lowest_freq
    t_ms = max(args.t_ms_floor, args.n_periods * period_ms)
    T = int(t_ms / args.dt_ms)
    print(f"  t_ms={t_ms:.2f}, T={T} steps; lowest freq={lowest_freq} Hz (period={period_ms:.2f} ms)")

    # Tracker
    tracker: TrackerProtocol
    if args.tracker == "mlflow":
        tracker = MLflowTracker(tracking_uri=args.tracking_uri, run_name=args.run_name)
    else:
        tracker = NullTracker()

    # One experiment for the locked geometry.
    exp, record_comp = make_bbp_pyr_experiment(
        ncomp=50,
        max_branch_len=args.max_branch_len_um,
        electrode_distance_um=args.electrode_distance_um,
        record_site="soma",
        sigma=args.sigma,
        dt_ms=args.dt_ms,
        T_ms=t_ms,
    )

    # Stack candidates into parallel arrays for batched simulation.
    aps = np.array([ap for (_, _, ap, _) in valid], dtype=np.float32)
    tp_steps_np = np.array(
        [round(tp * 1e-3 / args.dt_ms) for (tp, _, _, _) in valid], dtype=np.int32,
    )
    tn_steps_np = np.array(
        [round(tn * 1e-3 / args.dt_ms) for (_, tn, _, _) in valid], dtype=np.int32,
    )
    period_steps_np = np.array(
        [round(1000.0 / (fh * args.dt_ms)) for (_, _, _, fh) in valid], dtype=np.int32,
    )

    # Multi-device mesh: shards the leading (config) axis across all chips.
    # Degrades to a (1,) mesh on single-device, in which case shard_batch is
    # a no-op pad and device_put.
    mesh = make_device_mesh()
    sharding = config_sharding(mesh)
    n_devices = jax.device_count()
    print(f"Device mesh: {mesh.devices.shape} (n_devices={n_devices})")

    @jax.jit
    @jax.vmap
    def _run(
        ap: jax.Array, tp_steps: jax.Array, tn_steps: jax.Array, per_steps: jax.Array,
    ) -> dict[str, Any]:
        t_idx = jnp.arange(T)
        w = _make_asymmetric_biphasic(
            ap, tp_steps, tn_steps, per_steps, t_idx,
            cathodic_first=cathodic_first,
        )
        return exp.simulate_and_extract(
            w, record_comp,
            checkpoint_lengths=args.checkpoint_lengths,
        )

    def _to_sharded(arr: np.ndarray) -> jax.Array:
        padded, _ = pad_to_devices(jnp.asarray(arr), n_devices)
        return shard_batch(padded, sharding) if n_devices > 1 else padded

    def _run_chunk(
        ap_in: np.ndarray, tp_in: np.ndarray, tn_in: np.ndarray, per_in: np.ndarray,
    ) -> dict[str, np.ndarray]:
        n = ap_in.shape[0]
        ap_s = _to_sharded(ap_in)
        tp_s = _to_sharded(tp_in)
        tn_s = _to_sharded(tn_in)
        per_s = _to_sharded(per_in)
        feats = _run(ap_s, tp_s, tn_s, per_s)
        jax.block_until_ready(feats["spiked"])
        # Trim padding back to original chunk size n.
        return {k: np.asarray(v)[:n] for k, v in feats.items()}

    chunk_size = args.chunk_size if args.chunk_size > 0 else n_valid
    n_chunks = (n_valid + chunk_size - 1) // chunk_size
    print(f"  chunks: {n_chunks} (chunk_size={chunk_size}; n_valid={n_valid})")

    # Result accumulators
    cfg_arrays: dict[str, list[Any]] = {
        "tp_us": [], "tn_us": [], "ap_uA": [], "an_uA": [],
        "freq_hz": [], "period_ms": [], "polarity_first": [],
        "electrode_distance_um": [], "sigma": [],
    }
    metric_arrays: dict[str, list[Any]] = {
        "spiked": [], "spike_count": [], "firing_rate_hz": [],
        "mean_isi_ms": [], "vmax_mV": [], "vmin_mV": [],
        "latency_ms": [],
    }

    sweep_t0 = time.time()

    with tracker:
        tracker.log_params({
            "model": "bbp-pyr",
            "electrode_distance_um": args.electrode_distance_um,
            "sigma": args.sigma,
            "dt_ms": args.dt_ms,
            "t_ms": t_ms,
            "T": T,
            "polarity_first": args.polarity_first,
            "n_candidates": len(raw),
            "n_valid": n_valid,
            "n_chunks": n_chunks,
            "chunk_size": chunk_size,
            "n_devices": n_devices,
        })
        tracker.log_params(collect_environment_params())
        tracker.set_status("running")

        for chunk_idx in range(n_chunks):
            lo = chunk_idx * chunk_size
            hi = min(lo + chunk_size, n_valid)
            t0 = time.time()
            feats = _run_chunk(
                aps[lo:hi], tp_steps_np[lo:hi], tn_steps_np[lo:hi], period_steps_np[lo:hi],
            )
            wall = time.time() - t0

            for k in range(lo, hi):
                tp_us, tn_us, ap, freq_hz = valid[k]
                an = ap * tp_us / tn_us
                cfg_arrays["tp_us"].append(tp_us)
                cfg_arrays["tn_us"].append(tn_us)
                cfg_arrays["ap_uA"].append(ap)
                cfg_arrays["an_uA"].append(an)
                cfg_arrays["freq_hz"].append(freq_hz)
                cfg_arrays["period_ms"].append(1000.0 / freq_hz)
                cfg_arrays["polarity_first"].append(args.polarity_first)
                cfg_arrays["electrode_distance_um"].append(args.electrode_distance_um)
                cfg_arrays["sigma"].append(args.sigma)

                local = k - lo
                metric_arrays["spiked"].append(bool(feats["spiked"][local]))
                metric_arrays["spike_count"].append(int(feats["spike_count"][local]))
                metric_arrays["firing_rate_hz"].append(float(feats["firing_rate_hz"][local]))
                metric_arrays["mean_isi_ms"].append(float(feats["mean_isi_ms"][local]))
                metric_arrays["vmax_mV"].append(float(feats["vmax"][local]))
                metric_arrays["vmin_mV"].append(float(feats["vmin"][local]))
                metric_arrays["latency_ms"].append(float(feats["latency_ms"][local]))

            n_in_chunk = hi - lo
            tracker.log_metrics(
                {
                    "chunk_wall_s": wall,
                    "chunk_size": float(n_in_chunk),
                    "configs_done": float(hi),
                },
                step=chunk_idx,
            )
            print(
                f"  chunk [{chunk_idx+1}/{n_chunks}] configs={lo}..{hi-1} "
                f"({n_in_chunk} configs)  wall={wall:.2f}s "
                f"({wall / max(n_in_chunk, 1):.3f} s/config)"
            )

        # ---- Save Zarr ----------------------------------------------------
        outdir = Path(args.outdir) / args.run_name
        outdir.mkdir(parents=True, exist_ok=True)
        zarr_path = outdir / "grid.zarr"

        cfg_np = {k: np.asarray(v) for k, v in cfg_arrays.items()}
        metric_np = {k: np.asarray(v) for k, v in metric_arrays.items()}
        meta = sweep_metadata({
            "model": "bbp-pyr",
            "kind": "grid_sweep",
            "polarity_first": args.polarity_first,
            "electrode_distance_um": args.electrode_distance_um,
            "sigma": args.sigma,
            "dt_ms": args.dt_ms,
            "t_ms": t_ms,
        })
        ds = make_flat_dataset(cfg_np, metric_np, attrs=meta)
        save_zarr(ds, zarr_path)
        print(f"\nSaved {n_valid} results to {zarr_path}")

        tracker.log_artifact(zarr_path)
        total_time = time.time() - sweep_t0
        tracker.log_metrics({
            "summary/total_time_s": total_time,
            "summary/n_valid": float(n_valid),
            "summary/n_dropped": float(n_invalid),
        })
        tracker.set_status("completed")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        _LOG.error("uncaught exception in main(); traceback follows")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
