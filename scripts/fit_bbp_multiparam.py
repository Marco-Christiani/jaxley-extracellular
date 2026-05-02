"""Multi-parameter gradient recovery on BBP L2/3 Pyr.

Perturb ~10 channel conductances across soma / apical / axon, recover
them with Adam over P parallel restarts (vmap'd random inits), compare
to the ground truth that generated the target trace.

Each restart runs N_STEPS forward+backward sims of the ~700-comp Pyr
cell; with P = 64 restarts and N = 300 steps that is ~19k forward sims
per fit, coupled through the optimiser state. Gradient-free search at
K ~= 10 parameters needs orders of magnitude more evaluations.

Bounded params via per-parameter inverse-sigmoid transform, MAE loss
on the somatic voltage trace, restarts fused into a single jit'd
jax.lax.scan. P = 64 fits on an L40S; scale up on TPU.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import jaxley as jx
import numpy as np
import optax

from jaxley_extracellular.bbp.cell_factory import make_pyr_cell
from jaxley_extracellular.extracellular.tracker import (
    MLflowTracker,
    NullTracker,
    TrackerProtocol,
    collect_environment_params,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "results"

# Config

DT = 0.025
V_INIT = -75.0
T_MAX = 80.0
I_DELAY = 10.0
I_DUR = 40.0
I_AMP = 0.5  # nA, suprathresh, same stim as fit_demo.py for continuity

# jx.integrate checkpoint schedule for memory-efficient backward pass.
# At DT = 0.025 and T_MAX = 80 we run 3201 integration steps. The Jaxley
# paper uses a two-level schedule [101, 2] which re-materialises every 2
# outer checkpoints of 101 inner steps each, turning O(N) stored activations
# into O(sqrt(N))-ish. We tune the outer factor so that outer * inner >=
# total steps; [101, 33] = 3333 >= 3201. Without this, backward through
# jx.integrate OOMs even on a single GPU at modest batch size.
CHECKPOINT_LENGTHS: tuple[int, int] = (101, 33)


@dataclass(frozen=True)
class ParamSpec:
    """Single scalar parameter to recover.

    Args:
        section:  Jaxley cell section ("soma", "apical", "axon"). The
                  trainable is created via ``cell.<section>.make_trainable``,
                  so a single scalar shared across all compartments in that
                  section.
        key:      Jaxley parameter name matching the inserted channel
                  (e.g. "NaTs2T_gNaTs2T", "H_gH").
        true:     Ground-truth value used to generate the target trace.
        lo, hi:   Bounds for the inverse-sigmoid transform. Chosen to span
                  2-5x around the true value so the search space is real
                  but not trivially narrow.
    """
    section: str
    key: str
    true: float
    lo: float
    hi: float


# ~10 parameters spread across soma / apical / axon. Bounds are chosen to
# straddle the BBP default by 2-5x: wide enough that recovery is non-trivial,
# narrow enough that physical inertia (spike/no-spike boundary) stays
# consistent with the target within the valid domain.
PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("soma",   "NaTs2T_gNaTs2T", 0.926705, 0.3,   1.5),
    ParamSpec("soma",   "SKv3_1_gSKv3_1", 0.102517, 0.02,  0.4),
    ParamSpec("soma",   "SKE2_gSKE2",     0.099433, 0.02,  0.4),
    ParamSpec("soma",   "CaHVA_gCaHVA",   0.000374, 0.0,   0.002),
    ParamSpec("soma",   "H_gH",           0.000080, 0.0,   0.0004),
    ParamSpec("apical", "NaTs2T_gNaTs2T", 0.012009, 0.0,   0.04),
    ParamSpec("apical", "SKv3_1_gSKv3_1", 0.000513, 0.0,   0.005),
    ParamSpec("apical", "M_gM",           0.000740, 0.0,   0.003),
    ParamSpec("axon",   "NaTaT_gNaTaT",   3.429725, 1.0,   6.0),
    ParamSpec("axon",   "KPst_gKPst",     0.959296, 0.2,   3.0),
    ParamSpec("axon",   "SKv3_1_gSKv3_1", 0.094971, 0.01,  0.3),
)
K = len(PARAMS)


# Parameter transform: bounded via inverse sigmoid

def sigmoid(x: jax.Array) -> jax.Array:
    # Steeper slope (1.5x) matches the Jaxley paper. Pushes grads away from
    # the saturating tails faster than a standard sigmoid.
    return 1.0 / (1.0 + jnp.exp(-1.5 * x))


def logit(y: jax.Array) -> jax.Array:
    # Inverse of the scaled sigmoid above. Only used at init to warm-start
    # theta from a chosen physical value.
    return jnp.log(y / (1.0 - y)) / 1.5


def theta_to_phys(theta: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """Map unbounded theta in R^K to physical values in [lo, hi]^K."""
    return lo + (hi - lo) * sigmoid(theta)


def phys_to_theta(phys: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """Inverse: used to pick an init whose physical value sits at a target."""
    u = (phys - lo) / (hi - lo)
    # Keep u away from 0 and 1 so logit stays finite.
    u = jnp.clip(u, 1e-4, 1 - 1e-4)
    return logit(u)


LO = jnp.array([p.lo for p in PARAMS])
HI = jnp.array([p.hi for p in PARAMS])
TRUE = jnp.array([p.true for p in PARAMS])
THETA_TRUE = phys_to_theta(TRUE, LO, HI)


# Cell assembly and simulation

def build_cell() -> tuple[jx.Cell, Any]:
    """Build the BBP Pyr cell and mark our K trainable parameters.

    Returns (cell, soma_comp). soma_comp is the handle we use for recording
    and for stimulating. We stim intracellularly at the soma to match
    fit_demo.py and parity_bbp_pyr.py.
    """
    cell = make_pyr_cell(ncomp=2, max_branch_len=100.0)
    cell.set("v", V_INIT)
    soma_comp = cell.soma.branch(0).comp(0)  # pyright: ignore[reportOptionalMemberAccess]
    soma_comp.record("v")

    # Order of make_trainable calls determines the order in cell.get_parameters().
    # We rely on this when packing / unpacking flat theta <-> params list.
    for spec in PARAMS:
        section = getattr(cell, spec.section)
        section.make_trainable(spec.key)

    return cell, soma_comp


def pack_params_from_phys(phys: jax.Array, template: list[dict[str, jax.Array]]) -> list[dict[str, jax.Array]]:
    """Convert a flat (K,) physical value array into Jaxley's param list.

    Jaxley expects a list of single-key dicts; each dict's value has a shape
    matching how it was made trainable. We use ``jnp.full_like(template[i])``
    to get the right shape per entry (scalar for soma, 1-element array for
    sections with one branch, doesn't matter which as long as shape matches).
    """
    out: list[dict[str, jax.Array]] = []
    for i, spec in enumerate(PARAMS):
        tpl = template[i][spec.key]
        out.append({spec.key: jnp.full_like(tpl, phys[i])})
    return out


def make_forward(cell: jx.Cell, soma_comp: Any, template: list[dict[str, jax.Array]]):
    """Return forward(theta) -> somatic voltage trace.

    Closes over the cell, the stim, and the template parameter shape. The
    returned function is jit-able and grad-able end-to-end through
    jx.integrate.
    """
    stim = jx.step_current(I_DELAY, I_DUR, I_AMP, DT, T_MAX)

    def forward(theta: jax.Array) -> jax.Array:
        phys = theta_to_phys(theta, LO, HI)
        params = pack_params_from_phys(phys, template)
        ds = soma_comp.data_stimulate(stim, data_stimuli=None)
        v: jax.Array = jx.integrate(
            cell, params=params, delta_t=DT, t_max=T_MAX, data_stimuli=ds,
            checkpoint_lengths=list(CHECKPOINT_LENGTHS),
        )
        return v[0]  # (T,) somatic trace

    return forward


def loss_fn(v_sim: jax.Array, v_target: jax.Array) -> jax.Array:
    """Mean absolute error over the full somatic trace.

    Matches the Jaxley paper's synthetic fit loss. Smooth when init and
    target share spike counts; we pick INIT such that this holds (mild
    per-param perturbation).
    """
    return jnp.mean(jnp.abs(v_sim - v_target))


# Fit one restart (scan over Adam steps) and batch over P restarts (vmap)

def make_fit_one(forward, v_target: jax.Array, n_steps: int, lr: float,
                 grad_clip: float = 1.0):
    """Build a single-restart Adam fit loop.

    Three non-obvious behaviours to know about:

    1. Gradient clipping by global norm. Unbounded Adam steps can push
       theta into a regime where the biophysics integrates to NaN
       (instantaneous overcurrent, integrator blow-up in f32). Clipping
       the gradient's global norm to ``grad_clip`` caps the step size
       without interfering with convergence in the well-behaved region.
       optax.chain composes the clip with Adam in standard order.

    2. Best-so-far tracking. We keep (best_loss, best_theta) as extra
       scan carry, updated via jnp.where on each step. If a trajectory
       later hits NaN, we still return the pre-NaN best point. Cheaper
       than checkpointing, and it turns NaN from a failure into a
       graceful degradation. This also beats returning the final theta
       in the benign case (Adam sometimes overshoots after convergence).

    3. NaN-safe loss comparison. ``loss < best_loss`` is False when
       either side is NaN, so a NaN loss cannot win the argmin. We also
       replace a NaN ``loss`` with +inf in the carried trajectory so
       downstream argmin over the losses array stays clean.
    """
    opt = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(lr),
    )

    def fit_one(theta0: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Returns (theta_final, losses, best_theta, best_loss)."""
        state = opt.init(theta0)
        best_loss0 = jnp.asarray(jnp.inf, dtype=theta0.dtype)
        best_theta0 = theta0

        def step(carry, _):
            theta, state, best_loss, best_theta = carry
            loss, grad = jax.value_and_grad(
                lambda t: loss_fn(forward(t), v_target)
            )(theta)
            # Update best-seen with NaN-safe comparison.
            is_better = loss < best_loss  # False if loss is NaN
            best_loss = jnp.where(is_better, loss, best_loss)
            best_theta = jnp.where(is_better, theta, best_theta)
            # Adam + grad-clip update.
            updates, state = opt.update(grad, state)
            theta = optax.apply_updates(theta, updates)
            # Record +inf in the trajectory if the step blew up, so the
            # saved "losses" array has no NaN that would poison plotting.
            loss_recorded = jnp.where(jnp.isnan(loss), jnp.inf, loss)
            return (theta, state, best_loss, best_theta), loss_recorded

        (theta_f, _, best_loss, best_theta), losses = jax.lax.scan(
            step, (theta0, state, best_loss0, best_theta0),
            None, length=n_steps,
        )
        return theta_f, losses, best_theta, best_loss

    return fit_one


def sample_inits(key: jax.Array, p: int, k: int,
                 perturb_scale: float = 1.0) -> jax.Array:
    """Sample P initial theta vectors around THETA_TRUE.

    theta_init = THETA_TRUE + perturb_scale * N(0, 1).
    Small perturb keeps everyone in the same spike-count basin, matching
    the fit_demo.py lesson. For later scale-up we can increase this.
    """
    noise = jax.random.normal(key, (p, k))
    return THETA_TRUE[None, :] + perturb_scale * noise


# Main driver

def _build_tracker(experiment: str, run_name: str | None) -> TrackerProtocol:
    """Instantiate MLflowTracker if MLFLOW_TRACKING_URI is set, else NullTracker.

    Lets the script run unmodified locally (no server) and on the cluster
    (server reachable via injected env var) without conditional logic at the
    callsites.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("[tracker] MLFLOW_TRACKING_URI unset, using NullTracker")
        return NullTracker()
    print(f"[tracker] MLflowTracker uri={uri} experiment={experiment}")
    return MLflowTracker(experiment_name=experiment, tracking_uri=uri, run_name=run_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-restarts", type=int, default=16,
                    help="P = number of parallel Adam restarts (vmap axis).")
    ap.add_argument("--n-steps",    type=int, default=200,
                    help="Adam steps per restart.")
    ap.add_argument("--lr",         type=float, default=0.05,
                    help="Adam learning rate in theta space.")
    ap.add_argument("--perturb",    type=float, default=1.0,
                    help="Stdev of Gaussian init noise around THETA_TRUE (theta space).")
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--experiment", default="fit_bbp_multiparam",
                    help="MLflow experiment name (ignored if no tracking server).")
    ap.add_argument("--run-name",   default=None,
                    help="Optional MLflow run name.")
    args = ap.parse_args()

    print("jax", jax.__version__, "backend", jax.default_backend(),
          "devices", jax.devices())
    print(f"K = {K} parameters, P = {args.n_restarts} restarts, "
          f"N = {args.n_steps} Adam steps, lr = {args.lr}")

    tracker = _build_tracker(args.experiment, args.run_name)
    with tracker:
        # Print the run_id on its own line so the launcher's --retrieve
        # path can parse it deterministically from job logs.
        print(f"[tracker] mlflow_run_id={tracker.run_id}")
        _run_fit(args, tracker)


def _run_fit(args: argparse.Namespace, tracker: TrackerProtocol) -> None:

    # --- cell + target trace -------------------------------------------------
    t0 = time.perf_counter()
    cell, soma_comp = build_cell()
    ncomps = int(cell.nodes.shape[0])  # pyright: ignore[reportOptionalMemberAccess]
    template = cell.get_parameters()
    forward = make_forward(cell, soma_comp, template)
    print(f"Cell built: {ncomps} comps. Build + make_trainable: "
          f"{time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    v_target = cast(jax.Array, forward(THETA_TRUE))
    jax.block_until_ready(v_target)  # type: ignore[no-untyped-call]
    t_target = time.perf_counter() - t0
    print(f"Target trace: {v_target.shape}, integrate time {t_target:.2f}s")

    tracker.log_params({
        "n_restarts": args.n_restarts,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "perturb": args.perturb,
        "seed": args.seed,
        "K": K,
        "ncomps": ncomps,
        "dt": DT,
        "t_max": T_MAX,
        "v_init": V_INIT,
        "i_delay": I_DELAY,
        "i_dur": I_DUR,
        "i_amp": I_AMP,
        "checkpoint_lengths": list(CHECKPOINT_LENGTHS),
        "param_tags": [f"{p.section}.{p.key}" for p in PARAMS],
        **collect_environment_params(),
    })

    # --- fit (parallel restarts) --------------------------------------------
    fit_one = make_fit_one(forward, v_target, args.n_steps, args.lr)
    fit_parallel = jax.jit(jax.vmap(fit_one))

    key = jax.random.PRNGKey(args.seed)
    theta0s = sample_inits(key, args.n_restarts, K, perturb_scale=args.perturb)
    print(f"theta0s: {theta0s.shape}")

    # First call includes jit compile.
    print("Compiling + first run...")
    t0 = time.perf_counter()
    thetas, losses, best_thetas_p, best_losses_p = fit_parallel(theta0s)
    jax.block_until_ready(losses)  # type: ignore[no-untyped-call]
    t_first = time.perf_counter() - t0
    print(f"  first run (includes compile): {t_first:.2f}s")

    t0 = time.perf_counter()
    thetas, losses, best_thetas_p, best_losses_p = fit_parallel(theta0s)
    jax.block_until_ready(losses)  # type: ignore[no-untyped-call]
    t_cached = time.perf_counter() - t0
    print(f"  cached run: {t_cached:.2f}s  "
          f"({t_cached / (args.n_restarts * args.n_steps) * 1e3:.1f} ms / "
          f"forward+backward)")

    # --- pick best restart ------------------------------------------------
    # Use the best-seen-per-restart (across the trajectory) rather than the
    # final-step theta. Matters because (a) if a trajectory NaN'd after
    # converging, the final is garbage but best is good, and (b) Adam can
    # overshoot after convergence even in the benign case.
    best_losses_np = np.asarray(best_losses_p)
    best = int(np.argmin(best_losses_np))
    theta_best = np.asarray(best_thetas_p[best])
    phys_best = np.asarray(theta_to_phys(jnp.asarray(theta_best), LO, HI))
    phys_true = np.asarray(TRUE)

    final_losses = np.asarray(losses[:, -1])
    n_nan_trajectories = int(np.isinf(losses).any(axis=1).sum())
    if n_nan_trajectories > 0:
        print(f"  note: {n_nan_trajectories}/{args.n_restarts} restarts hit NaN "
              "at some point (grad-clipped; best-seen theta still used)")

    print("\nBest restart summary:")
    print(f"  restart idx: {best}   "
          f"best-seen loss: {float(best_losses_np[best]):.4f} mV   "
          f"final loss: {final_losses[best]:.4f} mV")

    metrics: dict[str, float] = {
        "best_loss": float(best_losses_np[best]),
        "final_loss": float(final_losses[best]),
        "best_restart_idx": float(best),
        "n_nan_trajectories": float(n_nan_trajectories),
        "t_target_integrate_s": float(t_target),
        "t_first_run_s": float(t_first),
        "t_cached_run_s": float(t_cached),
        "ms_per_fwd_bwd": float(t_cached / (args.n_restarts * args.n_steps) * 1e3),
    }
    for i, spec in enumerate(PARAMS):
        tag = f"{spec.section}.{spec.key}"
        metrics[f"recovered.{tag}"] = float(phys_best[i])
        metrics[f"true.{tag}"] = float(phys_true[i])
        if phys_true[i] != 0:
            metrics[f"err_pct.{tag}"] = float(
                (phys_best[i] - phys_true[i]) / phys_true[i] * 100
            )
    # MLflow rejects non-finite values. If a fit blows up entirely (best_loss=inf,
    # err_pct=NaN, etc.) we still want to log the partial run, so drop those keys.
    bad = [k for k, v in metrics.items() if not np.isfinite(v)]
    if bad:
        print(f"  note: dropping non-finite metrics from log: {bad}")
        for k in bad:
            del metrics[k]
    tracker.log_metrics(metrics)
    print(f"\n{'parameter':>24}{'true':>14}{'recovered':>14}{'err %':>10}")
    print("-" * 62)
    for i, spec in enumerate(PARAMS):
        e = (phys_best[i] - phys_true[i]) / phys_true[i] * 100 if phys_true[i] != 0 else float("nan")
        tag = f"{spec.section}.{spec.key}"
        print(f"{tag:>24}{phys_true[i]:>14.6g}{phys_best[i]:>14.6g}{e:>9.1f}%")

    # --- regenerate the best-restart trace for plotting ---------------------
    v_best = np.asarray(forward(jnp.asarray(theta_best)))

    # --- save everything -----------------------------------------------------
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fit_bbp_multiparam.npz"
    param_tags = np.array([f"{p.section}.{p.key}" for p in PARAMS])
    np.savez(
        out,
        # fit structure
        losses=np.asarray(losses),                       # (P, N_STEPS), +inf replaces NaN
        thetas_final=np.asarray(thetas),                 # (P, K)
        best_thetas=np.asarray(best_thetas_p),           # (P, K) best-seen per restart
        best_losses=best_losses_np,                      # (P,)    best-seen loss per restart
        final_losses=final_losses,                       # (P,)
        best_restart=np.array([best]),
        phys_best=phys_best,                             # (K,) best restart's best-seen phys
        phys_true=phys_true,                             # (K,)
        lo=np.asarray(LO), hi=np.asarray(HI),
        param_tags=param_tags,
        # traces
        v_target=np.asarray(v_target),                   # (T,)
        v_best=v_best,                                   # (T,)
        dt=DT, t_max=T_MAX, v_init=V_INIT,
        i_delay=I_DELAY, i_dur=I_DUR, i_amp=I_AMP,
        # metadata
        ncomps=ncomps,
        n_restarts=args.n_restarts,
        n_steps=args.n_steps,
        lr=args.lr,
        perturb=args.perturb,
        t_first=t_first, t_cached=t_cached,
        jax_backend=jax.default_backend(),
    )
    print(f"\nSaved {out}")
    tracker.log_artifact(out)


if __name__ == "__main__":
    main()
