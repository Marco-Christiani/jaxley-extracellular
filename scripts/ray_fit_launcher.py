"""Ray launcher for scripts/fit_bbp_multiparam.py.

Async by default: submits the job to the head's dashboard job server and
returns immediately with a job id. The head owns the job's lifetime so it
survives client disconnects, conversation boundaries, and laptop sleep.

Usage:
    # submit, return immediately:
    python scripts/ray_fit_launcher.py \\
        -- --n-restarts 16 --n-steps 200 --perturb 0.3 --lr 0.02

    # submit + block until done + pull artifact:
    python scripts/ray_fit_launcher.py --wait ...

    # status / logs / retrieve previously submitted jobs:
    python scripts/ray_fit_launcher.py --status JOB_ID
    python scripts/ray_fit_launcher.py --logs JOB_ID
    python scripts/ray_fit_launcher.py --retrieve JOB_ID

The fit script writes ``results/fit_bbp_multiparam.npz`` on the worker.
``--retrieve`` pulls the npz back via the MLflow run's artifact (so we
don't depend on a live ray.get; artifacts persist on the tracking
server even if the head dies).

Why dashboard job server (not ray.init + ray.get):
    `ray.init(ray://...)` ties the job's lifetime to the client process.
    If the client dies, the worker is killed. The dashboard job server
    runs on the head and owns its own lifetime; the client is just a
    submission proxy. Required for any unattended run.

Endpoints are not hardcoded:
    --dashboard-url and MLFLOW_TRACKING_URI are resolved from `tofu output`
    when not provided, so a tofu apply that re-IPs anything Just Works.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAST_JOB_FILE = PROJECT_ROOT / ".task" / "last_ray_job.json"
TOFU_DIR = PROJECT_ROOT / "infra" / "tofu"


def _tofu_output(name: str) -> str | None:
    """Return `tofu output -raw <name>` or None if tofu/state isn't available."""
    try:
        return subprocess.check_output(
            ["tofu", f"-chdir={TOFU_DIR}", "output", "-raw", name],
            stderr=subprocess.DEVNULL, text=True,
        ).strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def resolve_dashboard_url(local_port: str = "48265") -> str:
    """Tunnel-side dashboard URL. Local port from the LOCAL_RAY_DASHBOARD_PORT
    taskfile var; we don't read the taskfile, so the user can override via
    --dashboard-url or RAY_DASHBOARD_LOCAL_PORT.
    """
    p = os.environ.get("RAY_DASHBOARD_LOCAL_PORT", local_port)
    return f"http://127.0.0.1:{p}"


def resolve_mlflow_uri() -> str | None:
    """Worker's MLflow URI: VPC-internal address from tofu output. Returns
    None if tofu state has no tracking server (apply hasn't run, or stack
    deployed without tracking).
    """
    if v := os.environ.get("MLFLOW_TRACKING_URI"):
        return v
    return _tofu_output("tracking_server_internal_uri")


def build_runtime_env(working_dir: str, py_executable: str | None) -> dict:
    """Job-specific env. tcmalloc activation lives in the bundle's bin/python
    wrapper; pointing py_executable at it gets workers tcmalloc-active
    without LD_PRELOAD on the head daemon (which would poison ray's system
    bash subprocesses). JAX_PLATFORMS lives in the head's systemd unit.
    Workers inherit it via fork.

    Excludes come from `.gitignore` (Ray honors it once we pop the
    RAY_RUNTIME_ENV_IGNORE_GITIGNORE override in cmd_submit). We do not pass
    an explicit excludes list. When we tried, it produced empty uploads.
    Relying on gitignore + Ray's defaults (.git/, etc) is enough.
    """
    env_vars = {
        "JAX_ENABLE_X64": "1",  # f64 by default
        # JAX persistent compilation cache lives in the worker's home.
        # The worker side resolves "~" so this picks up the worker's HOME,
        # not the submitter's.
        "JAX_COMPILATION_CACHE_DIR": "~/.jax_cache",
        "JAX_COMPILATION_CACHE_MIN_SIZE_BYTES": "0",
        "JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS": "1",
    }
    if mlflow_uri := resolve_mlflow_uri():
        env_vars["MLFLOW_TRACKING_URI"] = mlflow_uri

    runtime: dict = {
        "working_dir": working_dir,
        "env_vars": env_vars,
        # Belt-and-suspenders excludes for paths .rayignore doesn't cover.
        # Add any local state dirs to .rayignore on the submitting host.
        "excludes": [".task", ".direnv", ".venv", ".git"],
    }
    if py_executable:
        runtime["py_executable"] = py_executable
    return runtime


def _client(dashboard_url: str):
    from ray.job_submission import JobSubmissionClient
    return JobSubmissionClient(dashboard_url)


def cmd_submit(args, fit_args: list[str]) -> str:
    # Ray's working_dir packager applies BOTH .gitignore and .rayignore when
    # include_gitignore is True. Our .gitignore has `.*` (line 209) which
    # pathspec interprets as matching the path "." (the working_dir root)
    # itself, short-circuiting traversal and producing an empty package.
    # Force include_gitignore=False so only .rayignore is consulted; that
    # file already excludes results/, reference/, tests/, etc explicitly.
    os.environ["RAY_RUNTIME_ENV_IGNORE_GITIGNORE"] = "1"

    client = _client(args.dashboard_url)
    runtime_env = build_runtime_env(args.working_dir, args.py_executable)
    fit_args_str = " ".join(f'"{a}"' if " " in a else a for a in fit_args)
    entrypoint = f"python {args.script} {fit_args_str}"
    print(f"[launcher] submit dashboard={args.dashboard_url}")
    print(f"[launcher] entrypoint: {entrypoint}")
    job_id = client.submit_job(
        entrypoint=entrypoint,
        runtime_env=runtime_env,
        entrypoint_num_gpus=args.num_gpus,
        entrypoint_resources={"TPU": args.num_tpus} if args.num_tpus > 0 else None,
    )
    print(f"[launcher] submitted job_id={job_id}")
    LAST_JOB_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_JOB_FILE.write_text(json.dumps({
        "job_id": job_id,
        "dashboard_url": args.dashboard_url,
        "fit_args": fit_args,
        "submitted_at": time.time(),
    }, indent=2))
    return job_id


def cmd_status(dashboard_url: str, job_id: str) -> None:
    client = _client(dashboard_url)
    info = client.get_job_info(job_id)
    print(f"job_id     : {job_id}")
    print(f"status     : {info.status}")
    print(f"start_time : {info.start_time}")
    print(f"end_time   : {info.end_time}")
    if info.message:
        print(f"message    : {info.message}")


def cmd_logs(dashboard_url: str, job_id: str) -> None:
    client = _client(dashboard_url)
    print(client.get_job_logs(job_id))


def cmd_wait(dashboard_url: str, job_id: str, poll_s: int = 30) -> str:
    """Block until terminal status. Returns final status."""
    from ray.job_submission import JobStatus
    client = _client(dashboard_url)
    terminal = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}
    last = None
    while True:
        info = client.get_job_info(job_id)
        if info.status != last:
            print(f"[launcher] {time.strftime('%H:%M:%S')} status={info.status}")
            last = info.status
        if info.status in terminal:
            return str(info.status)
        time.sleep(poll_s)


def cmd_retrieve(dashboard_url: str, job_id: str, out: Path) -> None:
    """Best-effort: pull the run's npz from MLflow. Falls back to printing the
    job's stdout if MLflow lookup fails so we at least have something on disk.
    """
    client = _client(dashboard_url)
    info = client.get_job_info(job_id)
    if info.status != "SUCCEEDED":
        print(f"[launcher] job did not succeed (status={info.status}); skipping retrieve")
        return
    # Pull stdout to a sibling log so we always have a record.
    log_path = out.with_suffix(".log")
    log_path.write_text(client.get_job_logs(job_id))
    print(f"[launcher] wrote {log_path}")

    # MLflow round-trip. Since this is the canonical artifact path, prefer it
    # over scp'ing from the worker. Artifacts persist on the tracking server
    # regardless of TPU lifetime. Requires MLFLOW_TRACKING_URI in the env.
    # If unset, we skip the round-trip and leave the user with the job log.
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_uri:
        print("[launcher] MLFLOW_TRACKING_URI unset; skipping artifact download. "
              f"Job log is at {log_path}.")
        return
    try:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)
        # Job stdout will contain a line like "MLflow run_id=...". Naive parse.
        # If we miss it, the user can re-pull manually with the run_id.
        log_text = log_path.read_text()
        run_id = None
        for line in log_text.splitlines():
            if "mlflow_run_id=" in line:
                run_id = line.rsplit("mlflow_run_id=", 1)[-1].strip()
                break
        if run_id:
            print(f"[launcher] mlflow run_id={run_id}; downloading artifacts...")
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path="fit_bbp_multiparam.npz",
                dst_path=str(out.parent),
            )
            print(f"[launcher] artifact -> {local}")
        else:
            print("[launcher] could not parse MLflow run_id from job logs; "
                  "open the MLflow UI and download manually.")
    except Exception as e:
        print(f"[launcher] mlflow retrieve failed ({e}); job log is at {log_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dashboard-url", default=None,
                   help=("Ray dashboard URL (default: tunnel-side, port from "
                         "RAY_DASHBOARD_LOCAL_PORT or 48265)."))
    p.add_argument("--working-dir", default=str(PROJECT_ROOT))
    p.add_argument("--script", default="scripts/fit_bbp_multiparam.py",
                   help="Entrypoint script path relative to working_dir.")
    p.add_argument(
        "--py-executable",
        default=os.path.expanduser("~/jx-tpu-env/bin/python"),
        help=("Worker python (the bundle's tcmalloc-wrapping interpreter). "
              "Pass empty string to inherit head's python instead."),
    )
    p.add_argument("--num-gpus", type=float, default=0.0)
    p.add_argument("--num-tpus", type=float, default=1.0)
    p.add_argument("--out", default=str(PROJECT_ROOT / "results" / "fit_bbp_multiparam.npz"))

    g = p.add_mutually_exclusive_group()
    g.add_argument("--status", metavar="JOB_ID")
    g.add_argument("--logs", metavar="JOB_ID")
    g.add_argument("--retrieve", metavar="JOB_ID")
    g.add_argument("--last", action="store_true",
                   help="Apply --status to the most recently submitted job.")
    p.add_argument("--wait", action="store_true",
                   help="With --submit (the default), block until terminal status.")
    p.add_argument("--poll-s", type=int, default=30)
    p.add_argument("fit_args", nargs=argparse.REMAINDER)
    args = p.parse_args()

    if args.dashboard_url is None:
        args.dashboard_url = resolve_dashboard_url()

    if args.status or args.logs or args.retrieve or args.last:
        job_id = args.status or args.logs or args.retrieve
        if args.last:
            if not LAST_JOB_FILE.exists():
                print(f"no record at {LAST_JOB_FILE}", file=sys.stderr)
                sys.exit(2)
            data = json.loads(LAST_JOB_FILE.read_text())
            job_id = data["job_id"]
            args.dashboard_url = data.get("dashboard_url", args.dashboard_url)
            print(f"[launcher] last job: {job_id} ({data.get('submitted_at')})")
        if args.status or args.last:
            cmd_status(args.dashboard_url, job_id)
        elif args.logs:
            cmd_logs(args.dashboard_url, job_id)
        elif args.retrieve:
            cmd_retrieve(args.dashboard_url, job_id, Path(args.out))
        return

    fit_args = args.fit_args
    if fit_args and fit_args[0] == "--":
        fit_args = fit_args[1:]
    job_id = cmd_submit(args, fit_args)

    if args.wait:
        status = cmd_wait(args.dashboard_url, job_id, poll_s=args.poll_s)
        print(f"[launcher] terminal status: {status}")
        if status == "SUCCEEDED":
            cmd_retrieve(args.dashboard_url, job_id, Path(args.out))


if __name__ == "__main__":
    main()
