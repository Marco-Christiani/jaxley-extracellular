# OpenTofu Infrastructure

Declarative infrastructure for the project's compute and experiment tracking resources.

## Prerequisites

```bash
gcloud auth login
gcloud auth application-default login
```

Pick a project name, a zone, and get a billing account id then set these for convenience while running `gcloud` commands for initial setup:

```bash
export GCP_PROJECT_ID="..."
export GCP_BILLING_ACCT_ID="..."
export GCP_ZONE="..."
```

```bash
gcloud config set project "$GCP_PROJECT_ID"
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
gcloud beta billing projects link "$GCP_PROJECT_ID" --billing-account "$GCP_BILLING_ACCT_ID"
gcloud services enable \
  compute.googleapis.com \
  tpu.googleapis.com \
  iap.googleapis.com \
  secretmanager.googleapis.com \
  sqladmin.googleapis.com \
  --project "$GCP_PROJECT_ID"
```

## Resources

### TPU VM (`enable_tpu`)

The core compute resource. Startup script installs `uv`. Discover zone-valid values with:

```bash
gcloud compute tpus accelerator-types list --zone <zone> --project "$GCP_PROJECT_ID"
gcloud compute tpus tpu-vm versions list --zone <zone> --project "$GCP_PROJECT_ID"
```

The TPU is opt-out. To provision the tracking stack alone:

```bash
tofu -chdir=infra/tofu apply -var="enable_tpu=false"
```

### Experiment tracking

Three resources that compose into the tracking stack, all opt-in (enabled by default):

- **Cloud SQL** (`enable_tracking_db`): Postgres 16 backend store.
- **GCS bucket** (`enable_artifact_bucket`): artifact storage with configurable retention.
- **Tracking server** (`enable_tracking_server`): dedicated GCE instance running the tracking server.

The tracking server depends on both Cloud SQL and GCS (enforced by `lifecycle { precondition }` blocks). It runs under a dedicated service account with minimal IAM:

- `roles/secretmanager.secretAccessor`: DB password fetched from Secret Manager at boot.
- `roles/storage.objectAdmin`: artifact read/write on the GCS bucket.

The DB password never appears in instance metadata or systemd unit files. A launcher script fetches it from Secret Manager at boot, constructs the connection URI in memory, and `exec`s the server.

The server is configured with `--serve-artifacts`, so sweep clients only need `--tracking-uri`. Artifact uploads are proxied through the tracking server to GCS rather than requiring direct GCS credentials on each worker.

The tracking server is configurable via three variables:

- `tracking_server_package`: package to install via `uv tool install` (default: `mlflow[db]>=2.12`).
- `tracking_server_command`: binary name (default: `mlflow`).
- `tracking_server_args`: full argument string. Three env vars are available for substitution: `$DB_URI`, `$ARTIFACT_ROOT`, `$PORT`.

### IAP access

The tracking server is not publicly reachable. Access is via Identity-Aware Proxy (IAP) tunnel, authenticated by your Google account. Grant access in `terraform.tfvars`:

```hcl
iap_users = ["user:you@gmail.com"]
```

## Known operational quirks

Read before bringing up or tearing down the stack.

- **Cloud SQL teardown via `tofu destroy` always fails after MLflow has
  populated the tracking schema.** The `tracker` SQL user owns ~48 mlflow
  tables, and tofu tries to drop the user before the instance. Symptom:
  `role "tracker" cannot be dropped because some objects depend on it`
  plus `database "tracking" is being accessed by other users`. The
  user-owns-tables condition is structural, not a connection-timeout
  issue, so waiting will not help. Recovery:
  ```bash
  gcloud sql instances delete jx-tpu-dev-tracking --project=$GCP_PROJECT_ID --quiet
  tofu -chdir=infra/tofu state rm \
       google_sql_database_instance.tracking[0] \
       google_sql_user.tracking[0] \
       google_sql_database.tracking[0]
  ```
  Cleaner long-term fix: pre-run `DROP OWNED BY tracker` against the DB
  before destroy.
- **The tracking server can't proxy artifact uploads to GCS.** Its
  google-auth library refreshes a GCE service account token via
  `https://metadata.google.internal:443`, whose TLS cert is signed by
  Google's *internal* CA, which is not in any standard trust bundle
  (system or certifi). MLflow returns 500 to clients on `PUT
  /api/2.0/mlflow-artifacts/.../*.npz`. Worked around by changing
  `tracking-server-start`'s `ARTIFACT_ROOT` to `/var/lib/mlflow-artifacts`
  (local disk). For paper-critical durability in GCS, copy files directly
  with `gsutil cp` (uses gcloud auth, not Python google-auth). Real fix
  would install Google's internal CA on the tracker, or pin google-auth
  to a version that uses HTTP for metadata refresh.
- **Cloud SQL boot is the long pole on apply** (~10-12 min before the
  instance reaches `RUNNABLE`). `tofu apply` blocks until the instance is up.

## Provisioning

1. Copy and fill in variables:

```bash
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars
```

2. Set the DB password (never commit this):

```bash
export TF_VAR_tracking_db_password="..."
```

3. Provision:

```bash
task infra:init
task infra:plan
task infra:apply
```

## Key outputs

```bash
tofu -chdir=infra/tofu output -raw tpu_name                     # TPU VM name
tofu -chdir=infra/tofu output -raw tracking_server_internal_uri # VPC-internal (for TPU workers)
tofu -chdir=infra/tofu output -raw tracking_server_external_uri # external IP (IAP tunnel target)
tofu -chdir=infra/tofu output -raw artifact_bucket_uri          # gs://...
```

## TPU runtime

Two runtime paths are supported on the TPU VM:

### Preferred: Nix

Bootstrap Nix once, then use the dedicated TPU dev shell:

```bash
task remote:tpu:sync
task remote:tpu:bootstrap:nix
task remote:tpu:ssh
# on the TPU VM:
cd ~/jaxley-extracellular
nix develop .#tpu
python -m jaxley_extracellular.cli smoke-tpu
python -m jaxley_extracellular.cli smoke-integrate
```

The bootstrap script handles the TPU image's `LD_PRELOAD` conflict and enables
flake support system-wide.

### Fallback: uv

If you do not want to rely on Nix on the TPU VM, `uv` remains a supported
fallback for direct/manual execution:

```bash
task remote:tpu:sync
task remote:tpu:ssh
# on the TPU VM:
cd ~/jaxley-extracellular
uv sync --frozen --group tpu --group tracking --group dev
. .venv/bin/activate
jaxley-extracellular smoke-tpu
jaxley-extracellular smoke-integrate
```

Policy:
- direct/manual TPU runs: Nix preferred, `uv` supported
- Ray-based runs: treat Ray as optional and configure it explicitly; do not assume "TPU up" implies "Ray ready"

## Standard path: tracking only (no Ray)

This is the baseline reproducible path for experiment tracking:

1. Provision infra:

```bash
task infra:init
task infra:plan
task infra:apply
```

2. Verify tracking stack health:

```bash
task remote:tracking:health
task remote:tracking:logs
```

3. Tunnel the MLflow UI to a non-conflicting local port if needed:

```bash
task remote:tracking:tunnel:bg LOCAL_TRACKING_PORT=45137
```

4. Open the UI locally:

```text
http://127.0.0.1:45137
```

Notes:
- TPU/worker jobs should log to `tracking_server_internal_uri`
- artifacts flow through the tracking server to `artifact_bucket_uri`
- the tracking VM is the only component that needs DB + GCS credentials

## Optional Ray path

Ray is opt-in. The recommended networking/observability contract is:

- Ray client: `10001`
- Ray dashboard: `8265`
- Ray metrics export: `8080`
- Grafana: `3000`
- Prometheus: `9090`

1. Start the Ray head on the TPU VM:

```bash
task remote:tpu:ray:head:start
```

2. Bootstrap Grafana + Prometheus on the TPU VM:

```bash
task remote:tpu:observability:bootstrap
task remote:tpu:observability:refresh
task remote:tpu:observability:health
```

3. Verify Ray head health:

```bash
task remote:tpu:ray:head:status
```

4. Tunnel Ray + observability + MLflow to local ports:

```bash
task remote:tunnels:up \
  LOCAL_RAY_CLIENT_PORT=46101 \
  LOCAL_RAY_DASHBOARD_PORT=48265 \
  LOCAL_GRAFANA_PORT=43000 \
  LOCAL_PROMETHEUS_PORT=49090 \
  LOCAL_TRACKING_PORT=45137
```

5. Access locally:

```text
ray://127.0.0.1:46101
http://127.0.0.1:48265   # Ray dashboard
http://127.0.0.1:43000   # Grafana
http://127.0.0.1:49090   # Prometheus
http://127.0.0.1:45137   # MLflow
```

6. Tear down local tunnels when done:

```bash
task remote:tunnels:down
```

Notes:
- Ray dashboard metrics require Grafana + Prometheus; a running Ray head alone is not sufficient
- if local ports collide with another project, override them in `task remote:tunnels:up`
- `task remote:tunnels:status` shows the tracked tunnel PIDs and local listeners

## Taskfile

All operations are available via `task`. See `task --list` for the full list.

| Namespace | Tasks | Description |
|-----------|-------|-------------|
| `infra:*` | `init`, `plan`, `apply`, `destroy` | OpenTofu lifecycle (all resources) |
| `remote:tpu:*` | `up`, `down`, `start`, `stop`, `ssh`, `sync`, `bootstrap:nix`, `ray:*`, `observability:*` | TPU provisioning, runtime, optional Ray, optional Ray observability |
| `remote:tracking:*` | `up`, `down`, `ssh`, `logs`, `health`, `tunnel:bg` | Tracking server lifecycle and access |
| `remote:tunnels:*` | `up`, `down`, `status` | Local tunnels for Ray, Grafana, Prometheus, and MLflow |
| `local:*` | `db`, `db:stop`, `db:container`, `db:container:stop`, `server`, `ui` | Local dev tracking stack |

Key distinctions:

- `infra:apply` / `infra:destroy`: tofu touches all resources (destructive, prompts for confirmation).
- `remote:tpu:up` / `remote:tpu:down`: tofu touches only the TPU resource.
- `remote:tpu:start` / `remote:tpu:stop`: gcloud pause/resume (no infra change).
- `remote:tracking:up` / `remote:tracking:down`: gcloud start/stop the GCE instance.
- `remote:tracking:tunnel:bg`: background IAP tunnel for MLflow UI.
- `remote:tpu:ray:*` / `remote:tpu:observability:*`: only needed if you are using Ray.
