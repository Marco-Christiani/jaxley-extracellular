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

The TPU is opt-out -- provision the tracking stack alone with:

```bash
tofu -chdir=infra/tofu apply -var="enable_tpu=false"
```

### Experiment tracking

Three resources that compose into the tracking stack, all opt-in (enabled by default):

- **Cloud SQL** (`enable_tracking_db`) -- Postgres 16 backend store
- **GCS bucket** (`enable_artifact_bucket`) -- artifact storage with configurable retention
- **Tracking server** (`enable_tracking_server`) -- dedicated GCE instance running the tracking server

The tracking server depends on both Cloud SQL and GCS (enforced by `lifecycle { precondition }` blocks). It runs under a dedicated service account with minimal IAM:

- `roles/secretmanager.secretAccessor` -- DB password fetched from Secret Manager at boot
- `roles/storage.objectAdmin` -- artifact read/write on the GCS bucket

The DB password never appears in instance metadata or systemd unit files. A launcher script fetches it from Secret Manager at boot, constructs the connection URI in memory, and `exec`s the server.

The server is configured with `--serve-artifacts`, so sweep clients only need `--tracking-uri` -- artifact uploads are proxied through the tracking server to GCS rather than requiring direct GCS credentials on each worker.

The tracking server is configurable via three variables:

- `tracking_server_package` -- package to install via `uv tool install` (default: `mlflow[db]>=2.12`)
- `tracking_server_command` -- binary name (default: `mlflow`)
- `tracking_server_args` -- full argument string; three env vars are available for substitution: `$DB_URI`, `$ARTIFACT_ROOT`, `$PORT`

### IAP access

The tracking server is not publicly reachable. Access is via Identity-Aware Proxy (IAP) tunnel, authenticated by your Google account. Grant access in `terraform.tfvars`:

```hcl
iap_users = ["user:you@gmail.com"]
```

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

## On the TPU VM

Option 1: use uv directly (installed by bootstrap script)

```bash
uv --version
git clone https://github.com/Marco-Christiani/jaxley-extracellular.git
# or sync tracked files: task remote:tpu:sync
cd jaxley-extracellular
uv sync --frozen --group tpu --group tracking --group dev
. .venv/bin/activate
jaxley-extracellular smoke-tpu
jaxley-extracellular smoke-integrate
# optionally run the test suite (tpu for impatient)
# uv run --group dev pytest
```

Option 2: use nix infra

Obviously, strongly preferred when possible.

Generally getting libunwind errors during nix install on TPU instances so either use `uv` bare metal or drop into a nix container (proper TPU passthrough requires docker extra flags).

```bash
# Known-good combo (pid, ulimit, and security-opt might warrant ablation tests):
docker run --rm -it \
  --name nixu \
  --privileged \
  --net=host \
  --ipc=host \
  --pid=host \
  --ulimit memlock=-1 \
  --security-opt seccomp=unconfined \
  -v ~/jaxley-extracellular:/jaxley-extracellular \
  -w /jaxley-extracellular \
  -e USER_UID="$(id -u)" \
  -e USER_GID="$(id -g)" \
  -e USER_NAME="$USER" \
  -e EP_PLUGINS="user nix-daemon" \
  marcochristiani/nixu-hm
```

## Taskfile

All operations are available via `task`. See `task --list` for the full list.

| Namespace | Tasks | Description |
|-----------|-------|-------------|
| `infra:*` | `init`, `plan`, `apply`, `destroy` | OpenTofu lifecycle (all resources) |
| `remote:tpu:*` | `up`, `down`, `start`, `stop`, `ssh`, `sync` | TPU provisioning and access |
| `remote:tracking:*` | `up`, `down`, `ui`, `ssh`, `logs` | Tracking server lifecycle and access |
| `local:*` | `db`, `db:stop`, `db:container`, `db:container:stop`, `server`, `ui` | Local dev tracking stack |

Key distinctions:

- `infra:apply` / `infra:destroy` -- tofu touches all resources (destructive, prompts for confirmation)
- `remote:tpu:up` / `remote:tpu:down` -- tofu touches only the TPU resource
- `remote:tpu:start` / `remote:tpu:stop` -- gcloud pause/resume (no infra change)
- `remote:tracking:up` / `remote:tracking:down` -- gcloud start/stop the GCE instance
- `remote:tracking:ui` -- IAP tunnel to `localhost:5000`
