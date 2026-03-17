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

Note: After this point, once you have OpenTofu set up and values in sync in `terraform.tfvars` in later step it is better to derive the values to avoid mistakes using commands like `tofu -chdir=infra/tofu output -raw name` shown later.

```bash
gcloud config set project "$GCP_PROJECT_ID"
gcloud auth application-default set-quota-project "$GCP_PROJECT_ID"
gcloud beta billing projects link "$GCP_PROJECT_ID" --billing-account "$GCP_BILLING_ACCT_ID"
# Make sure we enable access the GCP services we will be using
gcloud services enable \
  compute.googleapis.com \
  tpu.googleapis.com \
  iap.googleapis.com \
  secretmanager.googleapis.com \
  iap.googleapis.com \
  --project "$GCP_PROJECT_ID"
gcloud services enable  --project="$GCP_PROJECT_ID"
```

## Resources

All resources are configured in `terraform.tfvars`. Tracking resources are opt-in (disabled by default).

### TPU VM

The core compute resource. Startup script installs `uv`. Discover zone-valid values with:

```bash
gcloud compute tpus accelerator-types list --zone <zone> --project "$GCP_PROJECT_ID"
gcloud compute tpus tpu-vm versions list --zone <zone> --project "$GCP_PROJECT_ID"
```

Opt out (e.g., if you just want the tracking server) by flipping `enable_tpu` in in `terraform.tfvars` and ad-hoc with the CLI:

```bash
tofu -chdir=infra/tofu apply -var="enable_tpu=false"
```

### Experiment tracking

Three resources that compose into the tracking stack:

- **Cloud SQL** (`enable_tracking_db`) -- Postgres 16 for the tracking backend store
- **GCS bucket** (`enable_artifact_bucket`) -- artifact storage with configurable retention
- **Tracking server** (`enable_tracking_server`) -- dedicated GCE instance (default: MLflow, configurable via `tracking_server_package` / `tracking_server_command`)

The tracking server depends on both Cloud SQL and GCS (enforced by `lifecycle { precondition }` blocks). It runs under a dedicated service account with minimal IAM:

- `roles/secretmanager.secretAccessor` -- DB password fetched at runtime from Secret Manager
- `roles/storage.objectAdmin` -- artifact read/write on the GCS bucket

The DB password never appears in instance metadata or systemd unit files. A launcher script fetches it from Secret Manager at boot, constructs the connection URI in memory, and `exec`s the server binary.

## Stand up cloud infrastructure

1) Configure variables:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your project-id, billing-account-id, zone, etc that you set up in the prerequisites step (the comments document the settings, jointly with the descriptions in `variables.tf`). Can adjust TPU instance types, image, etc if desired (see comments).

2) Provision:

```bash
tofu -chdir=infra/tofu init
tofu -chdir=infra/tofu plan
tofu -chdir=infra/tofu apply
```

3) Connect:

```bash
gcloud compute tpus tpu-vm ssh "$(tofu -chdir=infra/tofu output -raw name)" \
  --zone "$(tofu -chdir=infra/tofu output -raw zone)"
```

4) Tracking is enabled by default. Provide the DB password before applying:

```bash
export TF_VAR_tracking_db_password="..."
tofu -chdir=infra/tofu apply
```

The three flags (`enable_tracking_db`, `enable_artifact_bucket`, `enable_tracking_server`) can be set to `false` individually in `terraform.tfvars` to disable components.

Key outputs:

```bash
tofu -chdir=infra/tofu output -raw name                          # TPU VM name
tofu -chdir=infra/tofu output -raw tracking_server_url            # internal (VPC)
tofu -chdir=infra/tofu output -raw tracking_server_external_url   # external
tofu -chdir=infra/tofu output -raw artifact_bucket_uri            # gs://...
```

## On the TPU VM

Option 1: use uv directly (installed by bootstrap script)

```bash
uv --version
git clone https://github.com/Marco-Christiani/jaxley-extracellular.git
# or sync tracked files: task tpu:sync
cd jaxley-extracellular
uv sync --frozen --group tpu --group dev
. .venv/bin/activate
jaxley-extracellular smoke-tpu
jaxley-extracellular smoke-integrate
```

Option 2: use nix infra

Generally getting libunwind errors during nix install on TPU instances so either use `uv` bare metal or drop into a nix container (proper TPU passthrough requires docker extra flags).

```
# This was the known-good combo (pid, ulimit, and security-opt might warrant ablation tests):
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

## Lifecycle

TODO: some tofu output variable names are out of sync

```bash
tofu -chdir=infra/tofu apply    # create / update
tofu -chdir=infra/tofu destroy  # tear down all resources
```

TPU start/stop (without destroying):

```bash
gcloud compute tpus tpu-vm stop "$(tofu -chdir=infra/tofu output -raw name)" \
  --zone "$(tofu -chdir=infra/tofu output -raw zone)"
gcloud compute tpus tpu-vm start "$(tofu -chdir=infra/tofu output -raw name)" \
  --zone "$(tofu -chdir=infra/tofu output -raw zone)"
```

Tracking server start/stop:

```bash
gcloud compute instances start "$(tofu -chdir=infra/tofu output -raw name)-tracking" \
  --zone "$(tofu -chdir=infra/tofu output -raw zone)"
gcloud compute instances stop "$(tofu -chdir=infra/tofu output -raw name)-tracking" \
  --zone "$(tofu -chdir=infra/tofu output -raw zone)"
```

## Task aliases

`taskfile.yml` provides convenience aliases. These wrap the tofu/gcloud commands above:

| Alias | Wraps |
|-------|-------|
| `task tpu:init` | `tofu init` |
| `task tpu:plan` | `tofu plan` |
| `task tpu:apply` | `tofu apply` |
| `task tpu:destroy` | `tofu destroy` |
| `task tpu:ssh` | `gcloud compute tpus tpu-vm ssh` |
| `task tpu:start` | `gcloud compute tpus tpu-vm start` |
| `task tpu:stop` | `gcloud compute tpus tpu-vm stop` |
| `task tpu:sync` | `git ls-files` + `tar` + `gcloud ssh` pipe |
| `task tracking:db` | local Postgres via nix |
| `task tracking:db:stop` | stop local Postgres |
| `task tracking:db:container` | local Postgres via docker/podman |
| `task tracking:server` | local tracking server (Postgres + local artifacts) |
| `task tracking:remote:start` | `gcloud compute instances start` (tracking server) |
| `task tracking:remote:stop` | `gcloud compute instances stop` (tracking server) |
| `task tracking:remote:ssh` | `gcloud compute ssh` (tracking server) |
| `task tracking:remote:logs` | `journalctl -u tracking-server -f` via SSH |
| `task tracking:remote:url` | print internal + external tracking URLs |
| `task tracking:ui` | open local tracking UI in browser |
| `task tracking:remote:ui` | open remote tracking UI in browser |
