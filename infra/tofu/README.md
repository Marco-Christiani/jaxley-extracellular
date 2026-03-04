# OpenTofu TPU Infra

This directory manages a Cloud TPU VM via OpenTofu using the Google provider.

## Initial setup

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <project-id>
gcloud auth application-default set-quota-project <project-id>
gcloud beta billing projects link <project-id> --billing-account <billing-account-id>
gcloud services enable compute.googleapis.com tpu.googleapis.com --project <project-id>
```

## Straight-line path

1) Configure variables:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your project/zone and valid TPU values. Discover zone-valid values with:

```bash
gcloud compute tpus accelerator-types list --zone <zone> --project <project-id>
gcloud compute tpus tpu-vm versions list --zone <zone> --project <project-id>
```

2) Provision TPU VM:

```bash
task tpu:init
task tpu:plan
task tpu:apply
```

3) Connect:

Note: `task tpu:ssh` requires state outputs, so run `task tpu:apply` first.

```bash
task tpu:ssh
```

4) On the TPU VM

Option 1: use uv directly (installed by bootstrap script)

```bash
uv --version
git clone https://github.com/Marco-Christiani/jaxley-extracellular.git
# or use task tpu:sync
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

```bash
task tpu:stop
task tpu:start
task tpu:destroy
```
