# jaxley-extracellular

## Overview

```bash
# GPU by default
nix develop

# TPU
nix develop .#tpu
```

Run smoke tests:

```bash
jaxley-extracellular smoke-devices
jaxley-extracellular smoke-integrate
```

Run all checks:

```bash
nix flake check
```

## TPU infra

Infrastructure code for TPU VM lifecycle lives in `infra/tofu/` see [./infra/tofu/README.md](./infra/tofu/README.md) for instructions on provisioning and managing TPU instances.
