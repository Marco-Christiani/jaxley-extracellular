#!/usr/bin/env bash
# One-shot bootstrap for installing Nix on a Cloud TPU VM.
#
# GCP TPU images ship with LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
#  in /etc/environment for JAX/XLA performance. The Nix install script and
#  nix-daemon binary segfault under that preload, so we disable it globally,
#  install Nix, then rely on the nix end to provide nix-built gperftools.
#
# Idempotent so re-running is a no-op once Nix is on PATH and /etc/environment
# is already patched.
#
# Run interactively on a fresh TPU VM:
#   bash infra/scripts/tpu-bootstrap-nix.sh
set -euo pipefail

if command -v nix >/dev/null 2>&1; then
  echo "[bootstrap] nix already installed: $(nix --version)"
else
  echo "[bootstrap] LD_PRELOAD before changes: ${LD_PRELOAD-<unset>}"

  # Patch /etc/environment so future logins (including nix-daemon's PAM
  # session) don't pull tcmalloc into nix processes. Backup once.
  if grep -qE '^[[:space:]]*LD_PRELOAD=.*libtcmalloc\.so\.4' /etc/environment; then
    sudo cp /etc/environment "/etc/environment.before-nix.$(date +%Y%m%d-%H%M%S)"
    sudo sed -i -E '/^[[:space:]]*LD_PRELOAD=.*libtcmalloc\.so\.4/s/^/# disabled for Nix bootstrap: /' /etc/environment
  fi
  unset LD_PRELOAD

  sudo env -u LD_PRELOAD apt-get update
  sudo env -u LD_PRELOAD apt-get install -y curl ca-certificates xz-utils

  env -u LD_PRELOAD bash -c 'curl -L https://nixos.org/nix/install | sh -s -- --daemon'

  # Source nix profile into this shell so the version check below works.
  if [ -r /etc/profile.d/nix.sh ]; then . /etc/profile.d/nix.sh; fi
  if [ -r /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh ]; then
    . /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh
  fi
  hash -r
fi

# Enable flakes + new CLI system-wide.
sudo mkdir -p /etc/nix
if sudo test -f /etc/nix/nix.conf && sudo grep -qE '^[[:space:]]*experimental-features[[:space:]]*=' /etc/nix/nix.conf; then
  sudo sed -i -E \
    's/^[[:space:]]*experimental-features[[:space:]]*=.*/experimental-features = nix-command flakes/' \
    /etc/nix/nix.conf
else
  printf '\nexperimental-features = nix-command flakes\n' | sudo tee -a /etc/nix/nix.conf >/dev/null
fi

# Trust the substituters declared in flake.nix nixConfig without per-invocation prompts.
if ! sudo grep -qE '^[[:space:]]*accept-flake-config[[:space:]]*=' /etc/nix/nix.conf; then
  printf 'accept-flake-config = true\n' | sudo tee -a /etc/nix/nix.conf >/dev/null
fi

# Mark the invoking user as trusted so they can use flake-declared substituters,
# add binary caches at runtime, and use the daemon for builds without prompts.
# Idempotent: only touches the line if our user isn't already listed.
TRUSTED_USER="${SUDO_USER:-$USER}"
if sudo grep -qE '^[[:space:]]*trusted-users[[:space:]]*=' /etc/nix/nix.conf; then
  if ! sudo grep -qE "^[[:space:]]*trusted-users[[:space:]]*=.*\b${TRUSTED_USER}\b" /etc/nix/nix.conf; then
    sudo sed -i -E "s/^([[:space:]]*trusted-users[[:space:]]*=.*)$/\1 ${TRUSTED_USER}/" /etc/nix/nix.conf
  fi
else
  printf 'trusted-users = root %s\n' "${TRUSTED_USER}" | sudo tee -a /etc/nix/nix.conf >/dev/null
fi

sudo systemctl restart nix-daemon

# Ray head systemd unit
# Installed once. Idempotent: writing the same bytes is a no-op for daemon-reload.
# References paths inside ~/jx-tpu-env, the symlink that the TPU worker bundle
# swap deploys. The unit will fail to start until that bundle is in place,
# which is expected on a fresh bootstrap.
#
# %h does not work here. For *system* services (units in /etc/systemd/system/)
# %h expands to root's home regardless of User=, because the manager that owns
# the unit is the system manager. Resolve TRUSTED_USER's actual home and bake
# absolute paths into the unit.
TRUSTED_HOME="$(getent passwd "$TRUSTED_USER" | cut -d: -f6)"
sudo tee /etc/systemd/system/ray-head.service >/dev/null <<UNIT
[Unit]
Description=Ray head (jaxley-extracellular)
After=network-online.target
Wants=network-online.target prometheus.service grafana-server.service

[Service]
Type=simple
User=${TRUSTED_USER}
WorkingDirectory=${TRUSTED_HOME}
Environment=PATH=${TRUSTED_HOME}/jx-tpu-env/bin:/usr/bin:/bin
Environment=JAX_PLATFORMS=tpu
# Persistent JAX compilation cache (workers fork from this daemon and inherit
# the env, so direct-bin/python and ray-spawned-worker paths share a cache).
Environment=JAX_COMPILATION_CACHE_DIR=${TRUSTED_HOME}/.jax_cache
Environment=JAX_COMPILATION_CACHE_MIN_SIZE_BYTES=0
Environment=JAX_COMPILATION_CACHE_MIN_COMPILE_TIME_SECS=1
Environment=RAY_CLIENT_SERVER_PORT=10001
Environment=RAY_DASHBOARD_PORT=8265
Environment=RAY_METRICS_EXPORT_PORT=8080
Environment=RAY_TPU_RESOURCE_COUNT=1
# LD_PRELOAD intentionally NOT set on the daemon. Ray subprocesses (e.g.,
# ray_client_server) spawn system bash for runtime_env handling, and that
# binary is linked against host glibc 2.31, which cannot load nix-built
# libtcmalloc (glibc 2.42+). tcmalloc activation lives in the bundle's
# bin/python wrapper instead. Workers pointed at that interpreter via
# runtime_env.py_executable get tcmalloc, while the daemon stays clean.
ExecStart=${TRUSTED_HOME}/jx-tpu-env/bin/ray-head-start
ExecStop=${TRUSTED_HOME}/jx-tpu-env/bin/ray stop --force
Restart=on-failure
RestartSec=5
# TPU libtpu mmaps large pinned regions; default RLIMIT_MEMLOCK (64K)
# fails with "Couldn't mmap: Resource temporarily unavailable".
LimitMEMLOCK=infinity
LimitNOFILE=1048576
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable ray-head

echo "[bootstrap] $(nix --version)"
echo "[bootstrap] systemd unit ray-head.service installed (not started)"
echo "[bootstrap] next: deploy the worker bundle to ~/jx-tpu-env, then start ray-head.service."
