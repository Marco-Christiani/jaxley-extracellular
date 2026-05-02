#!/usr/bin/env bash
# Fetch all log artifacts for a given tag from the TPU VM into a local dir.
#
# Why: when a TPU run dies we want every breadcrumb at once. That means
# stdout/stderr, exit code, peak RSS (time -v), kernel ring buffer (dmesg),
# systemd journal, faulthandler dump, and the wrapper trace. tpu-run.sh already writes these to
# ~/jx-runs/<tag>.{log,exit,time,dmesg,journal,faulthandler,trace}.
# This script just rsyncs them locally so we can grep without ssh round-trips.
#
# Usage:
#   tpu-fetch-logs.sh <tag> [<local-dir>]
# Default local dir: .task/logs/<tag>/
#
# Exit code reflects whether logs were successfully fetched (not the run's RC).

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <tag> [<local-dir>]" >&2
  exit 2
fi

TAG="$1"
LOCAL_DIR="${2:-.task/logs/$TAG}"
mkdir -p "$LOCAL_DIR"

TPU_NAME="$(tofu -chdir=infra/tofu output -raw tpu_name)"
TPU_ZONE="$(tofu -chdir=infra/tofu output -raw zone)"

# Pull every file matching the tag prefix. -P keeps progress visible.
gcloud compute tpus tpu-vm scp --recurse \
  "$TPU_NAME:~/jx-runs/$TAG.*" \
  "$LOCAL_DIR/" \
  --zone "$TPU_ZONE" 2>&1 | tail -5

echo
echo "=== local files ==="
ls -la "$LOCAL_DIR"
echo
echo "=== exit summary ==="
cat "$LOCAL_DIR/$TAG.exit" 2>/dev/null || echo "(no exit file: run still in flight or never reached wrapper end)"
