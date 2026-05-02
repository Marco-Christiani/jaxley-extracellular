#!/usr/bin/env bash
# Reliable launcher for long-running python jobs on the TPU VM.
#
# Why this exists: tmux + tee + ssh quoting kept dropping output, and silent
# crashes (OOM-killer, signal kills, libtpu aborts) had no traces. This wrapper
# forces every signal source to leave a record on disk:
#   - stdout/stderr -> ${LOG_DIR}/${TAG}.log (fully written, no pty truncation)
#   - exit code + signal name -> ${LOG_DIR}/${TAG}.exit
#   - peak RSS, max wallclock, page faults via /usr/bin/time -> ${LOG_DIR}/${TAG}.time
#   - kernel ring buffer at exit (OOM-killer, segfaults) -> ${LOG_DIR}/${TAG}.dmesg
#   - systemd journal slice for the run window -> ${LOG_DIR}/${TAG}.journal
#   - the launching script itself runs under `set -x` (trace) into ${TAG}.trace
#
# Usage:
#   tpu-run.sh <tag> <command...>
# Example:
#   tpu-run.sh validate JAX_ENABLE_X64=1 ~/jx-tpu-env/bin/python scripts/sweep.py ...
#
# Run the resulting process under tmux for detachment:
#   tmux new-session -d -s validate "infra/scripts/tpu-run.sh validate <cmd...>"
#
# Idempotent: re-running with the same tag overwrites prior logs.

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <tag> <command...>" >&2
  exit 2
fi

TAG="$1"; shift
LOG_DIR="${LOG_DIR:-$HOME/jx-runs}"
mkdir -p "$LOG_DIR"

LOG="$LOG_DIR/$TAG.log"
EXIT_FILE="$LOG_DIR/$TAG.exit"
TIME_FILE="$LOG_DIR/$TAG.time"
DMESG_FILE="$LOG_DIR/$TAG.dmesg"
JOURNAL_FILE="$LOG_DIR/$TAG.journal"
TRACE_FILE="$LOG_DIR/$TAG.trace"

# Trace this wrapper itself for forensic clarity.
exec 2> >(tee -a "$TRACE_FILE" >&2)
set -x

START_EPOCH="$(date -u +%s)"
START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Pre-flight system snapshot (so we can compare before/after).
{
  echo "=== tpu-run start: $START_ISO (tag=$TAG) ==="
  echo "=== command: $* ==="
  echo "=== /proc/meminfo (head) ==="; head -5 /proc/meminfo
  echo "=== ulimit -a ==="; ulimit -a
  echo "=== uname -a ==="; uname -a
  echo "=== uptime ==="; uptime
} > "$LOG" 2>&1

# Run the actual command under /usr/bin/time, capturing stdout+stderr to LOG.
# `time -v` records peak RSS, signal that killed it, exit status.
# Note: `time` returns the child's exit status if not signalled, else 128+sig.
set +e
/usr/bin/time -v -o "$TIME_FILE" -- bash -c "$*" >> "$LOG" 2>&1
RC=$?
set -e

END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Decode exit (signal vs exit-code) into a self-explanatory string.
SIG=""
if [[ $RC -gt 128 ]]; then
  SIG_NUM=$((RC - 128))
  SIG_NAME="$(kill -l "$SIG_NUM" 2>/dev/null || echo "unknown")"
  SIG=" (signal $SIG_NUM SIG$SIG_NAME)"
fi

{
  echo "tag=$TAG"
  echo "command=$*"
  echo "start=$START_ISO"
  echo "end=$END_ISO"
  echo "rc=$RC$SIG"
} > "$EXIT_FILE"

# Capture kernel ring buffer + journal slice. These reveal OOM-killer,
# libtpu aborts, segfaults that left no python-side trace.
sudo dmesg -T 2>/dev/null | tail -100 > "$DMESG_FILE" || true
sudo journalctl --since "@$START_EPOCH" --no-pager > "$JOURNAL_FILE" 2>/dev/null || true

# Final summary appended to the main log so a single `tail` shows it all.
{
  echo
  echo "=== tpu-run end: $END_ISO rc=$RC$SIG ==="
  echo "--- /usr/bin/time -v (peak RSS, signal, etc.) ---"
  cat "$TIME_FILE" || true
  echo "--- last 20 dmesg lines ---"
  tail -20 "$DMESG_FILE" || true
  echo "--- last 20 journal lines ---"
  tail -20 "$JOURNAL_FILE" || true
} >> "$LOG"

exit "$RC"
