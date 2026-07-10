#!/usr/bin/env bash

set -o pipefail

PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
EXP_ROOT="${POINTCEPT_EXP_ROOT:-$PROJECT/exp}"
LOG_ROOT="$EXP_ROOT/fixed_protocol/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$LOG_ROOT/instance_evidence_${TIMESTAMP}.log"

mkdir -p "$LOG_ROOT"
printf '%s\n' "$REPORT"

section() {
  echo
  echo "===== $1 ====="
  sync
}

exec > "$REPORT" 2>&1

section BASIC
date '+time=%F %T %z'
hostname
pwd
timeout 5 uptime || true
git -C "$PROJECT" rev-parse --short HEAD 2>&1 || true

section LEGACY_PROCESSES
pgrep -af '[r]un_fixed_screen_then_shutdown.sh|[w]atch_train_then_shutdown.sh|[r]un_fixed_eval_safe.sh|[e]val_s3dis_fixed_protocol.py' || true

section SCHEDULE_FILE
if [ -f /run/systemd/shutdown/scheduled ]; then
  stat /run/systemd/shutdown/scheduled || true
  sed -n '1,80p' /run/systemd/shutdown/scheduled || true
else
  echo "no systemd schedule file"
fi

section CRON
timeout 5 crontab -l 2>&1 || true
grep -RInE 'poweroff|halt|watch_train|run_fixed_screen_then' \
  /etc/crontab /etc/cron.d /etc/cron.hourly /etc/cron.daily \
  2>/dev/null || true

section STARTUP_FILES
grep -RInE 'poweroff|halt|watch_train|run_fixed_screen_then' \
  /root/.bashrc /root/.profile /root/.bash_profile /etc/profile /etc/profile.d \
  2>/dev/null || true

section PREVIOUS_BOOT
timeout 10 journalctl -b -1 -n 160 --no-pager 2>&1 || true
timeout 5 last -x 2>&1 | head -n 50 || true

section FIXED_PROTOCOL_FILES
find "$EXP_ROOT/fixed_protocol" -maxdepth 6 -type f \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null \
  | sort -r | head -n 160

section GPU
timeout 10 nvidia-smi 2>&1 || true

section COMPLETE
date '+time=%F %T %z'
sync
