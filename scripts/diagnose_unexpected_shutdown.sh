#!/usr/bin/env bash

set -o pipefail

PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
EXP_ROOT="${POINTCEPT_EXP_ROOT:-$PROJECT/exp}"
LOG_ROOT="$EXP_ROOT/fixed_protocol/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT="$LOG_ROOT/shutdown_diagnosis_${TIMESTAMP}.log"

mkdir -p "$LOG_ROOT"

{
  echo "===== BASIC ====="
  date '+time=%F %T %z'
  hostname
  uptime
  who -b 2>&1 || true
  echo

  echo "===== SHUTDOWN SCHEDULE ====="
  # Read system state directly. Do not invoke the platform's shutdown command.
  type -a shutdown 2>&1 || true
  if [ -f /run/systemd/shutdown/scheduled ]; then
    cat /run/systemd/shutdown/scheduled
  else
    echo "no /run/systemd/shutdown/scheduled"
  fi
  echo

  echo "===== RELATED PROCESSES ====="
  pgrep -af 'shutdown|watch_train_then_shutdown|run_fixed_screen|eval_s3dis_fixed_protocol' \
    2>&1 || true
  echo

  echo "===== ROOT CRONTAB ====="
  crontab -l 2>&1 || true
  echo

  echo "===== SYSTEM CRON MATCHES ====="
  grep -RInE 'shutdown|poweroff|halt|watch_train|fixed_screen' \
    /etc/crontab /etc/cron.d /etc/cron.hourly /etc/cron.daily \
    2>/dev/null || true
  echo

  echo "===== SYSTEMD TIMERS ====="
  systemctl list-timers --all --no-pager 2>&1 || true
  echo

  echo "===== PREVIOUS BOOT TAIL ====="
  journalctl -b -1 -n 120 --no-pager 2>&1 || true
  echo

  echo "===== LAST REBOOTS/SHUTDOWNS ====="
  last -x 2>&1 | head -n 40 || true
  echo

  echo "===== FIXED EVAL LOG FILES ====="
  find "$EXP_ROOT" -type f \
    \( -name '*fixed*log' -o -name 'worker.log' -o -name '*shutdown*log' \) \
    -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null \
    | sort -r | head -n 120
  echo

  echo "===== LATEST FIXED LOG TAILS ====="
  find "$EXP_ROOT" -type f \
    \( -name '*fixed*log' -o -name '*shutdown*log' \) \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -n 6 | cut -d' ' -f2- \
    | while IFS= read -r file; do
        echo "--- $file ---"
        tail -n 80 "$file" 2>&1 || true
      done
  echo

  echo "===== GPU ====="
  nvidia-smi 2>&1 || true
} > "$REPORT" 2>&1

echo "$REPORT"
