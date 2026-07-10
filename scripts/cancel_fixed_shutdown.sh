#!/usr/bin/env bash

set -o pipefail

SERVER_ID="${1:?Usage: $0 server-a}"
PROJECT="${POINTCEPT_ROOT:-/root/autodl-tmp/Pointcept}"
EXP_ROOT="${POINTCEPT_EXP_ROOT:-$PROJECT/exp}"
LEGACY_CANCEL_FILE="$EXP_ROOT/cancel_${SERVER_ID}_shutdown"
CURRENT_CANCEL_FILE="$EXP_ROOT/fixed_protocol/cancel_${SERVER_ID}_shutdown"

mkdir -p "$EXP_ROOT/fixed_protocol"
touch "$LEGACY_CANCEL_FILE" "$CURRENT_CANCEL_FILE"

WATCHER_PIDS="$(pgrep -f '[w]atch_train_then_shutdown.sh' || true)"
if [ -n "$WATCHER_PIDS" ]; then
  kill -TERM $WATCHER_PIDS 2>/dev/null || true
fi

echo "已写入新旧关机取消标记："
echo "  $LEGACY_CANCEL_FILE"
echo "  $CURRENT_CANCEL_FILE"
echo "已停止旧 watch_train_then_shutdown.sh（如存在）。"
echo "未调用 shutdown 命令；请在 AutoDL 控制台检查并取消平台自动关机策略。"
