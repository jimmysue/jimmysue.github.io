#!/usr/bin/env bash
# Start a local HTTP server for the paper-reading blog.
# Usage:
#   ./serve.sh             # default port 8765, foreground
#   ./serve.sh 9000        # custom port
#   ./serve.sh 8765 -bg    # background (logs to /tmp/paper-reading-server.log)

set -euo pipefail

PORT="${1:-8765}"
MODE="${2:-fg}"

cd "$(dirname "$0")"

# Pick an available port if requested one is taken
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port $PORT is already in use." >&2
  for try in 8766 8767 8768 8769 8770; do
    if ! lsof -iTCP:"$try" -sTCP:LISTEN >/dev/null 2>&1; then
      PORT="$try"
      echo "Using $PORT instead." >&2
      break
    fi
  done
fi

URL="http://localhost:$PORT/"

if [[ "$MODE" == "-bg" || "$MODE" == "bg" ]]; then
  LOG=/tmp/paper-reading-server.log
  : > "$LOG"
  nohup python3 -m http.server "$PORT" --bind 127.0.0.1 > "$LOG" 2>&1 &
  SERVER_PID=$!
  sleep 0.5
  echo "Serving paper-reading blog at $URL (PID $SERVER_PID, log: $LOG)"
  echo "To stop: kill $SERVER_PID"
else
  echo "Serving paper-reading blog at $URL  (Ctrl-C to stop)"
  exec python3 -m http.server "$PORT" --bind 127.0.0.1
fi
