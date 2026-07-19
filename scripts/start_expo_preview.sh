#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8765}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOBILE_ROOT="$REPO_ROOT/mobile"

cd "$REPO_ROOT"
python -m pip install -e ".[dev,visualization,mobile]"

LAN_ADDRESS="$(python - <<'PY'
from sightly_assist.mobile_api import discover_lan_ipv4_addresses
addresses = discover_lan_ipv4_addresses()
print(addresses[0] if addresses else "")
PY
)"

if [[ -z "$LAN_ADDRESS" ]]; then
  echo "No usable LAN IPv4 address was found." >&2
  exit 1
fi

export EXPO_PUBLIC_SIGHTLY_API_URL="http://${LAN_ADDRESS}:${PORT}"
echo "Phone backend URL: ${EXPO_PUBLIC_SIGHTLY_API_URL}"

python -m sightly_assist.mobile_cli serve --port "$PORT" --demo &
BRIDGE_PID=$!
trap 'kill "$BRIDGE_PID" 2>/dev/null || true' EXIT INT TERM

cd "$MOBILE_ROOT"
npm install
npm start
