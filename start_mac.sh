#!/usr/bin/env bash
# Launch the Alpaca paper-trading bot on macOS.
#   ./start_mac.sh              run the bot (Ctrl-C to stop; positions keep their GTC brackets)
#   ./start_mac.sh --dry-run    evaluate and journal only, no orders
#   ./start_mac.sh --once       single cycle smoke test
#
# caffeinate -i blocks idle sleep and -s blocks system sleep while on AC power, for as long
# as the bot runs. Neither stops a MacBook from sleeping when the lid is closed (unless it is
# in clamshell mode with power and an external display). Keep the lid open and plugged in.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
  echo "Python 3.9+ is required (found: $("$PYTHON_BIN" --version 2>&1)). Try: brew install python@3.12" >&2
  exit 1
fi

if [ ! -f .env ]; then
  echo "Missing .env. Run: cp .env.example .env  and add your Alpaca PAPER keys." >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Creating virtualenv in .venv ..."
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

REQ_HASH="$(shasum -a 256 requirements.txt | awk '{print $1}')"
if [ ! -f .venv/.requirements.sha256 ] || [ "$(cat .venv/.requirements.sha256)" != "$REQ_HASH" ]; then
  echo "Installing dependencies ..."
  python -m pip install --upgrade pip --quiet
  python -m pip install -r requirements.txt --quiet
  echo "$REQ_HASH" > .venv/.requirements.sha256
fi

mkdir -p logs
echo "Running offline test suite before going live ..."
python -m pytest -q || { echo "Tests failed; refusing to start." >&2; exit 1; }

echo "Starting bot (logs/bot.log, paper_trading_journal.csv). Ctrl-C to stop."
exec caffeinate -i -s python bot.py "$@"
