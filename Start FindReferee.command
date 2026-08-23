#!/bin/zsh

set -e

APP_ROOT="${0:A:h}"
cd "$APP_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3.10 or newer is required. Install Python, then double-click this launcher again."
  read -r "?Press Return to close."
  exit 1
fi

if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is required. Update Python, then double-click this launcher again."
  read -r "?Press Return to close."
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Preparing FindReferee for first use…"
  python3 -m venv .venv
fi

REQUIREMENTS_DIGEST="$(shasum requirements.txt | awk '{print $1}')"
STAMP_FILE=".venv/.findreferee-requirements"
INSTALLED_DIGEST=""
if [[ -f "$STAMP_FILE" ]]; then
  INSTALLED_DIGEST="$(<"$STAMP_FILE")"
fi

if [[ "$REQUIREMENTS_DIGEST" != "$INSTALLED_DIGEST" ]]; then
  echo "Installing required components…"
  .venv/bin/python -m pip install -r requirements.txt
  print -r -- "$REQUIREMENTS_DIGEST" > "$STAMP_FILE"
fi

APP_URL="http://127.0.0.1:8000/"
if curl --silent --fail "$APP_URL" >/dev/null 2>&1; then
  open "$APP_URL"
  exit 0
fi

echo "Starting FindReferee… Keep this window open while using the app."
(sleep 1; open "$APP_URL") &
exec env CODEX_TIMEOUT_SECONDS=1200 .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
