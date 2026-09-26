#!/bin/zsh
# Double-click to compare this FindReferee with an earlier version on your own
# referee reports. Safe to run again: finished cases and runs are skipped, so an
# interrupted benchmark continues where it stopped.
#
# Optional settings (environment variables):
#   FINDREFEREE_REPORTS   folder with the reports    (default: ~/Downloads/Test reports)
#   FINDREFEREE_BASELINE  version to compare against (default: 642043c, before the September 2026 changes)
#   FINDREFEREE_REPEAT    runs per case and version  (default: 1)
#   FINDREFEREE_MODEL     model                      (default: gpt-5.6-sol)
#   FINDREFEREE_EFFORT    reasoning strength         (default: xhigh)

set -e
setopt pipefail

APP_ROOT="${0:A:h}"
cd "$APP_ROOT"

REPORTS_DIR="${FINDREFEREE_REPORTS:-$HOME/Downloads/Test reports}"
BASELINE="${FINDREFEREE_BASELINE:-642043c}"
REPEAT="${FINDREFEREE_REPEAT:-1}"
MODEL="${FINDREFEREE_MODEL:-gpt-5.6-sol}"
EFFORT="${FINDREFEREE_EFFORT:-xhigh}"
BASELINE_DIR="benchmarks/.baseline-$BASELINE"
LOG="benchmarks/benchmark.log"

finish() {
  echo
  read -r "?Press Return to close." || true
  exit "$1"
}

if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is required. Install it, then double-click this file again."
  finish 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "git is required. Run 'xcode-select --install' in Terminal, then double-click this file again."
  finish 1
fi

if [[ ! -x .venv/bin/python ]]; then
  echo "Preparing FindReferee for first use…"
  python3 -m venv .venv
fi
REQUIREMENTS_DIGEST="$(shasum requirements.txt | awk '{print $1}')"
STAMP_FILE=".venv/.findreferee-requirements"
if [[ "$REQUIREMENTS_DIGEST" != "$(cat "$STAMP_FILE" 2>/dev/null)" ]]; then
  echo "Installing required components…"
  .venv/bin/python -m pip install -r requirements.txt || finish 1
  print -r -- "$REQUIREMENTS_DIGEST" > "$STAMP_FILE"
fi

# Use the same optional settings as the app launcher.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
mkdir -p benchmarks

echo
echo "Step 1 of 4: test cases"
if [[ -d "$REPORTS_DIR" ]]; then
  .venv/bin/python benchmark.py setup --from "$REPORTS_DIR" || finish 1
fi
cases=(benchmarks/*/case.json(N))
if (( ${#cases} == 0 )); then
  echo "No test cases yet. Put the reports in \"$REPORTS_DIR\" (or set FINDREFEREE_REPORTS), then double-click this file again."
  finish 1
fi

echo
echo "Step 2 of 4: checking the cases and the Codex sign-in"
.venv/bin/python benchmark.py check --model "$MODEL" || finish 1

echo
echo "Step 3 of 4: preparing the earlier version ($BASELINE)"
if [[ ! -f "$BASELINE_DIR/app.py" ]]; then
  git worktree prune
  git worktree add --detach "$BASELINE_DIR" "$BASELINE" >/dev/null || finish 1
fi
CURRENT="current-$(git rev-parse --short HEAD)"
EARLIER="baseline-$BASELINE"

RUNS=$(( ${#cases} * REPEAT * 2 ))
echo
echo "Step 4 of 4: up to $RUNS analyses with $MODEL at $EFFORT, roughly $(( RUNS * 26 / 60 ))–$(( (RUNS * 43 + 59) / 60 )) hours."
echo "Keep the Mac plugged in; it stays awake while this runs. Closing this window stops the"
echo "benchmark, and double-clicking this file again continues from the last finished run."
read -r "?Press Return to start." || true

KEEP_AWAKE=()
if command -v caffeinate >/dev/null 2>&1; then
  KEEP_AWAKE=(caffeinate -i)
fi
# The current version runs first, so warm public-source caches favor the earlier
# version and the analysis-time comparison stays conservative.
"${KEEP_AWAKE[@]}" .venv/bin/python benchmark.py run --resume --repeat "$REPEAT" --label "$CURRENT" \
  --model "$MODEL" --effort "$EFFORT" 2>&1 | tee -a "$LOG" || finish 1
"${KEEP_AWAKE[@]}" .venv/bin/python benchmark.py --app-dir "$BASELINE_DIR" run --resume --repeat "$REPEAT" \
  --label "$EARLIER" --model "$MODEL" --effort "$EFFORT" 2>&1 | tee -a "$LOG" || finish 1

REPORT="benchmarks/comparison-$EARLIER-vs-$CURRENT.txt"
{
  echo "FindReferee benchmark: $EARLIER vs $CURRENT ($MODEL at $EFFORT, $REPEAT run(s) per case and version)"
  echo
  echo "All cases"
  .venv/bin/python benchmark.py compare --baseline "$EARLIER" --candidate "$CURRENT" 2>/dev/null \
    || echo "No case has runs from both versions."
  echo
  echo "Confirmed labels only"
  .venv/bin/python benchmark.py compare --baseline "$EARLIER" --candidate "$CURRENT" --confirmed-only 2>/dev/null \
    || echo "No confirmed case has runs from both versions."
} > "$REPORT"
echo
cat "$REPORT"
echo
if command -v pbcopy >/dev/null 2>&1; then
  pbcopy < "$REPORT"
  echo "The comparison is on the clipboard; paste it into the Claude conversation. It contains no report text."
fi
echo "Saved to $APP_ROOT/$REPORT"
finish 0
