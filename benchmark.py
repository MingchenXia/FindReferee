"""Run, score, compare, and calibrate FindReferee on private labeled cases.

Each case is a folder under benchmarks/ (gitignored) with a case.json such as:

    {
      "target": "report.pdf",
      "candidates": ["Jane Example / J. Example", "Alex Sample"],
      "context": "arXiv:2504.21300",
      "underlying": "manuscript.pdf",
      "underlying_authors": ["Manuscript Author"],
      "references": {"Alex Sample": ["known-report.txt"]},
      "candidate_profiles": {},
      "controls": {},
      "explore_outside_candidates": true,
      "expected_author": "Alex Sample",
      "label_status": "confirmed"
    }

"target_text" may replace "target". Only the inputs reach the analysis; the
expected label and label status are read solely by the scorer.

Every run is saved under <case>/runs/ with a label (by default the git revision
of the app version that produced it), so two versions can be compared on the
same cases. --app-dir runs another checkout's app.py, for example an older
revision checked out with `git worktree add`.

    python benchmark.py setup --from "~/Downloads/Test reports"
    python benchmark.py check [--model M]
    python benchmark.py run [--cases NAME ...] [--repeat N] [--resume] [--label L] [--app-dir DIR] [--model M] [--effort E]
    python benchmark.py score [--label L] [--runs latest|all] [--confirmed-only] [--json]
    python benchmark.py compare --baseline L1 --candidate L2 [--confirmed-only] [--json]
    python benchmark.py fit-calibration --output calibration.json [--label L] [--confirmed-only]
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation_metrics import (
    NO_LISTED_CANDIDATE,
    aggregate_scores,
    fit_temperature,
    probability_distribution,
    repeated_run_stability,
    score_case,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "benchmarks"
LABEL_FIELDS = {"expected_author", "label_status"}
FALLBACK_STRATEGY = "safe timeout fallback"
UNLABELED = "unlabeled"
# Suggestions for the four VALIDATION.md cases, matched against report file names.
# The setup wizard shows each one as a default that can be accepted or replaced.
KNOWN_CASES: tuple[tuple[re.Pattern[str], dict[str, Any]], ...] = (
    (re.compile(r"orbifold", re.I), {
        "name": "orbifold",
        "candidates": ["Ya Deng", "Charles Favre", "Mingchen Xia"],
        "expected_author": "Ya Deng",
        "label_status": "confirmed",
    }),
    (re.compile(r"meng|zhou", re.I), {
        "name": "meng-zhou",
        "candidates": ["Mingchen Xia", "Valentino Tosatti"],
        "expected_author": "Mingchen Xia",
        "label_status": "confirmed",
    }),
    (re.compile(r"lnm|lecture", re.I), {
        "name": "lnm-xia",
        "candidates": ["Charles Favre", "Sébastien Boucksom", "Mingchen Xia"],
        "expected_author": "Charles Favre",
        "label_status": "confirmed",
    }),
    (re.compile(r"lewicka|decomposition", re.I), {
        "name": "su-lewicka",
        "candidates": ["Marta Lewicka", "László Székelyhidi Jr.", "Mohammad Reza Pakzad"],
        "expected_author": "Marta Lewicka",
        "label_status": "belief",
        "context": "arXiv:2504.21300",
    }),
)
CHECK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"reply": {"type": "string"}},
    "required": ["reply"],
}


class ProviderFailure(Exception):
    """A model call failed for a reason other than the analysis time budget."""
# The app module under test: this checkout's app.py unless --app-dir names another.
app: Any = None


def _load_app(app_dir: Path) -> Any:
    """Import app.py from app_dir so the same cases can run against another version."""
    app_dir = app_dir.resolve()
    loaded = sys.modules.get("app")
    if loaded is not None:
        if Path(loaded.__file__).resolve().parent != app_dir:
            raise SystemExit(f"app.py is already loaded from {Path(loaded.__file__).parent}; run one version per process.")
        return loaded
    if not (app_dir / "app.py").is_file():
        raise SystemExit(f"{app_dir} does not contain app.py.")
    # First on the path, so the version's own stylometry, corpus, and citation modules load with it.
    sys.path.insert(0, str(app_dir))
    return importlib.import_module("app")


def _revision(app_dir: Path) -> str:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(app_dir), *arguments], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        revision = git("rev-parse", "--short", "HEAD")
        return f"{revision}-dirty" if git("status", "--porcelain", "--untracked-files=no") else revision
    except (OSError, subprocess.CalledProcessError):
        return UNLABELED


def _cases(root: Path, names: list[str] | None) -> list[Path]:
    folders = sorted(path.parent for path in root.glob("*/case.json"))
    if names:
        wanted = set(names)
        folders = [folder for folder in folders if folder.name in wanted]
        missing = wanted - {folder.name for folder in folders}
        if missing:
            raise SystemExit(f"Unknown case(s): {', '.join(sorted(missing))}")
    if not folders:
        raise SystemExit(f"No benchmark cases were found under {root}.")
    return folders


def _read_case(folder: Path) -> dict[str, Any]:
    return json.loads((folder / "case.json").read_text(encoding="utf-8"))


def _document(folder: Path, relative: str) -> dict[str, Any]:
    path = folder / relative
    suffix = path.suffix.lower()
    if suffix not in app.SUPPORTED_EXTENSIONS:
        raise SystemExit(f"{path} is not a supported file type.")
    payload = path.read_bytes()
    parse = getattr(app, "_parse_upload_payload", None)
    if parse is not None:
        return parse(path.name, suffix, payload)
    # Older versions parsed inside the upload handler; this mirrors that code.
    metadata: dict[str, str] = {}
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(payload))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        for key, value in (reader.metadata or {}).items():
            if value is not None and str(value).strip():
                metadata[str(key).lstrip("/")] = str(value).strip()
    else:
        text = payload.decode("utf-8-sig")
    text, truncated = app._trim(text)
    return {"name": path.name, "text": text, "truncated": truncated, "metadata": metadata, "format": suffix.lstrip(".")}


def _analysis_inputs(folder: Path, case: dict[str, Any], model: str, effort: str) -> dict[str, Any]:
    """Build exactly what the web form would submit; label fields are never read here."""
    inputs = {key: value for key, value in case.items() if key not in LABEL_FIELDS}
    if inputs.get("target_text"):
        text, truncated = app._trim(str(inputs["target_text"]))
        documents = [{"name": "Pasted text", "text": text, "truncated": truncated, "metadata": {}, "format": "text"}]
    else:
        documents = [_document(folder, str(inputs["target"]))]
    candidate_list = app._validate_request("attribution", "\n".join(inputs.get("candidates", [])), documents)
    controls = app._parse_analysis_controls(json.dumps(inputs.get("controls", {})))
    controls["disable_outside_candidates"] = bool(candidate_list) and not inputs.get("explore_outside_candidates", True)
    reference_corpus: dict[str, list[dict[str, Any]]] = {}
    for author, files in (inputs.get("references") or {}).items():
        if author not in candidate_list:
            raise SystemExit(f"{folder.name}: reference files are assigned to {author!r}, which is not a candidate.")
        remaining = app.MAX_REFERENCE_CHARS_PER_AUTHOR
        for relative in files[: app.MAX_REFERENCE_FILES_PER_AUTHOR]:
            parsed = _document(folder, relative)
            clipped = parsed["text"][:remaining]
            if clipped:
                reference_corpus.setdefault(author, []).append(
                    {**parsed, "text": clipped, "truncated": parsed["truncated"] or len(clipped) < len(parsed["text"])}
                )
                remaining -= len(clipped)
    selected_model, selected_effort = app._requested_model_settings(model, effort)
    return {
        "mode": "attribution",
        "candidate_list": candidate_list,
        "documents": documents,
        "context_note": str(inputs.get("context", "")),
        "candidate_profiles": app._parse_candidate_context(json.dumps(inputs.get("candidate_profiles", {})), candidate_list),
        "controls": controls,
        "selected_model": selected_model,
        "selected_effort": selected_effort,
        "reference_corpus": reference_corpus,
        "underlying_document": _document(folder, str(inputs["underlying"])) if inputs.get("underlying") else None,
        "declared_underlying_authors": app._parse_underlying_authors("\n".join(inputs.get("underlying_authors", []))),
    }


def _run_case(folder: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    def progress(stage: str, clues: list[str] | None = None) -> None:
        print(f"[{folder.name}] {stage}", file=sys.stderr)

    started = time.monotonic()
    try:
        result = asyncio.run(
            app._perform_analysis(
                inputs["mode"],
                inputs["candidate_list"],
                inputs["documents"],
                inputs["context_note"],
                inputs["candidate_profiles"],
                inputs["controls"],
                inputs["selected_model"],
                inputs["selected_effort"],
                inputs["reference_corpus"],
                inputs["underlying_document"],
                progress,
                inputs["declared_underlying_authors"],
            )
        )
    except Exception as exc:
        # A login, quota, or connection failure says nothing about the version under test.
        if not getattr(app, "_is_time_budget_error", lambda _error: False)(exc):
            raise ProviderFailure(str(getattr(exc, "detail", exc))) from exc
        # A time-budget fallback is a genuine outcome of the version, so it is kept on record.
        result = app._analysis_fallback_result(
            inputs["mode"],
            inputs["candidate_list"],
            inputs["documents"],
            inputs["selected_model"],
            inputs["selected_effort"],
            exc,
            inputs["reference_corpus"],
        )
    result["total_elapsed_seconds"] = round(time.monotonic() - started, 1)
    return result


def command_run(args: argparse.Namespace) -> int:
    revision = _revision(args.app_dir)
    label = args.label or revision
    model = args.model or app.CODEX_MODEL
    effort = args.effort or app.CODEX_REASONING_EFFORT
    print(f"Running app from {args.app_dir.resolve()} as label {label!r} with {model} at {effort}.", file=sys.stderr)
    for folder in _cases(args.root, args.cases):
        try:
            inputs = _analysis_inputs(folder, _read_case(folder), model, effort)
        except app.HTTPException as exc:
            raise SystemExit(f"{folder.name}: {exc.detail}") from exc
        runs = folder / "runs"
        runs.mkdir(exist_ok=True)
        finished = (
            sum(1 for path in runs.glob("*.json") if _label_of(json.loads(path.read_text(encoding="utf-8"))) == label)
            if args.resume
            else 0
        )
        if finished >= args.repeat:
            print(f"[{folder.name}] {finished} run(s) labeled {label!r} already saved; skipped", file=sys.stderr)
            continue
        for repetition in range(finished, args.repeat):
            try:
                result = _run_case(folder, inputs)
            except ProviderFailure as exc:
                print(
                    f"\n[{folder.name}] The model provider failed: {exc}\n"
                    "Finished runs are kept. Fix the problem, then run again with --resume to continue.",
                    file=sys.stderr,
                )
                return 2
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            result["_benchmark"] = {
                "label": label,
                "revision": revision,
                "app_dir": str(args.app_dir.resolve()),
                "model": model,
                "effort": effort,
                "saved_at": stamp,
            }
            path = runs / f"{stamp}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{folder.name}] run {repetition + 1}/{args.repeat} saved to {path}", file=sys.stderr)
    return 0


def _label_of(result: dict[str, Any]) -> str:
    return str((result.get("_benchmark") or {}).get("label") or UNLABELED)


def _case_runs(args: argparse.Namespace, label: str | None) -> dict[str, tuple[str, list[tuple[str, dict[str, Any]]]]]:
    """case name -> (expected label, [(run name, result)]) for cases that have matching runs."""
    selected: dict[str, tuple[str, list[tuple[str, dict[str, Any]]]]] = {}
    for folder in _cases(args.root, args.cases):
        case = _read_case(folder)
        if args.confirmed_only and case.get("label_status", "confirmed") != "confirmed":
            continue
        runs = [(path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in sorted((folder / "runs").glob("*.json"))]
        if label is not None:
            runs = [(name, result) for name, result in runs if _label_of(result) == label]
        if getattr(args, "runs", "all") == "latest":
            runs = runs[-1:]
        if runs:
            selected[folder.name] = (str(case["expected_author"]), runs)
        else:
            print(f"[{folder.name}] no saved runs{f' labeled {label!r}' if label else ''}; skipped", file=sys.stderr)
    return selected


def _score_rows(case_runs: dict[str, tuple[str, list[tuple[str, dict[str, Any]]]]]) -> list[dict[str, Any]]:
    rows = []
    for case, (expected, runs) in case_runs.items():
        for run, result in runs:
            distribution = probability_distribution(result)
            if expected not in distribution:
                print(f"[{case}] expected label {expected!r} is not among {sorted(distribution)}", file=sys.stderr)
            rows.append(
                {
                    "case": case,
                    "run": run,
                    "label": _label_of(result),
                    "expected": expected,
                    "leader": max(distribution, key=distribution.get) if distribution else "",
                    "elapsed_seconds": float(result.get("total_elapsed_seconds") or 0.0),
                    "fallback": result.get("review_strategy") == FALLBACK_STRATEGY,
                    **score_case(result, expected),
                }
            )
    return rows


def _summary(case_runs: dict[str, tuple[str, list[tuple[str, dict[str, Any]]]]]) -> dict[str, Any]:
    rows = _score_rows(case_runs)
    stability = {
        case: repeated_run_stability([result for _, result in runs])
        for case, (_, runs) in case_runs.items()
        if len(runs) > 1
    }
    summary: dict[str, Any] = dict(aggregate_scores(rows))
    summary["mean_elapsed_minutes"] = sum(row["elapsed_seconds"] for row in rows) / len(rows) / 60 if rows else 0.0
    summary["fallback_rate"] = sum(row["fallback"] for row in rows) / len(rows) if rows else 0.0
    summary["mean_pairwise_js_divergence"] = (
        sum(values["mean_pairwise_js_divergence"] for values in stability.values()) / len(stability) if stability else None
    )
    return {"rows": rows, "aggregate": summary, "repeated_run_stability": stability}


def command_score(args: argparse.Namespace) -> int:
    report = _summary(_case_runs(args, args.label))
    rows, aggregate = report["rows"], report["aggregate"]
    if args.json:
        print(json.dumps({"cases": rows, **{key: value for key, value in report.items() if key != "rows"}}, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        print(
            f"{row['case']:<24} {row['label']:<14} {row['run']:<24} expected {row['expected']!r}: "
            f"{row['expected_probability']:.0%}, rank {row['expected_rank']}, leader {row['leader']!r}, "
            f"{row['determination_status']}"
        )
    if aggregate.get("case_count"):
        print(
            f"\n{aggregate['case_count']} scored run(s): unique Top-1 {aggregate['top1_accuracy']:.0%}, "
            f"Top-1 including ties {aggregate['top1_including_ties_accuracy']:.0%}, "
            f"MRR {aggregate['mean_reciprocal_rank']:.3f}, mean log loss {aggregate['mean_log_loss']:.3f}, "
            f"false precise claims {aggregate['false_precise_claim_rate']:.0%}"
        )
    for name, values in report["repeated_run_stability"].items():
        print(f"{name}: mean pairwise Jensen-Shannon divergence {values['mean_pairwise_js_divergence']:.4f} over {values['run_count']} runs")
    return 0


# (key, display name, True if higher is better / False if lower / None if neutral, format)
COMPARISON_METRICS = (
    ("top1_accuracy", "Unique Top-1", True, "percent"),
    ("top1_including_ties_accuracy", "Top-1 including ties", True, "percent"),
    ("mean_expected_probability", "Mean expected-author probability", True, "percent"),
    ("mean_true_class_margin", "Mean true-author margin", True, "points"),
    ("mean_reciprocal_rank", "Mean reciprocal rank", True, "decimal"),
    ("mean_log_loss", "Mean log loss", False, "decimal"),
    ("mean_brier_score", "Mean Brier score", False, "decimal"),
    ("false_precise_claim_rate", "False precise claims", False, "percent"),
    ("precise_claim_coverage", "Precise-claim coverage", None, "percent"),
    ("unable_to_determine_rate", "Unable-to-determine rate", None, "percent"),
    ("mean_pairwise_js_divergence", "Run-to-run divergence (JS)", False, "fine"),
    ("fallback_rate", "Safe-fallback runs", False, "percent"),
    ("mean_elapsed_minutes", "Mean analysis time (minutes)", False, "minutes"),
)


# Changes smaller than the displayed precision count as "same".
DISPLAY_TOLERANCE = {"percent": 0.0005, "points": 0.0005, "decimal": 0.0005, "fine": 0.00005, "minutes": 0.05}


def _format(value: float | None, style: str, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    sign = "+" if signed and value > 0 else ""
    if style == "percent":
        return f"{sign}{value * 100:.1f}{' pts' if signed else '%'}"
    if style == "points":
        return f"{sign}{value * 100:.1f} pts"
    if style == "minutes":
        return f"{sign}{value:.1f}"
    if style == "fine":
        return f"{sign}{value:.4f}"
    return f"{sign}{value:.3f}"


def command_compare(args: argparse.Namespace) -> int:
    baseline = _case_runs(args, args.baseline)
    candidate = _case_runs(args, args.candidate)
    shared = sorted(set(baseline) & set(candidate))
    if not shared:
        raise SystemExit(f"No case has runs labeled both {args.baseline!r} and {args.candidate!r}.")
    unpaired = sorted(set(baseline) ^ set(candidate))
    before = _summary({case: baseline[case] for case in shared})
    after = _summary({case: candidate[case] for case in shared})

    metrics = []
    for key, name, higher_is_better, style in COMPARISON_METRICS:
        old, new = before["aggregate"].get(key), after["aggregate"].get(key)
        change = None if old is None or new is None else new - old
        verdict = "n/a" if change is None else "same" if abs(change) < DISPLAY_TOLERANCE[style] else "neutral" if higher_is_better is None else (
            "better" if (change > 0) == higher_is_better else "worse"
        )
        metrics.append({"metric": name, "key": key, "baseline": old, "candidate": new, "change": change, "verdict": verdict, "style": style})

    cases = []
    for case in shared:
        expected = baseline[case][0]
        old_rows = [row for row in before["rows"] if row["case"] == case]
        new_rows = [row for row in after["rows"] if row["case"] == case]
        old_probabilities = [row["expected_probability"] for row in old_rows]
        new_probabilities = [row["expected_probability"] for row in new_rows]
        old_mean = sum(old_probabilities) / len(old_probabilities)
        new_mean = sum(new_probabilities) / len(new_probabilities)
        # A change smaller than either version's own run-to-run spread is not distinguishable from noise.
        spread = max(max(old_probabilities) - min(old_probabilities), max(new_probabilities) - min(new_probabilities))
        repeated = len(old_rows) > 1 and len(new_rows) > 1
        change = new_mean - old_mean
        cases.append(
            {
                "case": case,
                "expected": expected,
                "baseline_probability": old_mean,
                "candidate_probability": new_mean,
                "change": change,
                "baseline_top1_rate": sum(row["top1_correct"] for row in old_rows) / len(old_rows),
                "candidate_top1_rate": sum(row["top1_correct"] for row in new_rows) / len(new_rows),
                "baseline_status": sorted({row["determination_status"] for row in old_rows}),
                "candidate_status": sorted({row["determination_status"] for row in new_rows}),
                "run_to_run_spread": spread if repeated else None,
                "direction": (
                    "within run-to-run spread" if repeated and abs(change) <= spread
                    else "improved" if change > 0.005 else "worsened" if change < -0.005 else "unchanged"
                ),
            }
        )
    report = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "paired_cases": len(shared),
        "unpaired_cases": unpaired,
        "baseline_runs": before["aggregate"].get("case_count", 0),
        "candidate_runs": after["aggregate"].get("case_count", 0),
        "metrics": metrics,
        "cases": cases,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    print(
        f"{args.baseline} ({report['baseline_runs']} runs) vs {args.candidate} ({report['candidate_runs']} runs) "
        f"on {len(shared)} paired case(s)" + (f"; not paired: {', '.join(unpaired)}" if unpaired else "")
    )
    print(f"\n{'Metric':<34}{'Baseline':>12}{'Candidate':>12}{'Change':>13}  Verdict")
    for row in metrics:
        print(
            f"{row['metric']:<34}{_format(row['baseline'], row['style']):>12}{_format(row['candidate'], row['style']):>12}"
            f"{_format(row['change'], row['style'], signed=True):>13}  {row['verdict']}"
        )
    print(f"\n{'Case':<24}{'Expected author':<26}{'Baseline':>10}{'Candidate':>11}  Direction")
    for row in cases:
        print(
            f"{row['case']:<24}{row['expected'][:25]:<26}{row['baseline_probability']:>10.1%}{row['candidate_probability']:>11.1%}"
            f"  {row['direction']}"
        )
    directions = [row["direction"] for row in cases]
    print(
        f"\nExpected-author probability improved in {directions.count('improved')}, worsened in "
        f"{directions.count('worsened')}, and was unchanged or within run-to-run spread in "
        f"{len(directions) - directions.count('improved') - directions.count('worsened')} of {len(cases)} case(s)."
    )
    if not all(row["run_to_run_spread"] is not None for row in cases):
        print("Run each version with --repeat 3 or more to tell real changes from run-to-run variation.")
    return 0


def _ask(prompt: str, default: str = "") -> str:
    try:
        answer = input(f"  {prompt}{f' [{default}]' if default else ''}: ").strip()
    except EOFError as exc:
        raise SystemExit("\nSetup needs answers from the keyboard; run it in a Terminal window.") from exc
    return answer or default


def _split_names(value: str) -> list[str]:
    return [name.strip() for name in value.split(";") if name.strip()]


def _dropped_path(value: str) -> Path | None:
    """Read a path typed or dragged into Terminal, which may be quoted or backslash-escaped."""
    if not value.strip():
        return None
    try:
        parts = shlex.split(value)
    except ValueError:
        parts = [value.strip()]
    return Path(parts[0]).expanduser() if parts else None


def command_setup(args: argparse.Namespace) -> int:
    source = args.source.expanduser()
    if not source.is_dir():
        raise SystemExit(f"{source} is not a folder.")
    reports = sorted(
        path for path in source.iterdir() if path.is_file() and path.suffix.lower() in app.SUPPORTED_EXTENSIONS
    )
    if not reports:
        raise SystemExit(f"No PDF, TXT, Markdown, or TeX reports were found in {source}.")
    args.root.mkdir(parents=True, exist_ok=True)
    existing = {
        str(case.get("source_file")): folder.name
        for folder in sorted(path.parent for path in args.root.glob("*/case.json"))
        for case in [_read_case(folder)]
    }
    print(f"Setting up benchmark cases from {source}. Press Return to accept a suggestion in brackets.")
    for report in reports:
        if report.name in existing:
            print(f"\n{report.name}: already set up as {existing[report.name]!r}.")
            continue
        preset = next((values for pattern, values in KNOWN_CASES if pattern.search(report.stem)), {})
        print(f"\n{report.name}" + ("  (matches a case in VALIDATION.md)" if preset else ""))
        if _ask("Include this report as a test case? (y/n)", "y").lower().startswith("n"):
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", report.stem.casefold()).strip("-") or "case"
        while True:
            name = _ask("Case name", preset.get("name", slug))
            if not (args.root / name).exists():
                break
            print(f"  A case named {name!r} already exists; choose another name.")
        print("  List every candidate you want compared, including the true author.")
        while True:
            candidates = _split_names(_ask('Candidates, separated by ";"', "; ".join(preset.get("candidates", []))))
            if len(candidates) != 1:
                break
            print("  Enter at least two candidates, or none to let the app discover them.")
        while True:
            expected = _ask("True author (never shown to the analysis)", preset.get("expected_author", candidates[0] if candidates else ""))
            if expected and (not candidates or expected in candidates):
                break
            print("  The true author must be one of the candidates, spelled exactly the same.")
        while True:
            status = _ask("Is that label confirmed or only a belief? (confirmed/belief)", preset.get("label_status", "confirmed")).lower()
            if status in {"confirmed", "belief"}:
                break
        context = _ask("arXiv ID or DOI of the manuscript under review, or other context", preset.get("context", ""))
        manuscript = None
        while True:
            manuscript = _dropped_path(_ask("Manuscript file under review (drag it here, or press Return to skip)"))
            if manuscript is None or (manuscript.is_file() and manuscript.suffix.lower() in app.SUPPORTED_EXTENSIONS):
                break
            print("  That is not a readable PDF, TXT, Markdown, or TeX file.")
        manuscript_authors = _split_names(_ask('Manuscript authors, separated by ";" (optional)'))

        folder = args.root / name
        folder.mkdir(parents=True)
        shutil.copy2(report, folder / f"report{report.suffix.lower()}")
        case: dict[str, Any] = {
            "source_file": report.name,
            "target": f"report{report.suffix.lower()}",
            "candidates": candidates,
            "context": context,
            "expected_author": expected,
            "label_status": status,
        }
        if manuscript:
            shutil.copy2(manuscript, folder / f"manuscript{manuscript.suffix.lower()}")
            case["underlying"] = f"manuscript{manuscript.suffix.lower()}"
        if manuscript_authors:
            case["underlying_authors"] = manuscript_authors
        (folder / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved {folder / 'case.json'}")
    return 0


def command_check(args: argparse.Namespace) -> int:
    model = args.model or app.CODEX_MODEL
    problems = []
    print("Checking the benchmark cases:")
    for folder in _cases(args.root, args.cases):
        case = _read_case(folder)
        try:
            inputs = _analysis_inputs(folder, case, model, app.CODEX_REASONING_EFFORT)
        except (app.HTTPException, SystemExit, OSError, ValueError, KeyError) as exc:
            problems.append(f"{folder.name}: {getattr(exc, 'detail', exc)}")
            continue
        expected = str(case.get("expected_author", ""))
        candidates = inputs["candidate_list"]
        if not expected or (candidates and expected not in candidates):
            problems.append(f"{folder.name}: the true author {expected!r} is not one of the candidates.")
            continue
        words = len(inputs["documents"][0]["text"].split())
        print(
            f"  {folder.name}: {words} words, {len(candidates) or 'automatic'} candidates, "
            f"true author {expected!r} ({case.get('label_status', 'confirmed')})"
            + (", with manuscript" if inputs["underlying_document"] else "")
        )
        if words < 40:
            problems.append(f"{folder.name}: only {words} words could be extracted; a scanned PDF needs OCR first.")
    if problems:
        print("\nFix these before running:\n  " + "\n  ".join(problems))
        return 1
    print(f"\nChecking that {model} answers through the signed-in account…")
    try:
        reply = app._call_model(
            "Reply with the word OK.",
            "Connectivity check for the FindReferee benchmark.",
            "benchmark_check",
            CHECK_SCHEMA,
            model,
            "low",
            enable_search=False,
        )
    except Exception as exc:
        print(f"The model check failed: {getattr(exc, 'detail', exc)}")
        return 1
    print(f"  OK: {reply.get('_provider', 'provider')} answered with {reply.get('_model', model)}.")
    return 0


def command_fit_calibration(args: argparse.Namespace) -> int:
    pairs = []
    models = set()
    case_names = set()
    for name, (expected, runs) in _case_runs(args, args.label).items():
        for _run, result in runs:
            if result.get("review_strategy") == FALLBACK_STRATEGY:
                continue  # A non-determination carries no model distribution to calibrate.
            models.add(str(result.get("model", "")))
            case_names.add(name)
            pairs.append((result, expected))
    if not pairs:
        raise SystemExit("No completed runs are available to fit a calibration.")
    if len(models) > 1:
        raise SystemExit(f"Runs come from several models ({', '.join(sorted(models))}); fit one model at a time with --label.")
    fitted = fit_temperature(pairs)
    # Repeated runs of one case are correlated, so the app's minimum counts distinct cases.
    fitted.update({"case_count": len(case_names), "run_count": len(pairs)})
    fitted.update({"model": models.pop(), "fitted_at": datetime.now(timezone.utc).isoformat(), "no_listed_label": NO_LISTED_CANDIDATE})
    args.output.write_text(json.dumps(fitted, indent=2), encoding="utf-8")
    print(json.dumps(fitted, indent=2))
    minimum = getattr(app, "MIN_CALIBRATION_CASES", 20)
    if fitted["case_count"] < minimum:
        print(
            f"Warning: {fitted['case_count']} case(s) is below the {minimum} the app requires; "
            "the file will be ignored until more labeled cases are added.",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Folder that contains the case folders.")
    parser.add_argument("--app-dir", type=Path, default=HERE, help="Checkout whose app.py runs the analysis.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Analyze cases and save each result under <case>/runs/.")
    run.add_argument("--cases", nargs="*")
    run.add_argument("--repeat", type=int, default=1, help="Independent runs per case, for stability.")
    run.add_argument("--label", help="Name for these runs (default: the app checkout's git revision).")
    run.add_argument("--resume", action="store_true", help="Count runs already saved under this label toward --repeat.")
    run.add_argument("--model", help="Model (default: the app's CODEX_MODEL).")
    run.add_argument("--effort", help="Reasoning strength (default: the app's CODEX_REASONING_EFFORT).")
    run.set_defaults(handler=command_run)

    setup = commands.add_parser("setup", help="Create case folders from a folder of reports, asking for each case's details.")
    setup.add_argument("--from", dest="source", type=Path, required=True, help="Folder that holds the reports.")
    setup.set_defaults(handler=command_setup)

    check = commands.add_parser("check", help="Validate every case and make one small model call.")
    check.add_argument("--cases", nargs="*")
    check.add_argument("--model", help="Model to check (default: the app's CODEX_MODEL).")
    check.set_defaults(handler=command_check)

    for name, handler, help_text in (
        ("score", command_score, "Score saved runs against the withheld labels."),
        ("compare", command_compare, "Compare two labeled sets of runs on the cases they share."),
        ("fit-calibration", command_fit_calibration, "Fit a temperature from saved runs."),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--cases", nargs="*")
        sub.add_argument("--confirmed-only", action="store_true", help="Skip cases whose label is only a belief.")
        if name == "compare":
            sub.add_argument("--baseline", required=True, help="Run label to compare from.")
            sub.add_argument("--candidate", required=True, help="Run label to compare to.")
        else:
            sub.add_argument("--label", help="Only use runs with this label.")
            sub.add_argument("--runs", choices=("latest", "all"), default="latest" if name == "score" else "all")
        if name == "fit-calibration":
            sub.add_argument("--output", type=Path, required=True)
        else:
            sub.add_argument("--json", action="store_true")
        sub.set_defaults(handler=handler)

    args = parser.parse_args(argv)
    if getattr(args, "repeat", 1) < 1:
        parser.error("--repeat must be at least 1.")
    global app
    app = _load_app(args.app_dir)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
