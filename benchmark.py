"""Run, score, and calibrate FindReferee on private labeled cases.

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
expected label and label status are read solely by the scorer. Usage:

    python benchmark.py run [--cases NAME ...] [--repeat N] [--model M] [--effort E]
    python benchmark.py score [--cases NAME ...] [--runs latest|all] [--confirmed-only] [--json]
    python benchmark.py fit-calibration --output calibration.json [--confirmed-only]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import app
from evaluation_metrics import (
    NO_LISTED_CANDIDATE,
    aggregate_scores,
    fit_temperature,
    probability_distribution,
    repeated_run_stability,
    score_case,
)


DEFAULT_ROOT = app.APP_DIR / "benchmarks"
LABEL_FIELDS = {"expected_author", "label_status"}


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
    return app._parse_upload_payload(path.name, suffix, path.read_bytes())


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
    except Exception as exc:  # Keep the run on record exactly as the web app would.
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
    for folder in _cases(args.root, args.cases):
        try:
            inputs = _analysis_inputs(folder, _read_case(folder), args.model, args.effort)
        except app.HTTPException as exc:
            raise SystemExit(f"{folder.name}: {exc.detail}") from exc
        runs = folder / "runs"
        runs.mkdir(exist_ok=True)
        for repetition in range(args.repeat):
            result = _run_case(folder, inputs)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            path = runs / f"{stamp}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{folder.name}] run {repetition + 1}/{args.repeat} saved to {path}", file=sys.stderr)
    return 0


def _scored_runs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any], str, list[Path]]]:
    """(case name, case, expected label, run files) for every case that has runs."""
    selected = []
    for folder in _cases(args.root, args.cases):
        case = _read_case(folder)
        if args.confirmed_only and case.get("label_status", "confirmed") != "confirmed":
            continue
        run_files = sorted((folder / "runs").glob("*.json"))
        if not run_files:
            print(f"[{folder.name}] no saved runs; skipped", file=sys.stderr)
            continue
        if args.runs == "latest":
            run_files = run_files[-1:]
        selected.append((folder.name, case, str(case["expected_author"]), run_files))
    return selected


def command_score(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    stability: dict[str, Any] = {}
    for name, _case, expected, run_files in _scored_runs(args):
        results = [json.loads(path.read_text(encoding="utf-8")) for path in run_files]
        for path, result in zip(run_files, results):
            distribution = probability_distribution(result)
            if expected not in distribution:
                print(f"[{name}] expected label {expected!r} is not among {sorted(distribution)}", file=sys.stderr)
            score = score_case(result, expected)
            leader = max(distribution, key=distribution.get) if distribution else ""
            rows.append({"case": name, "run": path.stem, "expected": expected, "leader": leader, **score})
        if len(results) > 1:
            stability[name] = repeated_run_stability(results)
    report = {"cases": rows, "aggregate": aggregate_scores(rows), "repeated_run_stability": stability}
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        print(
            f"{row['case']:<24} {row['run']:<24} expected {row['expected']!r}: "
            f"{row['expected_probability']:.0%}, rank {row['expected_rank']}, leader {row['leader']!r}, "
            f"{row['determination_status']}"
        )
    aggregate = report["aggregate"]
    if aggregate.get("case_count"):
        print(
            f"\n{aggregate['case_count']} scored run(s): unique Top-1 {aggregate['top1_accuracy']:.0%}, "
            f"Top-1 including ties {aggregate['top1_including_ties_accuracy']:.0%}, "
            f"MRR {aggregate['mean_reciprocal_rank']:.3f}, mean log loss {aggregate['mean_log_loss']:.3f}, "
            f"false precise claims {aggregate['false_precise_claim_rate']:.0%}"
        )
    for name, values in stability.items():
        print(f"{name}: mean pairwise Jensen-Shannon divergence {values['mean_pairwise_js_divergence']:.4f} over {values['run_count']} runs")
    return 0


def command_fit_calibration(args: argparse.Namespace) -> int:
    pairs = []
    models = set()
    case_names = set()
    for name, _case, expected, run_files in _scored_runs(args):
        for path in run_files:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("review_strategy") == "safe timeout fallback":
                continue  # A non-determination carries no model distribution to calibrate.
            models.add(str(result.get("model", "")))
            case_names.add(name)
            pairs.append((result, expected))
    if not pairs:
        raise SystemExit("No completed runs are available to fit a calibration.")
    if len(models) > 1:
        raise SystemExit(f"Runs come from several models ({', '.join(sorted(models))}); fit one model at a time with --cases.")
    fitted = fit_temperature(pairs)
    # Repeated runs of one case are correlated, so the app's minimum counts distinct cases.
    fitted.update({"case_count": len(case_names), "run_count": len(pairs)})
    fitted.update({"model": models.pop(), "fitted_at": datetime.now(timezone.utc).isoformat(), "no_listed_label": NO_LISTED_CANDIDATE})
    args.output.write_text(json.dumps(fitted, indent=2), encoding="utf-8")
    print(json.dumps(fitted, indent=2))
    if fitted["case_count"] < app.MIN_CALIBRATION_CASES:
        print(
            f"Warning: {fitted['case_count']} run(s) is below the {app.MIN_CALIBRATION_CASES} the app requires; "
            "the file will be ignored until more labeled runs are added.",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Folder that contains the case folders.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Analyze cases and save each result under <case>/runs/.")
    run.add_argument("--cases", nargs="*")
    run.add_argument("--repeat", type=int, default=1, help="Independent runs per case, for stability.")
    run.add_argument("--model", default=app.CODEX_MODEL)
    run.add_argument("--effort", default=app.CODEX_REASONING_EFFORT)
    run.set_defaults(handler=command_run)

    for name, handler, help_text in (
        ("score", command_score, "Score saved runs against the withheld labels."),
        ("fit-calibration", command_fit_calibration, "Fit a temperature from saved runs."),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--cases", nargs="*")
        sub.add_argument("--runs", choices=("latest", "all"), default="latest" if name == "score" else "all")
        sub.add_argument("--confirmed-only", action="store_true", help="Skip cases whose label is only a belief.")
        if name == "score":
            sub.add_argument("--json", action="store_true")
        else:
            sub.add_argument("--output", type=Path, required=True)
        sub.set_defaults(handler=handler)

    args = parser.parse_args(argv)
    if getattr(args, "repeat", 1) < 1:
        parser.error("--repeat must be at least 1.")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
