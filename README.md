# FindReferee

FindReferee is an open-source assistant for investigating who may have written a referee report or other document. Its interface is hosted at [mingchenxia.github.io/FindReferee](https://mingchenxia.github.io/FindReferee/), while a small local companion keeps document handling and Codex subscription access on the user's own computer. Give it a target file and a candidate shortlist, and it returns an uncertainty-aware probability distribution, a concise result, and an expandable evidence report. If the shortlist is empty, it can build a cautious public candidate shortlist before attribution.

It can also compare several documents for likely common authorship.

> FindReferee is an exploratory research aid, not a forensic identification system. Its probabilities are model estimates, not calibrated frequencies or proof of identity.

## Highlights

- PDF, TXT, Markdown, TeX, and pasted-text input
- Several target documents combined into one analysis
- Optional underlying manuscript for referee-role and subject-context checks
- Candidate aliases using `/`, editable local author cards, and private known-author corpora
- Public candidate-background and solo-paper research, with reusable local caching
- Language profile, grammar and spelling-error fingerprints, terminology, academic fit, and English-fluency evidence
- Lower prose weight for 2026-and-later work that may have been AI-polished; stronger weight for original pre-2026 solo work
- PDF metadata and low-weight TeX-habit analysis
- Offline RapidFuzz overlap checks, character n-grams, Burrows Delta, function-word Delta, and Lingua language detection
- Multi-pass review with targeted follow-up rounds when finalists remain close
- Explicit “No listed candidate” probability and same-name identity questions
- Minimal and detailed interface modes, animated progress, stable clue display, and completed-run duration
- Downloadable JSON and self-contained HTML reports
- Signed-in Codex subscription by default; OpenAI API key only as an optional fallback

## How it works

The analysis pipeline separates observable evidence from final scoring:

1. Extract the target text, document metadata, and stable language/domain features.
2. Collect known private samples and exact-name solo-author public works when available.
3. Run deterministic overlap, stylometry, and surface-language checks.
4. Ask independent model passes to evaluate linguistic, academic, provenance, and reviewer-role evidence.
5. Run focused counter-evidence rounds for close finalists.
6. Adjudicate the evidence, preserve a “none of the above” alternative, and render concise and detailed reports.

The workflow reports high-level stages and evidence clues, but does not expose private model chain-of-thought.

## Requirements

- Python 3.10 or newer
- A locally installed Codex CLI signed in with the user's own ChatGPT/Codex account
- Internet access for public-source research and arXiv collection

Each computer uses its own Codex login. Never copy Codex credential files between computers.

## Quick start

### Hosted interface with the macOS companion

1. Open [FindReferee online](https://mingchenxia.github.io/FindReferee/).
2. Download and extract the companion from the page, or clone this repository.
3. Install and sign in to Codex once with `codex login`.
4. Double-click **Start FindReferee.command** and keep its window open while using the site.

The launcher creates the virtual environment, installs dependencies on first use, starts the loopback-only companion, and opens the hosted interface. If the browser asks whether the site may access devices on the local network, allow it so the page can reach the companion on this computer.

### Manual setup

```bash
git clone https://github.com/MingchenXia/FindReferee.git
cd FindReferee
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open [FindReferee online](https://mingchenxia.github.io/FindReferee/). The locally served interface at [http://127.0.0.1:8000](http://127.0.0.1:8000) remains available as an offline fallback.

## Using FindReferee

1. Paste text or upload one or more target files.
2. Enter one candidate per line. Use `/` between alternate names for the same person. Leave the list empty for automatic shortlist discovery.
3. Optionally upload the manuscript being reviewed and known private samples for each candidate.
4. Choose a model and reasoning strength. The defaults are `gpt-5.6-sol` and `xhigh`.
5. Select **Analyze automatically with Codex**.

The result begins with the top probabilities and total analysis time. Expand the detailed report for language profile, candidate-by-candidate evidence, public sources, review-round snapshots, stylometry, metadata, limitations, and uncertainty notes.

## Configuration

Copy `.env.example` values into your shell environment or launcher configuration as needed. Important settings include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AUTHOR_ATTRIBUTION_PROVIDER` | `auto` | Use `codex`, `api`, or automatically prefer Codex. |
| `CODEX_MODEL` | `gpt-5.6-sol` | Default subscription model shown by the app. |
| `CODEX_REASONING_EFFORT` | `xhigh` | `low`, `medium`, `high`, or `xhigh`. |
| `CODEX_TIMEOUT_SECONDS` | `1200` | Guard for each individual Codex model call. |
| `CODEX_CLI_PATH` | auto-detected | Explicit local Codex executable path. |
| `OPENAI_API_KEY` | unset | Optional API fallback; not required for Codex subscription use. |
| `OPENAI_MODEL` | `gpt-5-mini` | Model used by the optional API fallback. |
| `AUTHOR_ATTRIBUTION_ADAPTIVE_ROUNDS` | `3` | Maximum targeted finalist rounds, from 0 to 4. |
| `AUTHOR_ATTRIBUTION_PUBLIC_CORPUS` | `true` | Enable exact-name solo-work collection from arXiv. |
| `FINDREFEREE_ALLOWED_ORIGINS` | unset | Optional comma-separated additional hosted frontend origins. |

Model names must be available to the signed-in account or configured API project. If a subscription, quota, model-access, or authentication problem occurs, FindReferee displays the provider error in the interface.

## Privacy and data handling

- The server binds to `127.0.0.1` by default.
- The hosted page contains no model credential and performs no analysis by itself; it sends requests only to the companion on the same computer.
- Browser requests are restricted to the official GitHub Pages origin and configured local origins. Mutating requests also require a random session token that changes whenever the companion restarts.
- The app does not read, copy, or transmit Codex authentication files.
- Uploaded files are held in memory for the active analysis and are not intentionally written to persistent app storage.
- Private candidate corpora are sent to the selected model for that run, but are not used as public-search queries or shown as public citations.
- Public author cards are saved only in that browser's local storage.
- Public arXiv excerpts may be cached locally to make later runs more reliable and efficient.
- OpenAI API requests set `store=False` where supported.

Review the data policies for the model provider and account you choose before analyzing confidential material.

## API

The browser uses background jobs so long analyses do not depend on one long-lived HTTP request.

Start a run:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/start \
  -F 'mode=attribution' \
  -F $'candidates=Candidate One\nCandidate Two' \
  -F 'files=@report.pdf'
```

Poll the returned job identifier:

```bash
curl http://127.0.0.1:8000/api/analyze/status/JOB_ID
```

Completed results include `total_elapsed_seconds`.

## Accuracy notes

Authorship attribution is inherently uncertain. Short, translated, heavily edited, collaborative, formulaic, or AI-polished texts can suppress individual style. Academic fit and reviewer-role evidence can be informative but may also reflect a broad research community. PDF metadata can be stale or inherited. TeX habits often belong to templates or collaborators. Deterministic stylometry metrics are diagnostics, not probability estimators.

Use the report as structured evidence for further investigation, not as a basis for accusation or consequential decisions without independent verification.

## Development

Run the tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

FindReferee is released under the [MIT License](LICENSE).
