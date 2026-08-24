# FindReferee

FindReferee is a local, open-source assistant for investigating who may have written a referee report or other document. Give it a target file and a candidate shortlist, and it returns an uncertainty-aware probability distribution, a concise result, and an expandable evidence report. If the shortlist is empty, it can build a cautious public candidate shortlist before attribution.

It can also compare several documents for likely common authorship.

> FindReferee is an exploratory research aid, not a forensic identification system. Its probabilities are model estimates, not calibrated frequencies or proof of identity.

## Highlights

- PDF, TXT, Markdown, TeX, and pasted-text input
- Several target documents combined into one analysis
- Optional underlying manuscript and manuscript-author field for referee-role and subject-context checks
- Mandatory self-referee exclusion: verified manuscript authors are removed before style scoring and shown only in a separate eligibility record, never in the candidate ranking; same-name strings are never enough for exclusion
- Candidate aliases using `/`, editable local author cards, and private known-author corpora
- A candidate-scope switch: allow source-backed names beyond the shortlist, or keep every named result inside it while retaining an anonymous “No listed candidate” probability
- Public candidate-background and solo-paper research, with reusable local caching
- Language profile, grammar and spelling-error fingerprints, review voice, terminology, capped academic fit, and English-fluency evidence
- Lower prose weight for 2026-and-later work that may have been AI-polished; stronger weight for original pre-2026 solo work
- PDF metadata and low-weight TeX-habit analysis
- Time-truncated direct and second-order citation-network candidate priors, with prior-collaborator penalties
- Offline RapidFuzz overlap checks, character n-grams, Burrows Delta, function-word Delta, review-voice distance, and Lingua language detection
- Multi-pass review with targeted follow-up rounds when finalists remain close
- Explicit “No listed candidate” probability and same-name identity questions
- Minimal and detailed interface modes, animated progress, stable clue display, and completed-run duration
- Downloadable JSON and self-contained HTML reports
- A post-result, explicit-consent email-draft export that attaches the user-supplied files, input manifest, and complete result for voluntary benchmark sharing with the maintainer. It creates a local unsent `.eml` draft; the user reviews and sends it. Shared material is used only to improve FindReferee and is not distributed to third parties.
- Signed-in Codex subscription by default; OpenAI API key only as an optional fallback

## How it works

The analysis pipeline separates observable evidence from final scoring:

1. Extract the target text, document metadata, and stable language/domain features.
2. Resolve the underlying manuscript, verify its author identities, and remove any confirmed manuscript author from the referee pool. Name or initial collisions remain eligible until source-backed ORCID, author ID, official profile, affiliation, field, and publication evidence distinguishes the person from a namesake. Automatic exclusion requires two distinct public sources: normally an authoritative manuscript byline and a separate identity source. A single page never triggers exclusion.
3. Trace a year-bounded citation graph. Direct citations are stronger candidate priors than second-order citations; neither is writing evidence.
4. Collect known private samples and exact-name solo-author public works when available.
5. Run deterministic overlap, stylometry, report-voice, and surface-language checks.
6. Ask independent model passes to evaluate direct writing evidence separately from academic and network proximity.
7. Run focused expertise-ablation and counter-evidence rounds for close finalists.
8. Adjudicate the evidence, preserve a “none of the above” alternative, and render concise and detailed reports.

When usable original prose exists, direct author-linked writing evidence receives the main evidence budget. Academic fit and citation/network proximity remain one correlated indirect family and are capped together at 30%. Broad field overlap stays weak. If direct writing evidence leaves two finalists essentially tied, authoritative pre-report evidence that one candidate repeatedly worked on the report’s narrow problem-and-method chain may supply a bounded tie-breaker; this can produce a useful lead but cannot by itself support a precise or high-confidence identification. Coauthored prose is used for background, not to distinguish which collaborator wrote a report. Cross-genre stylometry disagreement is retained as counterevidence but cannot veto an independently repeated rare-error fingerprint. Review-voice distance cannot count as a direct family from a single reference report; it requires at least two independent review-like samples and low within-author dispersion. A high model-reported private-corpus fit is ignored unless at least two user-confirmed private files were actually supplied for that leader.

The workflow reports high-level stages and evidence clues, but does not expose private model chain-of-thought.

## Requirements

- Python 3.10 or newer
- A locally installed Codex CLI signed in with the user's own ChatGPT/Codex account
- Internet access for public-source research and arXiv collection

Each computer uses its own Codex login. Never copy Codex credential files between computers.

## Quick start

### macOS launcher

1. Install and sign in to Codex once with `codex login`.
2. Double-click **Start FindReferee.command**.
3. Keep the launcher window open while using the app.

The launcher creates the virtual environment, installs dependencies on first use, starts the local server, and opens the app.

### Manual setup

```bash
git clone https://github.com/MingchenXia/FindReferee.git
cd FindReferee
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Using FindReferee

1. Upload one or more target files, or paste text.
2. Enter one candidate per line. Use `/` between alternate names for the same person. Leave the list empty for automatic shortlist discovery.
3. Leave **Explore beyond my candidate list** on to let AI investigate source-backed alternatives, or turn it off to keep every named candidate and source within your shortlist. The app still estimates an unnamed **No listed candidate** probability so it is never forced to select a poor match. When the candidate box is empty, discovery is required and this switch is locked on.
4. Optionally upload the manuscript being reviewed and enter its authors. The app also tries to resolve authors from an arXiv/DOI record or PDF metadata. For ambiguous names, add an ORCID, official profile, affiliation, or publication link in Context. Known private samples may be uploaded for each candidate.
5. Choose a model and reasoning strength. The defaults are `gpt-5.6-sol` and `xhigh`.
6. Select **Let's go!**.

The result begins with an explicit determination status, the top probabilities, and total analysis time. The workflow targets 55 minutes and has a 60-minute hard budget: near the limit it stops additional focused rounds and broad searches, preserves time for adjudication, and can return the last complete review instead of losing the run. “Meaningfully separated leader” requires a stable multi-round lead plus at least two direct writing-evidence families. “Leading candidate, but not precise” reports a useful lead without claiming identification. “Unable to determine” is used when candidates remain close, evidence conflicts, an identity is unresolved, or someone outside the shortlist is at least as plausible as its leader. Expand the detailed report for language profile, candidate-by-candidate evidence, public sources, review-round snapshots, stylometry, metadata, limitations, and uncertainty notes.

## Configuration

Copy `.env.example` values into your shell environment or launcher configuration as needed. Important settings include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AUTHOR_ATTRIBUTION_PROVIDER` | `auto` | Use `codex`, `api`, or automatically prefer Codex. |
| `CODEX_MODEL` | `gpt-5.6-sol` | Default subscription model shown by the app. |
| `CODEX_REASONING_EFFORT` | `xhigh` | `low`, `medium`, `high`, or `xhigh`. |
| `CODEX_TIMEOUT_SECONDS` | `1200` | Guard for each individual Codex model call. |
| `CODEX_TIMEOUT_RETRIES` | `1` | Retry only a timed-out Codex call; login, quota, and other errors still fail immediately. |
| `CODEX_TIMEOUT_RETRY_EFFORT` | `high` | Reasoning strength used for timeout recovery, while the first attempt keeps the user's selection. |
| `AUTHOR_ATTRIBUTION_TARGET_SECONDS` | `3300` | Soft user-experience target (55 minutes); late optional rounds are skipped to protect adjudication time. |
| `AUTHOR_ATTRIBUTION_HARD_SECONDS` | `3600` | End-to-end hard budget (60 minutes); the last complete review is retained if final adjudication reaches it. |
| `CODEX_CLI_PATH` | auto-detected | Explicit local Codex executable path. |
| `OPENAI_API_KEY` | unset | Optional API fallback; not required for Codex subscription use. |
| `OPENAI_MODEL` | `gpt-5-mini` | Model used by the optional API fallback. |
| `AUTHOR_ATTRIBUTION_ADAPTIVE_ROUNDS` | `2` | Maximum targeted finalist rounds, from 0 to 4. Two is the measured default; more repeated rounds were slow and did not improve separation on the private quick-opinion baseline. |
| `AUTHOR_ATTRIBUTION_MIN_MARGIN_GAIN` | `0.03` | Stop after two focused rounds when the candidate margin has not improved by at least this amount. |
| `AUTHOR_ATTRIBUTION_PUBLIC_CORPUS` | `true` | Enable exact-name solo-work collection from arXiv. |
| `AUTHOR_ATTRIBUTION_PUBLIC_FULL_TEXTS` | `8` | Distributed pre-2026/chronological solo-paper excerpts per candidate; capped at 8 to preserve room for careful comparison. |
| `AUTHOR_ATTRIBUTION_CITATION_NETWORK` | `true` | Enable the time-truncated Semantic Scholar citation prior. |
| `SEMANTIC_SCHOLAR_API_KEY` | unset | Optional; the public API normally works without a key. |
| `AUTHOR_ATTRIBUTION_CITATION_CACHE` | system cache | Optional citation-metadata cache directory. |

Model names must be available to the signed-in account or configured API project. If a subscription, quota, model-access, or authentication problem occurs, FindReferee displays the provider error in the interface.

## Privacy and data handling

- The server binds to `127.0.0.1` by default.
- The app does not read, copy, or transmit Codex authentication files.
- Uploaded files are held in memory for the active analysis and are not intentionally written to persistent app storage.
- Private candidate corpora are sent to the selected model for that run, but are not used as public-search queries or shown as public citations.
- Public author cards are saved only in that browser's local storage.
- Public arXiv excerpts may be cached locally to make later runs more reliable and efficient.
- Public citation metadata may be cached locally. Only the public manuscript identifier or title is sent to Semantic Scholar; the private referee-report text is not.
- When public-source search finds zbMATH Open or MathSciNet reviews, they are treated as secondary, genre-shifted comparison prose. Subscription-only MathSciNet review text is never written to public output or the repository.
- OpenAI API requests set `store=False` where supported.

Review the data policies for the model provider and account you choose before analyzing confidential material.

## API

The browser uses background jobs so long analyses do not depend on one long-lived HTTP request.

Start a run:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/start \
  -F 'mode=attribution' \
  -F $'candidates=Candidate One\nCandidate Two' \
  -F 'underlying_authors=Manuscript Author One; Manuscript Author Two' \
  -F 'files=@report.pdf'
```

Poll the returned job identifier:

```bash
curl http://127.0.0.1:8000/api/analyze/status/JOB_ID
```

Completed results include `total_elapsed_seconds`.

## Accuracy notes

Authorship attribution is inherently uncertain. Short, translated, heavily edited, collaborative, formulaic, or AI-polished texts can suppress individual style. Academic fit, citations, and reviewer networks can identify plausible experts but are intentionally capped because they do not identify the writer. PDF metadata can be stale or inherited. TeX habits often belong to templates or collaborators. Deterministic stylometry and report-voice metrics are diagnostics, not probability estimators.

Use the report as structured evidence for further investigation, not as a basis for accusation or consequential decisions without independent verification.

## Development

Run the tests with:

```bash
.venv/bin/python -m unittest discover -s tests
```

`evaluation_metrics.py` provides blind-test unique Top-1 accuracy, a separate Top-1-including-ties rate, tie rate, MRR, true-author margin, log loss, Brier score, entropy, decisive accuracy, and repeated-run Jensen-Shannon stability. A label tied at the highest probability is not counted as a unique Top-1 win. Keep private reports, expected labels, and run artifacts under the ignored `benchmarks/` directory so ground truth is never sent to the model or committed accidentally.

The project deliberately builds on maintained, general-purpose components rather than low-usage end-to-end attribution repositories: RapidFuzz supplies reproducible string matching and Lingua identifies the written language. LanguageTool, spaCy, and neural embedding stacks were evaluated but are not required because they add large downloads, privacy-sensitive services, or topic leakage without a validated gain on the private benchmark.

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

FindReferee is released under the [MIT License](LICENSE).
