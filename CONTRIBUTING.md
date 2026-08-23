# Contributing to FindReferee

Thank you for helping improve FindReferee.

## Before opening a change

- Keep the application interface and documentation in English.
- Do not commit uploaded documents, private corpora, benchmark answers, credentials, or local Codex files.
- Treat model probability outputs as uncertain estimates; avoid claims of forensic certainty.
- Add or update tests for pipeline, parsing, scoring, or privacy-sensitive behavior.

## Local development

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/uvicorn app:app --reload
```

Open a pull request with a short explanation of the problem, the approach, and the checks you ran.
