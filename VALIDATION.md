# Validation snapshot

Validated on 2026-08-24 with `gpt-5.6-sol` at `xhigh` reasoning. Model outputs are
non-deterministic comparative estimates, not forensic probabilities or proof of
identity. Expected labels were withheld from every model prompt.

## Current blind-test results

| Case | Evidence status | Expected label | Final distribution (leading entries) | Outcome | Elapsed |
| --- | --- | --- | --- | --- | ---: |
| Orbifold | Confirmed by the user | Ya Deng | Ya Deng 35%, Charles Favre 23%, Mingchen Xia 15% | Unique Top-1; unable to determine precisely | 43.4 min |
| Meng–Zhou | Confirmed by the user | Mingchen Xia | Mingchen Xia 34%, Valentino Tosatti 22% | Unique Top-1; unable to determine precisely | 38.9 min |
| LNM–Xia | Confirmed by the user | Charles Favre | Charles Favre 65%, Sébastien Boucksom 26% | Unique Top-1; leading but not precise | 35.8 min |
| Su–Lewicka X | Strong submitting-author belief; not independently confirmed | Marta Lewicka | Marta Lewicka 21%, László Székelyhidi Jr. 21%, Pakzad 17%, no listed candidate 17% | Tied Top-1; unable to determine | 26.4 min |

The three confirmed-label cases have 3/3 unique Top-1 accuracy, zero false precise
claims, a mean expected-label probability of 44.7%, and a mean true-label margin of
21.0 percentage points. If the weaker Su–Lewicka belief label is included, unique
Top-1 accuracy is 3/4, Top-1-including-ties is 4/4, and the tie rate is 1/4. A tie is
never counted as a unique Top-1 win.

The method-specific reweighting changed the Orbifold separation between Ya Deng and
Mingchen Xia from 1.5 to 20 percentage points. The final report still abstained from
a precise attribution because expertise ablation left competing evidence and no rare
error fingerprint was verified.

## Validation properties

- A verified manuscript author is excluded before stylistic scoring. Exact name
  equality alone is never enough; the LNM run used multiple authoritative identity
  sources and removed Mingchen Xia without merging unrelated namesakes.
- Direct author-linked writing evidence normally receives 45–65% of the comparative
  budget. Academic fit receives 8–20%; academic fit plus citation/network proximity
  is capped at 30% when usable writing evidence exists.
- A verified, unusually narrow pre-report method trajectory may contribute a bounded
  10–15-point tie-break only when direct writing evidence leaves finalists within
  about five points and does not favor the less-specific candidate. Such dependence
  cannot by itself produce a precise attribution.
- Repeated spelling and grammar errors across independent solo works receive high
  weight. Isolated typos, coauthored prose, generic terminology, fame, and broad field
  overlap do not.
- Optional phases stop at their phase deadlines; incomplete calls are discarded, the
  last complete evidence packet is preserved, and analysis has a one-hour hard limit.
- The user can disable named discovery outside the supplied candidate list while the
  unnamed no-listed-candidate probability remains available.

## Verification

The source package passes 75 automated tests, Python bytecode compilation, frontend
JavaScript syntax checks, and `git diff --check`. The only test-suite warning is a
harmless Python `ResourceWarning` from the deliberately mocked HTTP 429 path.
