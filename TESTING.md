# TESTING.md — test architecture and verification report

> CITATION BRANCH (prereg rev 3): OpenCitations is the oracle; the agent
> walks two paginated citation indexes one page per capture occasion.
> New test layers for this design: `test_indexes.py` (index plug-in and
> design arithmetic), `test_oracle_opencitations.py` (oracle universe,
> cutoff, self-validation), and the index-walk cases in
> `test_pipeline_e2e.py` — the statistical certificate fires end to end
> through a two-index walk; a partial-overlap world must REFUSE to
> certify; a failed page is retried; the oracle cannot be walked.

Suite: **112 tests (105 offline + 7 opt-in online)**, pyflakes clean. Run
with `pytest` from the repo root. The suite is the gate: it must be green
before any live spend. (The layer table below predates rev 3; file names
ending `exp2s` are now `exp2c`, and the oracle rows describe the sports
branch's NHL driver, which is kept for its tests.)

## Layers

| Layer | File(s) | What it protects |
|---|---|---|
| 1. Typed validators & canonicalization | `test_canon.py` | season/points/name forms; century rollover; span rejection |
| 2. Stopping statistics | `test_frontier.py` | occasions-not-sources (pitfall #1), Û eq. (1), ψ formula, the two conjuncts, cardinality brake, certified-skip |
| 3. Claims engine | `test_claims.py`, `test_bugfix_regressions.py` | two-domain corroboration, echo-one-vote, COUNT closure, gap direction, Δ⁺, conflict refusal, SUM-vs-COUNT contradiction block |
| 4. Reader & gate | `test_extract.py`, `test_fetch_er_conformal.py` | relevance gate, typed + conformal rejection, deny list (no network), ER variants/contamination, cold-start refusal |
| 5. Oracle & truth | `test_oracle_truth.py`, `test_boundaries.py` | NHL seasonTotals filtering (league + gameTypeId), self-validation vs careerTotals, 404/timeout handling, freeze-hash tamper detection, duplicate-key refusal, exact grading math |
| 6. End-to-end pipeline | `test_pipeline_e2e.py`, `test_pipeline_hardening.py`, `test_boundaries.py` | checksum path with exact sums, honest abandonment, post-ER revocation, CWD independence, in-loop denial/paywall, brake→gap-search→closure, Y_CAP, budget/max_steps/economic stops, φ-audit rows, empty-results occasions |
| 7. Properties & fuzz | `test_properties.py` | **live stop test ≡ replay predicate over 400 random states** (the identity Exp 2's offline arms stand on), f-partition, ψ monotonicities, replay ε-monotonicity over random traces, corroboration echo invariance, claims order-invariance, ER normalization invariance, 1500-string canon fuzz |
| 8. Failure injection | `test_failure_injection.py` | 429/5xx retry-then-loud for LLM and Serper, malformed-JSON retry with raw persisted, missing keys/prompts, fetch outcome classifier, **crash→fresh_ledger→rerun yields one clean replayable trajectory**, malformed calibration CSV, ledger run-id isolation |
| 9. Online sanity preflight | `test_online_sanity.py` | opt-in (`RUN_ONLINE=1`): oracle reachable + rows parse, open-web fetch `ok`, deny list blocks the real oracle, Serper/LLM keys live (tiny spend); skipped on normal runs |
| 10. Experiment entry points | `test_exp2s_scripts.py`, `test_notebook_execution.py`, `test_stress_random_worlds.py` | the scripts as real subprocesses (CLI, refusal gates, CSV outputs); **the notebook executed headlessly end to end** (S1/S2 assert inside it); 60-world stress with benign and adversarial arms |

## Stress invariants (checked on every generated world)

I1 a stop always recorded with a known reason · I2 total = Σ resolved ·
I3 adopted values equal truth wherever sources agree · I4 certified live
stops reproduce under replay (S1) · I5 replay monotone in ε, certified ⇒ no
looser arm censored (S2) · I6 full replay context on every measurement row,
`certified` always a string · I7 grading math sane · I8 coverage events = N,
spend ≤ budget + one search · I9 a wrong corroborated COUNT above the
discoverable records **never** certifies (the brake holds).

## Bugs found by testing (all fixed, each with a regression test)

| # | Bug | Found by | Risk if shipped |
|---|---|---|---|
| 1 | crash-resume appended a 2nd trajectory under one run_id | review | silently corrupted replays |
| 2 | `max_steps` exhaustion mislabeled `frontier_exhausted` | review | wrong stop-reason tables |
| 3 | PPG decimals truncated into season totals (28.4→28) | review | fidelity poisoning |
| 4 | conflicting corroborated claims resolved by dict order | review | nondeterministic checksums |
| 5 | SUM checksum could certify against a contradicting COUNT | review | spurious-record certificates |
| 6 | comma-only strings crashed `canonical_points` (`int("")`) | **fuzzer** | live crash mid-spend |
| 7 | duplicate (player, season) truth rows accepted silently | boundary design | corrupt frozen key, last-row-wins |
| 8 | one transient 429/5xx forfeited a whole player run | review | wasted live spend |
| 9 | gate rejections counted then discarded | review | no δ_E audit trail |

## Known exclusions (deliberate)

- **Live HTTP** (Serper, OpenAI, api-web.nhle.com) is mocked; the real wire is
  exercised only in gated live runs. The retry/refusal logic around it is
  fully tested.
- **The ψ formula** is the impl guide's, verified as implemented — its
  *reconciliation with the production finance psi* is a research decision,
  flagged in the notebook and prereg, not a test matter.
- The notebook's live cell is verified to **skip** under the gate; its body
  shares `end_to_end` with the fully tested offline path.
- Defensive "unreachable" lines and the HTMLParser except-pass account for
  the ~2% uncovered.

## Running

```powershell
pytest                       # whole suite (~15 s; includes notebook execution)
pytest -m "not slow"         # skip the notebook-execution test
pytest --cov=webaggs --cov-report=term-missing
```
